"""One main-thread harness episode per subprocess, with a public event projection."""
from __future__ import annotations
from dataclasses import asdict, replace
import json
import os
import signal
import sys
import time
import threading
from pathlib import Path
from agent_harness.artifacts import ArtifactStore
from agent_harness.code_tools import code_understanding_tools
from agent_harness.contracts import FinalAnswer, ToolCall, ResearchRequest, RunLimits, ModelResponse, Usage
from agent_harness.registry import ToolRegistry
from agent_harness.runner import AgentRunner
from agent_harness.sandbox import ExecutionEnvironment, ModalSandbox
from product_api.catalog import open_repository, read
from product_api.model import ProductModel
from product_api.limits import RUN_LIMITS, MAX_REPAIRS


def atomic_write(path, value):
    path = Path(path)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False))
    tmp.replace(path)


def project(event):
    kind = event['kind']
    common = {k: event[k] for k in ('sequence', 'kind', 'elapsed_seconds')}
    fields = {'started': ('tools', 'model_id', 'limits'), 'model_request': (),
        'model_response': (), 'tool_call': ('call_id', 'name', 'arguments'),
        'tool_observation': ('call_id', 'name', 'observation', 'artifact_id'),
        'submission': (), 'finished': ('termination_reason', 'metrics'),
        'error': (), 'invalid_action': ('usage',), 'invalid_submission': (),
        'budget_limit': ('limit',), 'context_compacted': ()}
    fields['submission_rejected'] = ('feedback',)
    fields['action_rejected'] = ('feedback', 'attempt')
    for key in fields.get(kind, ()):
        if key in event:
            common[key] = event[key]
    if kind in ('error', 'invalid_action', 'invalid_submission'):
        common['error'] = {'error': 'The provider or environment could not complete this run.',
            'invalid_action': 'The model returned an invalid action.',
            'invalid_submission': 'The final answer did not match the required format.'}[kind]
    if kind == 'tool_observation':
        obs = common['observation'] = dict(common['observation'])
        if obs.get('error'):
            obs['error'] = 'Tool failed. Check the input, pinned source, or sandbox availability.'
    if kind == 'model_response':
        common['usage'] = event['response'].get('usage')
    return common


class PublicStore(ArtifactStore):
    def __init__(self, root, public_path):
        super().__init__(root)
        self.public_path = Path(public_path)
        self.observed = {}
        self.harness = 'structured'
        self.bash_observed = False

    def append_event(self, episode_id, event):
        super().append_event(episode_id, event)
        if event['kind'] == 'tool_observation' and event['name'] == 'bash' and event['observation']['status'] == 'ok':
            self.bash_observed = True
        if event['kind'] == 'tool_observation' and event['name'] == 'read_file':
            obs = event['observation']
            content = obs.get('content') or {}
            # Only complete delivered lines are citation evidence, not hidden/truncated tails.
            if obs['status'] == 'ok' and isinstance(content, dict) and 'text' in content:
                lines = content['text'].splitlines()
                if obs.get('truncated') and lines:
                    lines = lines[:-1]
                start = content['start_line']
                self.observed.setdefault(content['path'], set()).update(range(start, start + len(lines)))
        with self.public_path.open('a') as stream:
            stream.write(json.dumps(project(event), allow_nan=False) + '\n')
            stream.flush()

    def validate_answer(self, value):
        if not isinstance(value, dict) or not isinstance(value.get('text'), str) or not isinstance(value.get('citations'), list):
            raise ValueError('Answer must contain text and citations')
        citations = []
        for ref in value['citations']:
            if not isinstance(ref, dict):
                continue
            path, start, end = ref.get('path'), ref.get('start_line'), ref.get('end_line')
            valid = isinstance(path, str) and type(start) is int and type(end) is int and 1 <= start <= end <= start + 500
            valid = valid and set(range(start, end + 1)).issubset(self.observed.get(path, set()))
            citations.append({'path': str(path or ''), 'start_line': start if type(start) is int else 0,
                'end_line': end if type(end) is int else 0, 'verified': bool(valid)})
            claim = ref.get('claim')
            if isinstance(claim, str) and claim.strip() and claim in value['text']:
                citations[-1]['claim'] = claim
        return {'text': value['text'], 'citations': citations}

    def answer_feedback(self, value):
        try:
            answer = self.validate_answer(value)
        except (ValueError, TypeError):
            return 'Return an answer with text and source citations after investigating with tools.'
        if self.harness == 'bash':
            return None if self.bash_observed else 'Investigate the repository with bash before answering.'
        if not any(self.observed.values()):
            return 'No source has been read. Discover relevant files using search_code or list_files, then call read_file before answering.'
        if not answer['citations'] or not all(c['verified'] for c in answer['citations']):
            return 'Cite only complete line ranges returned by read_file. Read any missing evidence and correct the citations before answering.'
        return None


from dataclasses import dataclass

@dataclass
class PreviewShellResult:
    stdout: str = 'SCRIPTED PREVIEW — no command was executed.'
    stderr: str = ''
    exit_code: int = 0
    truncated: bool = False

class PreviewEnvironment:
    python_executable = 'python'
    test_runner = ('python', '-m', 'pytest')

    def run(self, repository, command, stdin=None):
        return {'stdout': 'SCRIPTED PREVIEW — no code was executed.\nValidationError: Instance is frozen [type=frozen_instance]\n',
                'stderr': '', 'exit_code': 0, 'timed_out': False, 'truncated': False, 'synthetic': True}


class PreviewModel:
    model_id = 'scripted-preview'
    def __init__(self, repo, execution, variant=None):
        paths = repo.files()
        path = 'pydantic/main.py' if 'pydantic/main.py' in paths else ('responses/__init__.py' if 'responses/__init__.py' in paths else paths[0])
        lines = repo.text(path)[0].splitlines()
        start = next((i + 1 for i, line in enumerate(lines) if 'def _check_frozen' in line), 1)
        count = min(20, len(lines) - start + 1)
        self.actions = [ToolCall('list_files', {'glob': '*.py', 'limit': 20}),
            ToolCall('search_code', {'query': 'frozen' if 'pydantic' in path else 'class', 'glob': path}),
            ToolCall('read_file', {'path': path, 'start_line': start, 'line_count': count})]
        if variant == 'right':
            self.actions.insert(2, ToolCall('list_files', {'glob': '*.py', 'limit': 5}))
        if execution:
            self.actions.append(ToolCall('python_probe', {'code': "from pydantic import BaseModel, ConfigDict\n\nclass User(BaseModel):\n    model_config = ConfigDict(frozen=True)\n    name: str\n\nuser = User(name='Ada')\ntry:\n    user.name = 'Grace'\nexcept Exception as exc:\n    print(type(exc).__name__, str(exc))"}))
        self.actions.append(FinalAnswer({'text': 'This is a scripted interface preview, not a model-generated answer. The agent searched the repository and read the pinned source shown below. The execution card contains synthetic output; no code was executed. Select a live model and ask a question to run real inference.',
            'citations': [{'path': path, 'start_line': start, 'end_line': start + count - 1}]}))

    def generate(self, *args, **kwargs):
        time.sleep(0.55)
        return ModelResponse(self.actions.pop(0), usage=Usage(0, 0, 0))


def main(run_dir):
    directory = Path(run_dir)
    config = read(directory / 'config.json')
    status_path = directory / 'result.json'
    started = time.monotonic()
    parent = os.getppid()
    def watch_parent():
        while True:
            time.sleep(1)
            if os.getppid() != parent:
                os.kill(os.getpid(), signal.SIGTERM)
                return
    def stop(_signal, _frame):
        raise KeyboardInterrupt('Run cancelled')
    signal.signal(signal.SIGTERM, stop)
    threading.Thread(target=watch_parent, daemon=True).start()
    bash_sandbox = None
    try:
        repo, manifest = open_repository(config['repo'])
        store = PublicStore(directory / 'episodes', directory / 'events.jsonl')
        harness = config.get('harness', 'structured')
        store.harness = harness
        execution = config['repo']['execution'] and harness != 'bash'
        resources = {'repository': repo}
        if execution:
            if config['preview']:
                resources['sandbox'] = PreviewEnvironment()
            else:
                environment = ExecutionEnvironment(backend=ModalSandbox(), manifest=manifest)
                environment.verify(repo)
                resources['sandbox'] = environment
        model_config = {**config['model'], 'repository': {
            'name': config['repo']['name'], 'commit': config['repo']['commit']}}
        if config.get('prior_context'):
            prior = config['prior_context']
            excerpts = []
            for citation in (prior.get('answer') or {}).get('citations', [])[:3]:
                if not citation.get('verified'): continue
                path, start, end = citation['path'], citation['start_line'], citation['end_line']
                end = min(end, start + 79)
                text, _ = repo.text(path)
                lines = text.splitlines()[start-1:end]
                excerpt = '\n'.join(lines)
                if sum(len(e['text']) for e in excerpts) + len(excerpt) > 6000: continue
                excerpts.append(dict(path=path, start_line=start, end_line=start+len(lines)-1, text=excerpt))
                store.observed.setdefault(path,set()).update(range(start,start+len(lines)))
            model_config['prior_context'] = {'question': prior['question'],
                'answer': (prior.get('answer') or {}).get('text','')[:2000], 'source_excerpts': excerpts}
        model = PreviewModel(repo, execution, config.get('side')) if config['preview'] else ProductModel(model_config)
        if harness == 'bash':
            from product_api.harness import BashTool, create_bash_sandbox
            if config['preview']:
                from types import SimpleNamespace
                bash_sandbox = SimpleNamespace(execute=lambda argv: PreviewShellResult(), close=lambda: None)
                model.actions = [ToolCall('bash', {'command':'ls'}), FinalAnswer({'text':'Scripted bash preview. No commands were executed.', 'citations':[]})]
            else:
                bash_sandbox = create_bash_sandbox(repo, manifest, directory, config['id'], config['limits'])
            resources['bash_sandbox'] = bash_sandbox
            tools = [BashTool()]
        else:
            tools = code_understanding_tools(include_execution=execution)
        if config.get('permitted_tools') is not None:
            tools = [t for t in tools if t.spec.name in config['permitted_tools']]
        for tool in tools:
            tool.spec = replace(tool.spec, max_output_bytes=3500 if harness == 'bash' else 10000)
        runner = AgentRunner(model, ToolRegistry(tools), store,
            output_validator=store.validate_answer, submission_feedback=store.answer_feedback,
            max_action_repairs=config.get('max_action_repairs', MAX_REPAIRS),
            max_submission_repairs=config.get('max_submission_repairs', MAX_REPAIRS))
        request = ResearchRequest(config['id'], config['question'], resources,
            limits=RunLimits(**config.get('limits', asdict(RUN_LIMITS))))
        result = runner.run(request, episode_id=config['id'])
        answer = store.validate_answer(result.submission) if result.submission is not None else None
        atomic_write(status_path, {'status': result.termination_reason, 'answer': answer,
                                  'metrics': result.metrics})
    except KeyboardInterrupt:
        atomic_write(status_path, {'status': 'interrupted' if os.getppid() != parent else 'cancelled', 'answer': None,
                                  'metrics': {'elapsed_seconds': time.monotonic() - started}})
    except Exception:
        # Provider exceptions may include credentials; expose neither raw text nor headers.
        atomic_write(status_path, {'status': 'infrastructure_error', 'answer': None,
            'error': 'Run setup failed. Check pinned source, environment readiness, and provider configuration.',
            'metrics': {'elapsed_seconds': time.monotonic() - started}})

    finally:
        if bash_sandbox is not None:
            bash_sandbox.close()

if __name__ == '__main__':
    main(sys.argv[1])
