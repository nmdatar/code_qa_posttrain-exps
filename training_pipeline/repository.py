"""Strict repository adapter; private task/reference material remains on the host."""
from dataclasses import asdict
import hashlib
import json
import secrets
import subprocess
from pathlib import Path
from agent_harness.modal_backend import ModalSandboxBackend, SandboxLimits
from agent_harness.repository_tools import command
from qa_eval.harness import EpisodeRecorder
from qa_eval.grading import evaluate
from qa_eval.schema import validate_task, validate, SUBMISSION, EXPERIMENT
from qa_eval.deterministic import snapshot
from agent_harness.images import validate_manifest
from .contracts import VerificationResult, ConfigurationError
from .storage import atomic_json, digest

TOOLS = {'list_files', 'search_code', 'read_file', 'python_probe'}


def validate_repository_task(row):
    task, experiment, environment = row['task'], row['experiment'], row['environment']
    validate_task(task)
    validate(experiment, EXPERIMENT)
    validate_manifest(environment)
    if task['split'] != row['split'] or task['id'] != row['id'] or task['repository']['family_id'] != row['family_id'] or task['lineage_id'] != row['lineage_id']:
        raise ConfigurationError('Task identity/split mismatch')
    if not task['human_reviewed'] or task['gold_status'] != 'accepted':
        raise ConfigurationError('Strict verifier requires reviewed gold')
    if environment['commit'] != task['repository']['commit']:
        raise ConfigurationError('Environment commit mismatch')
    snapshot(row['snapshot_root'], task['repository']['commit'])
    if not experiment['frozen'] or experiment['training_judge_family'] == experiment['evaluation_judge_family']:
        raise ConfigurationError('Frozen experiment and independent judges required')
    from qa_eval.security import digest as qa_digest
    if {'id': task['id'], 'task_hash': qa_digest(task)} not in experiment['task_manifest']:
        raise ConfigurationError('Task absent from frozen experiment')
    if not set(task['permitted_tools']) <= TOOLS:
        raise ConfigurationError('Unsupported repository tools')


class RepositoryFactory:
    reward_version = 'strict-repository-v1'

    def __init__(self, config, root, backend_factory=ModalSandboxBackend, judge_call=None):
        self.config, self.root, self.backend_factory = config, Path(root), backend_factory
        self.identity = 'repository-' + digest(config['environment'])
        self.key = secrets.token_bytes(32)
        self.judge_call = judge_call

    def create(self, task, episode_id, trajectory_path):
        limits = self.config['limits']
        budget = task['task']['budgets']
        sandbox_limits = SandboxLimits(max_tool_calls=max(1, min(limits['max_tool_calls'], budget['max_tool_calls'])),
            max_output_bytes=limits['max_tool_output_bytes'],
            lifetime_seconds=min(limits['latency_seconds'], int(budget['latency_seconds'])))
        backend = self.backend_factory(task['environment'], self.root / 'sandbox-events', limits=sandbox_limits)
        sandbox = backend.create(episode_id)
        try:
            return RepositoryEpisode(task, sandbox, self, episode_id, trajectory_path)
        except BaseException:
            sandbox.close('infrastructure_error')
            raise


class RepositoryEpisode:
    def __init__(self, row, sandbox, factory, episode_id, trajectory_path):
        self.row, self.task, self.sandbox, self.factory = row, row['task'], sandbox, factory
        catalog = subprocess.run(['git', '-C', row['snapshot_root'], 'ls-tree', '-r', '--name-only', '-z', self.task['repository']['commit']], check=True, capture_output=True, timeout=30).stdout
        self.source_files = [p for p in catalog.decode().split('\0') if p]
        self.recorder = EpisodeRecorder(self.task, row['experiment']['id'], episode_id, str(trajectory_path))
        self.limits = {'max_tool_calls': self.task['budgets']['max_tool_calls'],
                       'max_output_tokens': self.task['budgets']['max_output_tokens'],
                       'latency_seconds': self.task['budgets']['latency_seconds']}
        public = {k: self.task[k] for k in ('id', 'question', 'repository', 'permitted_tools', 'budgets')}
        protocol = ('Investigate the pinned repository. Return one JSON action per turn: '
            '{"tool":"read_file","arguments":{"path":"...","start_line":1,"end_line":30}}, '
            'or list_files(glob,offset), search_code(query,glob), python_probe(code), with the same tool/arguments envelope. '
            'Final: {"answer":{"schema_version":"1.0","task_id":"...","text":"...",'
            '"citations":[{"id":"src","path":"...","start_line":1,"end_line":2,"file_sha256":"..."}],"diagram":null}}. '
            'Use hashes returned by read_file. Cite evidence and make only supported claims.')
        self.messages = [{'role': 'system', 'content': protocol}, {'role': 'user', 'content': json.dumps(public)}]

    def usage(self, input_tokens, output_tokens):
        self.recorder.token_received()
        self.recorder.usage(input_tokens, output_tokens, cost=None)

    def step(self, action):
        if set(action) == {'answer'}:
            validate(action['answer'], SUBMISSION)
            if action['answer']['task_id'] != self.task['id'] or len(json.dumps(action['answer']).encode()) > self.task['budgets']['max_submission_bytes']:
                raise ValueError('Invalid answer binding or size')
            return True, action['answer']
        if set(action) != {'tool', 'arguments'} or action['tool'] not in self.task['permitted_tools']:
            raise ValueError('Tool is not permitted')
        argv = command(action['tool'], action['arguments'], self.source_files)
        result = self.recorder.tool(action['tool'], lambda: self.sandbox.execute(argv))
        if action['tool'] == 'python_probe':
            self.recorder.record['execution_records'].append({'id': 'exec-' + str(len(self.recorder.record['execution_records'])),
                'command': json.dumps(argv), 'exit_code': result.exit_code,
                'stdout_sha256': hashlib.sha256(result.stdout.encode()).hexdigest()})
        return False, asdict(result)

    def verify(self, trajectory):
        submission = trajectory.submission or {'schema_version': '1.0', 'task_id': self.task['id'],
                                              'text': '', 'citations': [], 'diagram': None}
        envelope = self.recorder.finish(submission, self.factory.key, trajectory.termination)
        role = 'training' if self.task['split'] == 'train' else 'evaluation'
        report, _ = evaluate(self.task, submission, envelope, self.row['experiment'],
            self.row['snapshot_root'], self.factory.key, role=role, call=self.factory.judge_call)
        # Grade records are private, never stored in policy-visible observations.
        atomic_json(self.factory.root / 'private' / (trajectory.episode_id + '.grade.json'), report)
        return VerificationResult('unresolved' if report['tier'] == 'unresolved' else 'resolved',
            report['reward'], digest({'contract': report['scoring_contract_hash'], 'judge': self.row['experiment']['judge_version']}),
            report['reasons'], {'tier': report['tier'], 'coverage': report['coverage'], 'judge_role': role})

    def close(self):
        self.sandbox.close()
