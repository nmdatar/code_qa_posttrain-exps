"""Prepare immutable public inputs and separate private grading inputs locally."""
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

from qa_eval.schema import EXPERIMENT, validate, validate_task
from qa_eval.security import digest
from .code_tools import GitRepository, code_understanding_tools
from .contracts import RunLimits
from .remote_contracts import atomic_json, identifier, validate_job, validate_model, job_hash, public_environment


def _git(*args):
    env = {key: value for key, value in os.environ.items() if not key.startswith('GIT_')}
    env.update(GIT_CONFIG_NOSYSTEM='1', GIT_CONFIG_GLOBAL=os.devnull, GIT_TERMINAL_PROMPT='0')
    return subprocess.run(['git', '-c', 'core.hooksPath=/dev/null', '-c', 'protocol.allow=never',
                           '-c', 'protocol.file.allow=always', *map(str, args)],
                          env=env, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120).stdout


def _bundle(repository, commit, destination):
    """Bundle only the selected commit's reachable history, without checkout.

    No working-tree files, other branches, hooks, configuration or private task
    records are copied. A bare temporary repository avoids executing filters.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='harness-bundle-') as tmp:
        clone = Path(tmp) / 'repo'
        template = Path(tmp) / 'empty-template'
        template.mkdir()
        _git('init', '--bare', '--template=' + str(template), clone)
        _git('-C', clone, 'fetch', '--no-tags', '--', Path(repository).resolve(), commit)
        _git('-C', clone, 'update-ref', 'refs/heads/harness-snapshot', commit)
        _git('-C', clone, 'bundle', 'create', destination.resolve(), 'refs/heads/harness-snapshot')
    return hashlib.sha256(destination.read_bytes()).hexdigest()


def _read_input(item, key):
    value = item.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError('Task input requires a file path: ' + key)
    return json.loads(Path(value).read_text())


def prepare_batch(config, output):
    """Preflight all tasks before writing; model jobs contain public fields only."""
    required = {'run_id', 'model', 'tasks', 'episodes_per_task', 'max_rollouts', 'max_graders', 'coordinator_seconds'}
    if not isinstance(config, dict) or set(config) - (required | {'limits', 'synthetic'}) or not required <= set(config):
        raise ValueError('Batch configuration has missing or unknown fields')
    run_id = identifier(config['run_id'])
    for field in ('episodes_per_task', 'max_rollouts', 'max_graders', 'coordinator_seconds'):
        if type(config[field]) is not int or config[field] <= 0:
            raise ValueError('Positive integer required: ' + field)
    if config['max_rollouts'] > 32 or config['max_graders'] > 16 or config['coordinator_seconds'] > 86000:
        raise ValueError('Deployment limits: 32 rollouts, 16 graders, 86000 coordinator seconds')
    if type(config.get('synthetic', False)) is not bool:
        raise ValueError('synthetic must be an explicit boolean')
    if not isinstance(config['tasks'], list) or not config['tasks']:
        raise ValueError('At least one task is required')
    if len(config['tasks']) * config['episodes_per_task'] > 1000:
        raise ValueError('Initial remote batches are limited to 1000 episodes')
    validate_model(config['model'])
    if config['model']['kind'] == 'scripted' and config.get('synthetic') is not True:
        raise ValueError('Scripted batches must be explicitly synthetic')
    try:
        requested_limits = asdict(RunLimits(**config.get('limits', {})))
    except TypeError:
        raise ValueError('Invalid batch limits') from None
    for key in ('max_steps', 'max_tool_calls', 'max_output_tokens', 'max_context_chars'):
        if type(requested_limits[key]) is not int or requested_limits[key] < (0 if key == 'max_tool_calls' else 1):
            raise ValueError('Invalid batch limit: ' + key)
    jobs, records, sources, seen = [], [], {}, set()
    for item in config['tasks']:
        if (not isinstance(item, dict) or not {'task', 'experiment', 'repository'} <= set(item)
                or set(item) - {'task', 'experiment', 'repository', 'environment_manifest', 'role', 'semantic'}):
            raise ValueError('Missing or unknown task input field')
        if not isinstance(item['repository'], str) or not item['repository']:
            raise ValueError('A local repository path is required')
        task, experiment = _read_input(item, 'task'), _read_input(item, 'experiment')
        validate_task(task)
        validate(experiment, EXPERIMENT)
        task_id = identifier(task['id'])
        if task_id in seen:
            raise ValueError('Duplicate task id')
        seen.add(task_id)
        role = item.get('role', 'evaluation')
        if role not in ('training', 'evaluation') or role == 'training' and task['split'] != 'train':
            raise ValueError('Training requires train-split tasks')
        if not experiment['frozen'] or not any(row['id'] == task_id and row['task_hash'] == digest(task)
                                               for row in experiment['task_manifest']):
            raise ValueError('Task must belong to the frozen experiment')
        if 'semantic' in item and config.get('synthetic') is not True:
            raise ValueError('Preloaded semantic assessments are only allowed in synthetic smoke batches')
        commit = task['repository']['commit']
        repo = GitRepository(item['repository'], commit)
        sources.setdefault(commit, item['repository'])
        relative = f'{run_id}/repos/{commit}.bundle'
        private_record = {'task': task, 'experiment': experiment, 'role': role}
        if 'semantic' in item:
            private_record['semantic'] = _read_input(item, 'semantic')
            private_record['synthetic'] = True
        records.append((task_id, private_record))
        limits = dict(requested_limits)
        for field in ('max_tool_calls', 'max_output_tokens'):
            limits[field] = min(limits[field], task['budgets'][field])
        environment = public_environment(_read_input(item, 'environment_manifest')) if item.get('environment_manifest') is not None else None
        available = {tool.spec.name for tool in code_understanding_tools(include_execution=environment is not None)}
        if set(task['permitted_tools']) - available:
            raise ValueError('Task requests tools unavailable in repository workers')
        if environment is not None and environment.get('snapshot_files') != repo.snapshot_hashes():
            raise ValueError('Environment snapshot does not match pinned repository')
        for index in range(config['episodes_per_task']):
            group_id = 'group-' + hashlib.sha256(task_id.encode()).hexdigest()[:20]
            model = dict(config['model'])
            if model['kind'] == 'tinker' and model.get('seed') is not None:
                seed_material = json.dumps([model['seed'], task_id], separators=(',', ':')).encode()
                task_seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:4], 'big')
                model['seed'] = (task_seed + index) % (2 ** 31)
            job = {'schema_version': 1, 'run_id': run_id, 'episode_id': f'{group_id}-{index:04d}',
                'group_id': group_id, 'task_id': task_id, 'task_hash': digest(task), 'question': task['question'],
                'repository': {'bundle': relative, 'bundle_sha256': '0' * 64, 'commit': commit},
                'permitted_tools': task['permitted_tools'], 'limits': limits,
                'max_submission_bytes': task['budgets']['max_submission_bytes'],
                'model': model, 'environment_manifest': environment}
            validate_job(job)
            jobs.append(job)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    public, private = output / 'public', output / 'private'
    public.mkdir()
    private.mkdir(mode=0o700)
    hashes = {commit: _bundle(repository, commit, public / run_id / 'repos' / (commit + '.bundle'))
              for commit, repository in sources.items()}
    for job in jobs:
        job['repository']['bundle_sha256'] = hashes[job['repository']['commit']]
    for task_id, record in records:
        atomic_json(private / run_id / 'tasks' / (task_id + '.json'), record)
    manifest = {key: config[key] for key in ('run_id', 'max_rollouts', 'max_graders', 'coordinator_seconds')}
    manifest.update(schema_version=1, synthetic=config.get('synthetic', False), jobs=jobs)
    atomic_json(public / run_id / 'manifest.json', manifest)
    atomic_json(output / 'bundle.json', {'run_id': run_id, 'manifest_hash': job_hash(manifest), 'episodes': len(jobs)})
    return manifest
