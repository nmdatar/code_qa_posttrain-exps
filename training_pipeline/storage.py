"""Durable local records are authoritative; W&B is a best-effort mirror."""
import copy
import hashlib
import json
import os
from pathlib import Path
import uuid
import threading
from .concurrency import synchronized


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    with tmp.open('x') as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def run_label(config):
    """Human-readable experimental condition, independent of the unique run ID."""
    solver = config['model']['base_model'].rsplit('/', 1)[-1]
    judge = config.get('judge', {}).get('base_model')
    if judge is None:
        judge = config['model']['base_model'] if config['environment']['kind'] == 'collection' else 'task-verifier'
    stages = []
    for stage in config['stages']:
        label = f"{stage['kind']}-lr{stage['learning_rate']:g}-b{stage['batch_size']}"
        if stage['kind'] == 'grpo':
            label += f"-g{stage['group_size']}"
        stages.append(label)
    return f"{solver}__judge-{judge.rsplit('/', 1)[-1]}__{'_then_'.join(stages)}__r{config['model']['rank']}-s{config['seed']}"


class Tracker:
    def __init__(self, root, config, run_id, wandb_module=None, run_config=None, job_type='training'):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id
        self._io_lock = threading.RLock()
        self.remote = None
        self.wandb = None
        # Resume keeps the original remote identity and display name. A standalone
        # evaluation is a separate W&B job, even when writing beside training logs.
        suffix = '-eval-' + uuid.uuid4().hex[:8] if job_type == 'evaluation' else ''
        identity_path = self.root / ('tracking' + suffix + '.json')
        name = config.get('run_name') or (run_label(run_config) if run_config else run_id)
        self.organization = {'id': run_id + suffix, 'name': name + '__' + run_id + suffix,
            'group': config.get('experiment_id'), 'entity': config.get('entity'),
            'project': config.get('project'), 'job_type': job_type,
            'tags': config.get('tags', []), 'notes': config.get('notes')}
        if identity_path.exists():
            self.organization = read(identity_path)
        else:
            atomic_json(identity_path, self.organization)
        self.event('run_metadata', organization=self.organization, config=run_config)
        if config.get('mode', 'disabled') != 'disabled':
            try:
                if wandb_module is None:
                    import wandb as wandb_module
                self.wandb = wandb_module
                self.remote = wandb_module.init(**self.organization, mode=config['mode'],
                    resume='allow', dir=str(self.root), config=run_config,
                    settings=wandb_module.Settings(disable_git=True))
            except Exception as exc:
                self._append({'event': 'tracking_failure', 'error_type': type(exc).__name__})

    @synchronized
    def _append(self, event):
        with (self.root / 'events.jsonl').open('a') as f:
            f.write(json.dumps(event, allow_nan=False) + '\n')
            f.flush()
            os.fsync(f.fileno())

    @synchronized
    def event(self, event, **fields):
        row = {'event_id': uuid.uuid4().hex, 'run_id': self.run_id, 'event': event, **fields}
        self._append(row)
        if self.remote:
            try:
                self.remote.log(row)
            except Exception as exc:
                self._append({'event': 'tracking_failure', 'event_id': row['event_id'],
                              'error_type': type(exc).__name__})
        return row

    @synchronized
    def trajectory(self, trajectory):
        path = self.root / 'trajectories' / (trajectory.episode_id + '.json')
        atomic_json(path, trajectory.to_dict())
        self.event('trajectory', episode_id=trajectory.episode_id, policy_id=trajectory.policy_id,
                   artifact=str(path), termination=trajectory.termination,
                   reward=trajectory.verification.reward if trajectory.verification else None,
                   usage=trajectory.usage)
        # Stable sample; full traces remain local regardless of upload success.
        if self.remote and int(digest(trajectory.episode_id)[:8], 16) % 8 == 0:
            public = trajectory.to_dict()
            public.pop('verification', None)
            public_path = self.root / 'public-traces' / (trajectory.episode_id + '.json')
            atomic_json(public_path, public)
            self.artifact(public_path, 'rollout')
            try:
                table = self.wandb.Table(columns=['episode_id', 'policy_id', 'submission', 'events'],
                    data=[[trajectory.episode_id, trajectory.policy_id, json.dumps(trajectory.submission),
                           json.dumps(trajectory.events)]])
                self.remote.log({'rollouts': table})
            except Exception as exc:
                self._append({'event': 'tracking_failure', 'error_type': type(exc).__name__})

    @synchronized
    def artifact(self, path, kind):
        """Mirror an explicitly selected local artifact; never walk private folders."""
        if not self.remote:
            return
        try:
            file = Path(path)
            artifact = self.wandb.Artifact(self.run_id + '-' + file.stem, type=kind)
            artifact.add_file(str(file), name=file.name)
            self.remote.log_artifact(artifact)
        except Exception as exc:
            self._append({'event': 'tracking_failure', 'artifact': str(path),
                          'error_type': type(exc).__name__})

    @synchronized
    def finish(self, status):
        self.event('run_finish', status=status)
        if self.remote:
            try:
                self.remote.finish(exit_code=0 if status == 'complete' else 1)
            except Exception as exc:
                self._append({'event': 'tracking_failure', 'error_type': type(exc).__name__})


class Checkpoints:
    def __init__(self, root):
        self.root = Path(root) / 'checkpoints'

    def commit(self, backend, config, state, parent=None):
        ident = 'ckpt-' + uuid.uuid4().hex
        self.root.mkdir(parents=True, exist_ok=True)
        pending = self.root / (ident + '.pending.json')
        atomic_json(pending, {'id': ident, 'status': 'saving', 'state': state})
        artifacts = backend.save(ident)
        backend.verify_artifacts(artifacts)
        manifest = {'schema_version': '1.0', 'id': ident, 'parent': parent,
                    'config': copy.deepcopy(config), 'config_hash': semantic_hash(config),
                    'identity': copy.deepcopy(backend.identity), 'artifacts': artifacts, 'state': copy.deepcopy(state)}
        manifest['manifest_hash'] = digest(manifest)
        path = self.root / (ident + '.json')
        atomic_json(path, manifest)
        atomic_json(self.root / 'latest.json', {'path': str(path.resolve())})
        pending.unlink()
        return path, manifest


def semantic_hash(config):
    return digest({k: v for k, v in config.items() if k not in {'output', 'tracking', 'run_id'}})


def load_checkpoint(path, config=None):
    value = read(path)
    expected = value.pop('manifest_hash', None)
    if digest(value) != expected:
        raise ValueError('Checkpoint manifest integrity failure')
    value['manifest_hash'] = expected
    if config is not None and semantic_hash(config) != value['config_hash']:
        raise ValueError('Resume configuration changed; use fork')
    return value
