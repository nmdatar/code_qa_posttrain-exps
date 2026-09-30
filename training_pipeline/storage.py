"""Durable local records are authoritative; W&B is a best-effort mirror."""
import copy
import csv
import hashlib
import json
import os
import re
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
        if stage['kind'] in {'grpo', 'reinforce'}:
            label += f"-g{stage['group_size']}"
        stages.append(label)
    return f"{solver}__judge-{judge.rsplit('/', 1)[-1]}__{'_then_'.join(stages)}__r{config['model']['rank']}-s{config['seed']}"


class Tracker:
    def __init__(self, root, config, run_id, wandb_module=None, run_config=None, job_type='training'):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id
        self.flush_every = config.get('flush_every', 32)
        # Large traces remain in the authoritative experiment archive. Rich W&B
        # uploads require an explicit opt-in, independent of local flush cadence.
        self.upload_media = config.get('upload_policy', 'metrics-only') == 'full'
        self._io_lock = threading.RLock()
        self.remote = None
        self.wandb = None
        self.context = {'phase': 'benchmark' if job_type == 'benchmark' else 'training', 'optimizer_step': 0}
        self.answer_rows = read(self.root/'answers.json')['data'] if (self.root/'answers.json').exists() else []
        for row in self.answer_rows:
            if len(row) == 10:
                row.extend([row[6], None])
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
                    settings=wandb_module.Settings(disable_git=True, disable_code=True,
                                                  console='off', x_disable_stats=True))
                if isinstance(getattr(self.remote, 'url', None), str) and self.remote.url:
                    atomic_json(self.root/'tracking-url.json', {'url': self.remote.url})
                self.remote.define_metric('evaluation/*', step_metric='optimizer_step')
                self.remote.define_metric('training/*', step_metric='attempted_batches')
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
        if event == 'training_batch':
            curve = self.root / 'reward-curve.csv'
            columns = ['attempted_batches', 'optimizer_step', 'mean_reward', 'reward_ema',
                       'eligible_mean_reward', 'scoring_coverage', 'excluded_groups',
                       'zero_variance_groups', 'contributing_trajectories']
            needs_header = not curve.exists()
            with curve.open('a', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=columns)
                if needs_header:
                    writer.writeheader()
                writer.writerow({key: fields.get(key) for key in columns})
                stream.flush()
                os.fsync(stream.fileno())
        if self.remote and (self.upload_media or event in {
                'evaluation', 'training_batch', 'training_step_timing', 'update',
                'run_start', 'run_finish', 'experiment_stop', 'skipped_update', 'stage_complete'}):
            try:
                if event in {'evaluation', 'training_batch'}:
                    prefix = 'evaluation' if event == 'evaluation' else 'training'
                    row.update({prefix+'/'+k: v for k, v in fields.items()
                        if k.startswith(('mean_', 'accepted_', 'compute_units_'))
                        or k.endswith(('_per_accepted_answer', '_per_fully_correct_answer', '_measurement_coverage'))})
                if event == 'evaluation':
                    row.update({'evaluation/'+k: fields[k] for k in
                                ('demonstrated_quality', 'scoring_coverage', 'mean_reward', 'completion_rate', 'wall_seconds') if k in fields})
                elif event == 'training_batch':
                    row.update({'training/'+k: fields[k] for k in
                                ('mean_reward', 'eligible_mean_reward', 'reward_ema', 'scoring_coverage',
                                 'attempted_trajectories', 'resolved_trajectories', 'excluded_group_fraction',
                                 'reward_variance', 'contributing_trajectories', 'excluded_groups',
                                 'zero_variance_groups') if k in fields})
                elif event == 'training_step_timing':
                    row.update({'training/'+k:v for k,v in fields.items() if k.endswith('_seconds')})
                elif event == 'update':
                    row.update({'training/'+k: fields[k] for k in
                                ('reward_variance', 'contributing_trajectories', 'excluded_groups', 'zero_variance_groups') if k in fields})
                if self.upload_media:
                    self.remote.log(row)
                else:
                    # Do not serialize nested diagnostics, prompts, token arrays,
                    # task IDs, paths or arbitrary strings into remote history.
                    metrics = {k:v for k,v in row.items()
                               if v is None or type(v) in (bool, int, float)}
                    metrics['event'] = event
                    if isinstance(fields.get('status'), str):
                        metrics['status'] = fields['status'][:128]
                    self.remote.log(metrics)
            except Exception as exc:
                self._append({'event': 'tracking_failure', 'event_id': row['event_id'],
                              'error_type': type(exc).__name__})
        return row

    @synchronized
    def trajectory(self, trajectory):
        path = self.root / 'trajectories' / (trajectory.episode_id + '.json')
        atomic_json(path, trajectory.to_dict())
        self.event('trajectory', episode_id=trajectory.episode_id, policy_id=trajectory.policy_id,
                   task_id=trajectory.task_id, **self.context,
                   artifact=str(path), termination=trajectory.termination,
                   reward=trajectory.verification.reward if trajectory.verification else None,
                   usage=trajectory.usage)
        verification = trajectory.verification
        self.answer_rows.append([trajectory.episode_id, trajectory.task_id, self.context['phase'],
            self.context['optimizer_step'], trajectory.policy_id,
            json.dumps(trajectory.submission), verification.reward if verification else None,
            verification.status if verification else 'unresolved',
            '; '.join(verification.reasons) if verification else 'No verification', trajectory.termination,
            verification.diagnostics.get('strict_score', verification.reward) if verification else None,
            verification.diagnostics.get('training_reward') if verification else None,
            verification.diagnostics.get('correctness_score') if verification else None,
            verification.diagnostics.get('citation_score') if verification else None,
            verification.diagnostics.get('citation_status') if verification else None])
        if self.flush_every and len(self.answer_rows) % self.flush_every == 0:
            self.flush_answers()
        # Stable sample; full traces remain local regardless of upload success.
        if self.remote and self.upload_media and int(digest(trajectory.episode_id)[:8], 16) % 8 == 0:
            public = trajectory.to_dict()
            public.pop('verification', None)
            public_path = self.root / 'public-traces' / (trajectory.episode_id + '.json')
            atomic_json(public_path, public)
            self.artifact(public_path, 'rollout')
            try:
                table = self.wandb.Table(columns=['episode_id', 'policy_id', 'submission', 'events', 'reward', 'grading_status', 'grading_reason'],
                    data=[[trajectory.episode_id, trajectory.policy_id, json.dumps(trajectory.submission),
                           json.dumps(trajectory.events), verification.reward if verification else None,
                           verification.status if verification else 'unresolved',
                           '; '.join(verification.reasons) if verification else 'No verification']])
                self.remote.log({'rollouts': table})
            except Exception as exc:
                self._append({'event': 'tracking_failure', 'error_type': type(exc).__name__})

    @synchronized
    def artifact(self, path, kind):
        """Mirror an explicitly selected local artifact; never walk private folders."""
        if not self.remote or not self.upload_media:
            return
        try:
            file = Path(path)
            # W&B caps collection names; long run IDs plus evaluation UUIDs exceeded it.
            name = kind + '-' + digest({'run': self.run_id, 'file': file.name})[:32]
            artifact = self.wandb.Artifact(name, type=kind)
            artifact.add_file(str(file), name=file.name)
            self.remote.log_artifact(artifact)
        except Exception as exc:
            self._append({'event': 'tracking_failure', 'artifact': str(path),
                          'error_type': type(exc).__name__})

    @synchronized
    def flush_answers(self):
        columns = ['episode_id', 'task_id', 'phase', 'optimizer_step', 'policy_id', 'submission',
                   'reward', 'grading_status', 'grading_reason', 'termination', 'strict_score', 'training_reward', 'correctness_score', 'citation_score', 'citation_status']
        atomic_json(self.root/'answers.json', {'columns': columns, 'data': self.answer_rows})
        # Judge responses have their own explicit observability surface; never
        # upload whole private directories or provider prompts.
        try:
            from .judge_viewer import write_report as write_judges, log_report as log_judges
            judge_path, judge_records = write_judges(self.root)
            if self.remote and self.upload_media and judge_records:
                log_judges(self.remote, self.wandb, judge_path, judge_records)
                self.artifact(judge_path, 'judge-viewer')
        except Exception as exc:
            self._append({'event': 'tracking_failure', 'artifact': 'judge-viewer', 'error_type': type(exc).__name__})
        if self.answer_rows:
            try:
                from .trajectory_viewer import write_report, log_report
                report_path, records = write_report(self.root)
                if self.remote and self.upload_media:
                    self.remote.log({'answers': self.wandb.Table(columns=columns, data=list(self.answer_rows))})
                    log_report(self.remote, self.wandb, report_path, records)
                    self.artifact(report_path, 'trajectory-viewer')
            except Exception as exc:
                self._append({'event': 'tracking_failure', 'artifact': 'answers', 'error_type': type(exc).__name__})

    @synchronized
    def finish(self, status):
        self.flush_answers()
        self.event('run_finish', status=status)
        if self.remote:
            try:
                self.remote.finish(exit_code=0 if status == 'complete' or status.startswith('stopped_') else 1)
            except Exception as exc:
                self._append({'event': 'tracking_failure', 'error_type': type(exc).__name__})


class Checkpoints:
    def __init__(self, root):
        self.root = Path(root) / 'checkpoints'

    def commit(self, backend, config, state, parent=None):
        run_name = re.sub(r'[^A-Za-z0-9_-]+', '-', config.get('run_id', 'run')).strip('-')[:72] or 'run'
        ident = f"{run_name}-step-{state.get('optimizer_step', 0):04d}-{uuid.uuid4().hex[:8]}"
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
