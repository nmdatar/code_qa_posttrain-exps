"""Synchronous stages, complete groups, and immutable checkpoint boundaries."""
import copy
import random
import uuid
import hashlib
import fcntl
from pathlib import Path
from agent_harness.runner import run_episode
from .config import inputs, resolve_run_config
from .storage import Tracker, Checkpoints, atomic_json, load_checkpoint, semantic_hash, read
from .strategies import grpo_batch
from .rendering import sft_batch
from .toy import ToyFactory
from .repository import RepositoryFactory
from .contracts import ConfigurationError
from .budget import SpendLedger, current_prices
from .tinker_backend import TinkerBackend


def tuples(value):
    return tuple(tuples(v) for v in value) if isinstance(value, list) else value


def make_backend(config, evaluation_only=False):
    settings = config['spend']
    ledger_path = Path(settings['ledger'])
    if ledger_path.exists():
        prices = read(ledger_path)['prices']
        if prices.get('model', config['model']['base_model']) != config['model']['base_model']:
            raise ConfigurationError('Spend ledger belongs to another solver model; select a new ledger')
    else:
        prices = settings['prices'] or current_prices(config['model']['base_model'])
    if config['environment']['kind'] == 'collection':
        from .launch import estimate
        plan = estimate(config, prices, evaluation_only=evaluation_only)
        prior = read(ledger_path)['reserved_usd'] if ledger_path.exists() else 0
        plan['prior_reserved_usd'] = prior
        plan['combined_upper_estimate_usd'] = prior + plan['upper_estimate_usd']
        if plan['status'] != 'within_ceiling' or plan['combined_upper_estimate_usd'] > settings['cap_usd']:
            raise ConfigurationError('Planned collection run exceeds spending ceiling')
        atomic_json(Path(config['output']).parent / 'launch-estimate.json', plan)
    ledger = SpendLedger(ledger_path, settings['cap_usd'], prices, read(ledger_path)['ttl_seconds'] if ledger_path.exists() else config['model']['checkpoint_ttl_seconds'])
    return TinkerBackend(config['model'], config['limits'], ledger)


class Pipeline:
    def __init__(self, config, backend=None, tracker=None, factory=None, data=None):
        config = resolve_run_config(config)
        self.config = config
        self.data = inputs(config) if data is None else data
        self.root = Path(config['output'])
        self.tracker = tracker
        self.backend = backend
        self.factory = factory
        self.checkpoints = Checkpoints(self.root)
        self.rng = random.Random(config['seed'])
        self.state = {'stage': 0, 'optimizer_step': 0, 'stage_updates': 0, 'stage_batches': 0,
                      'attempted_batches': 0, 'cursor': 0, 'order': [], 'needs_fresh_optimizer': False,
                      'data_identity': self.data['identity'], 'rng': self.rng.getstate()}
        self.last_path = None
        self.last_manifest = None
        self._lock = None

    def setup(self, create_run, job_type='training'):
        if 'judge' in self.config and self.config['judge']['prices'] is None:
            self.config['judge']['prices'] = current_prices(self.config['judge']['base_model'], sampling_only=True)
        if create_run:
            self.root.mkdir(parents=True, exist_ok=False)
            atomic_json(self.root / 'config.json', self.config)
            source_root = Path(__file__).resolve().parent.parent
            hashes = {}
            for package in ('training_pipeline', 'agent_harness', 'qa_eval'):
                for file in (source_root / package).glob('*.py'):
                    relative = file.relative_to(source_root)
                    target = self.root / 'source' / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    contents = file.read_bytes()
                    target.write_bytes(contents)
                    hashes[str(relative)] = hashlib.sha256(contents).hexdigest()
            atomic_json(self.root / 'source' / 'manifest.json', hashes)
        else:
            self.root.mkdir(parents=True, exist_ok=True)
        self.tracker = self.tracker or Tracker(self.root, self.config['tracking'], self.config['run_id'],
                                               run_config=self.config, job_type=job_type)
        self.backend = self.backend or make_backend(self.config, evaluation_only=job_type == 'evaluation')
        if self.factory is None and self.config['environment']['kind'] == 'collection':
            from .collection import CollectionFactory
            self.factory = CollectionFactory(self.config, self.root, self.backend.ledger)
            try:
                self.factory.prepare_judge()
            except BaseException:
                self.backend.close()
                raise
        self.factory = self.factory or (ToyFactory() if self.config['environment']['kind'] == 'toy' else
                                       RepositoryFactory(self.config, self.root))
        self.tracker.event('resolved_models', solver=self.backend.identity,
            judge=getattr(getattr(self.factory, 'judge', None), 'identity', None))

    def commit(self):
        self.state['rng'] = self.rng.getstate()
        self.last_path, self.last_manifest = self.checkpoints.commit(self.backend, self.config, self.state,
            self.last_manifest['id'] if self.last_manifest else None)
        self.backend.use_sampler(self.last_manifest['artifacts'])
        self.tracker.event('checkpoint', checkpoint_id=self.last_manifest['id'], path=str(self.last_path),
                           optimizer_step=self.state['optimizer_step'], stage=self.state['stage'])
        self.tracker.artifact(self.last_path, 'checkpoint-manifest')
        return self.last_manifest

    def _batch(self, rows, count, partial=False):
        if not self.state['order']:
            self.state['order'] = list(range(len(rows)))
            self.rng.shuffle(self.state['order'])
        if partial:
            if self.state['cursor'] == len(rows):
                self.state['cursor'] = 0
                self.rng.shuffle(self.state['order'])
            count = min(count, len(rows) - self.state['cursor'])
        if count > len(rows):
            raise ConfigurationError('Batch size exceeds distinct available tasks')
        picked = []
        for _ in range(count):
            if self.state['cursor'] == len(rows):
                self.state['cursor'] = 0
                self.rng.shuffle(self.state['order'])
                # Preserve a complete permutation, but defer already-picked rows
                # until after this batch. The cursor remains exactly resumable.
                used = {r['id'] for r in picked}
                self.state['order'].sort(key=lambda i: rows[i]['id'] in used)
            picked.append(rows[self.state['order'][self.state['cursor']]])
            self.state['cursor'] += 1
        return picked

    def rollout(self, task, group_id, temperature):
        return run_episode(self.backend, self.factory, task, self.config['limits'],
            run_id=self.config['run_id'], stage=self.state['stage'], group_id=group_id,
            episode_id='ep-' + uuid.uuid4().hex, experiment_hash=semantic_hash(self.config),
            temperature=temperature, tracker=self.tracker)

    def group(self, task, size, temperature):
        retries = self.config.get('group_retries', 1)
        for attempt in range(retries + 1):
            group_id = 'group-' + uuid.uuid4().hex
            group = [self.rollout(task, group_id, temperature) for _ in range(size)]
            unresolved = [t for t in group if t.verification is None or t.verification.status == 'unresolved']
            if not unresolved:
                return group
            self.tracker.event('excluded_group', group_id=group_id, attempt=attempt,
                               episodes=[t.episode_id for t in group])
            if attempt == retries or not all(t.verification and t.verification.retryable for t in unresolved):
                return group
        raise AssertionError('Unreachable')

    def evaluate(self, manifest=None, tasks=None, cohort=None, base=False):
        manifest = manifest or self.last_manifest
        if not base and (manifest is None or self.backend.policy_id != manifest['artifacts']['sampler']):
            raise ConfigurationError('Evaluation sampler does not match checkpoint')
        from .cohorts import select_tasks
        if tasks is None:
            selected, cohort_identity = select_tasks(self.data, self.config['evaluation'], cohort)
        else:
            if cohort is not None or 'cohort_manifest' in self.config['evaluation']:
                raise ConfigurationError('Cannot override pinned cohort tasks')
            selected, cohort_identity = tasks[:self.config['evaluation']['max_tasks']], None
        if any(t['split'] != 'development' for t in selected):
            raise ConfigurationError('Automatic evaluation is development-only')
        rows = []
        for task in selected:
            trajectory = self.rollout(task, 'evaluation-' + uuid.uuid4().hex, self.config['evaluation']['temperature'])
            v = trajectory.verification
            rows.append({'task_id': task['id'], 'episode_id': trajectory.episode_id,
                         'family_id': task['family_id'], 'usage': trajectory.usage,
                         'status': v.status if v else 'unresolved', 'reward': v.reward if v else None,
                         'termination': trajectory.termination})
        resolved = [r for r in rows if r['status'] == 'resolved']
        report = {'checkpoint_id': manifest['id'] if manifest else 'unchanged-base', 'policy_id': self.backend.policy_id,
                  'cohort': cohort_identity,
                  'data_identity': self.data['identity'], 'environment': self.factory.identity,
                  'config_hash': semantic_hash(self.config), 'synthetic': self.config['environment']['kind'] == 'toy',
                  'reward_version': self.factory.reward_version,
                  'reward_calibrated': False if self.config['environment']['kind'] == 'collection' else None,
                  'expected': len(selected), 'attempted': len(rows), 'resolved': len(resolved),
                  'demonstrated_quality': sum(r['reward'] for r in resolved)/len(rows) if rows else None,
                  'scoring_coverage': len(resolved)/len(rows) if rows else 0,
                  'completion_rate': sum(r['termination'] == 'completed' for r in rows)/len(rows) if rows else 0,
                  'coverage_eligible': bool(rows) and len(resolved)/len(rows) >= .95,
                  'mean_reward': sum(r['reward'] for r in resolved)/len(resolved) if resolved else None, 'results': rows}
        path = self.root / 'evaluations' / (report['checkpoint_id'] + '-' + uuid.uuid4().hex + '.json')
        atomic_json(path, report)
        self.tracker.event('evaluation', artifact=str(path), **{k: v for k, v in report.items() if k != 'results'})
        self.tracker.artifact(path, 'evaluation')
        return report

    def _regression_stop(self, report):
        """Persist selection-only regression decisions in checkpoint framework state."""
        controls = self.config.get('stopping', {})
        if 'regression_delta' not in controls:
            return False
        quality = report['demonstrated_quality']
        if 'baseline_quality' not in self.state:
            if not report['coverage_eligible']:
                raise ConfigurationError('Regression baseline requires at least 95% scoring coverage')
            self.state['baseline_quality'] = quality
            self.state['regression_count'] = 0
            return False
        bad = quality is not None and self.state['baseline_quality'] - quality >= controls['regression_delta'] - 1e-12
        self.state['regression_count'] = self.state.get('regression_count', 0) + 1 if bad else 0
        return self.state['regression_count'] >= controls.get('regression_checks', 2)

    def run(self, checkpoint=None, purpose='run', stop_after_updates=None):
        if purpose not in {'run', 'fork', 'resume'}:
            raise ValueError('Invalid run purpose')
        original = load_checkpoint(checkpoint, self.config if purpose == 'resume' else None) if checkpoint else None
        if purpose == 'resume' and original and original['state'].get('stop_reason'):
            raise ConfigurationError('Experiment stopped by a safety rule; diagnose and use a new fork')
        if purpose != 'run' and original is None:
            raise ValueError('Checkpoint required')
        if original and (original['identity']['base_model'] != self.config['model']['base_model'] or
                         original['identity']['rank'] != self.config['model']['rank']):
            raise ConfigurationError('Incompatible checkpoint model')
        self.setup(create_run=purpose != 'resume')
        status = 'failed'
        self._lock = (self.root / '.training.lock').open('a')
        try:
            fcntl.flock(self._lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self._lock.close()
            raise ConfigurationError('Another process owns this training run') from None
        try:
            if original:
                for key in ('template_hash', 'tokenizer_class', 'renderer'):
                    if original['identity'][key] != self.backend.identity[key]:
                        raise ConfigurationError('Checkpoint renderer/tokenizer mismatch')
            # Rendering needs a tokenizer, but no paid generation or trainer allocation.
            if any(s['kind'] == 'sft' for s in self.config['stages']):
                for example in self.data['sft']:
                    sft_batch(self.backend.renderer, [example])
            self.backend.create_trainer(self.config['seed'])
            if original:
                self.backend.load(original['artifacts'], purpose)
                self.last_manifest, self.last_path = original, Path(checkpoint)
                if purpose == 'resume':
                    self.state = copy.deepcopy(original['state'])
                    if self.state['data_identity'] != self.data['identity']:
                        raise ConfigurationError('Resume data changed')
                    self.rng.setstate(tuples(self.state['rng']))
            if purpose != 'resume':
                self.commit()
            controls = self.config.get('stopping', {})
            if 'regression_delta' in controls and 'baseline_quality' not in self.state:
                if self.state['optimizer_step'] != 0:
                    raise ConfigurationError('Regression stopping requires a saved initial baseline')
                self._regression_stop(self.evaluate())
                self.commit()
            elif self.config['environment']['kind'] == 'collection' and self.state['optimizer_step'] == 0 and not list((self.root/'evaluations').glob(self.last_manifest['id']+'-*.json')):
                self.evaluate()
            self.tracker.event('run_start', purpose=purpose, optimizer_step=self.state['optimizer_step'])
            while self.state['stage'] < len(self.config['stages']):
                stage_index = self.state['stage']
                stage = self.config['stages'][stage_index]
                if self.state['needs_fresh_optimizer']:
                    self.backend.create_trainer(self.config['seed'] + stage_index)
                    self.backend.load(self.last_manifest['artifacts'], 'fork')
                    self.state['needs_fresh_optimizer'] = False
                if stage['kind'] == 'sft':
                    examples = self._batch(self.data['sft'], stage['batch_size'], partial=True)
                    rows = sft_batch(self.backend.renderer, examples)
                    stats = {'example_ids': [e['id'] for e in examples]}
                    loss = 'cross_entropy'
                else:
                    tasks = self._batch(self.data['tasks'], stage['batch_size'])
                    groups = [self.group(t, stage['group_size'], stage['temperature']) for t in tasks]
                    rows, stats = grpo_batch(groups)
                    loss = 'importance_sampling'
                self.state['attempted_batches'] += 1
                self.state['stage_batches'] += 1
                if rows:
                    options = {'optimizer': stage['optimizer']} if 'optimizer' in stage else {}
                    result = self.backend.update(rows, loss, stage['learning_rate'], **options)
                    self.state['optimizer_step'] += 1
                    self.state['stage_updates'] += 1
                    self.tracker.event('update', stage=stage_index, optimizer_step=self.state['optimizer_step'],
                        attempted_batches=self.state['attempted_batches'], behavior_policy=self.backend.policy_id,
                        learning_rate=stage['learning_rate'], batch_size=stage['batch_size'],
                        group_size=stage.get('group_size'), optimizer=stage.get('optimizer', {}),
                        **result, **stats)
                else:
                    self.tracker.event('skipped_update', stage=stage_index, **stats)
                if (stage['kind'] == 'grpo' and not self.state['stage_updates'] and
                        self.state['stage_batches'] >= controls.get('initial_zero_batches', float('inf'))):
                    self.state['stop_reason'] = 'initial_zero_batches'
                    self.tracker.event('experiment_stop', reason=self.state['stop_reason'])
                    self.commit()
                    self.evaluate()
                    status = 'stopped_initial_zero_batches'
                    return self.last_path
                done = self.state['stage_updates'] >= stage['max_updates'] or self.state['stage_batches'] >= stage['max_batches']
                if done:
                    self.tracker.event('stage_complete', stage=stage_index, updates=self.state['stage_updates'],
                                       attempted_batches=self.state['stage_batches'])
                    self.state.update(stage=stage_index+1, stage_updates=0, stage_batches=0, cursor=0, order=[],
                                      needs_fresh_optimizer=stage_index+1 < len(self.config['stages']))
                # Every RL update needs an immutable sampler. Commit includes the
                # optimizer so it is also a resumable boundary. SFT honors cadence.
                every = self.config['evaluation']['every']
                scheduled_evaluation = bool(rows) and every and self.state['optimizer_step'] % every == 0
                should_commit = done or scheduled_evaluation or (bool(rows) and (stage['kind'] == 'grpo' or
                    self.state['optimizer_step'] % self.config['checkpoint_every'] == 0))
                if should_commit:
                    self.commit()
                    if done or scheduled_evaluation:
                        regression = self._regression_stop(self.evaluate())
                        if 'regression_delta' in controls:
                            # Save the evaluated decision state, including a stop,
                            # so a resume cannot forget a regression streak.
                            if regression:
                                self.state['stop_reason'] = 'selection_regression'
                            self.commit()
                        if regression:
                            self.tracker.event('experiment_stop', reason='selection_regression')
                            status = 'stopped_selection_regression'
                            return self.last_path
                if stop_after_updates is not None and self.state['optimizer_step'] >= stop_after_updates:
                    if not should_commit:
                        self.commit()
                    status = 'interrupted_at_committed_boundary'
                    return self.last_path
            status = 'complete'
            return self.last_path
        finally:
            try:
                self.tracker.finish(status)
            finally:
                fcntl.flock(self._lock, fcntl.LOCK_UN)
                self._lock.close()
                if hasattr(self.factory, 'close'):
                    try:
                        self.factory.close()
                    except Exception as exc:
                        self.tracker.event('judge_cleanup_failure', error_type=type(exc).__name__)
                if hasattr(self.backend, 'close'):
                    try:
                        self.backend.close('success' if status == 'complete' else 'interrupted' if status == 'interrupted_at_committed_boundary' else 'errored')
                    except Exception as exc:
                        self.tracker.event('client_cleanup_failure', error_type=type(exc).__name__)


def evaluate_checkpoint(path, output=None, backend=None, tracker=None, factory=None, cohort=None):
    manifest = load_checkpoint(path)
    config = copy.deepcopy(manifest['config'])
    if output:
        config['output'] = str(output)
    pipeline = Pipeline(config, backend=backend, tracker=tracker, factory=factory)
    from .cohorts import select_tasks
    select_tasks(pipeline.data, pipeline.config['evaluation'], cohort)
    pipeline.setup(create_run=False, job_type='evaluation')
    status = 'failed'
    try:
        for key in ('base_model', 'rank', 'template_hash', 'tokenizer_class', 'renderer'):
            if pipeline.backend.identity[key] != manifest['identity'][key]:
                raise ConfigurationError('Evaluation checkpoint identity mismatch')
        pipeline.backend.load(manifest['artifacts'], 'evaluate')
        pipeline.last_manifest = manifest
        report = pipeline.evaluate(manifest, cohort=cohort)
        status = 'complete'
        return report
    finally:
        pipeline.tracker.finish(status)
        if hasattr(pipeline.factory, 'close'):
            try:
                pipeline.factory.close()
            except Exception as exc:
                pipeline.tracker.event('judge_cleanup_failure', error_type=type(exc).__name__)
        if hasattr(pipeline.backend, 'close'):
            try:
                pipeline.backend.close('success' if status == 'complete' else 'errored')
            except Exception as exc:
                pipeline.tracker.event('client_cleanup_failure', error_type=type(exc).__name__)


def evaluate_base(config, cohort=None, backend=None, factory=None):
    """Evaluate unchanged weights without creating a trainer or checkpoint."""
    pipeline = Pipeline(config, backend=backend, factory=factory)
    from .cohorts import select_tasks
    select_tasks(pipeline.data, pipeline.config['evaluation'], cohort)
    pipeline.setup(create_run=True, job_type='evaluation')
    status = 'failed'
    try:
        if not pipeline.backend.policy_id.startswith('base:'):
            raise ConfigurationError('Baseline requires unchanged base weights')
        report = pipeline.evaluate(cohort=cohort, base=True)
        status = 'complete'
        return report
    finally:
        try:
            pipeline.tracker.finish(status)
        finally:
            try:
                if hasattr(pipeline.factory, 'close'):
                    pipeline.factory.close()
            finally:
                if hasattr(pipeline.backend, 'close'):
                    pipeline.backend.close('success' if status == 'complete' else 'errored')
