"""Bounded, checkpoint-aware selection among prespecified research candidates.

No code generation or confirmation-set tuning. The first configuration is the
matched control. Existing Pipeline ledger limits remain authoritative.
"""
from __future__ import annotations
from collections import Counter
import copy
import fcntl
import math
import re
from pathlib import Path
import time

from .budget import BudgetLimit, process_lock_path
from .contracts import AmbiguousUpdate, ConfigurationError
from .orchestrator import Pipeline
from .storage import atomic_json, digest, load_checkpoint, read


def _events(root, offset=0):
    path = Path(root) / 'events.jsonl'
    if not path.exists():
        return []
    with path.open('rb') as stream:
        stream.seek(offset)
        lines = stream.read().splitlines()
    import json
    return [json.loads(line) for line in lines if line.strip()]


def _checkpoint(config, path=None):
    root = Path(config['output'])
    if list((root / 'checkpoints').glob('*.pending.json')):
        raise ConfigurationError('Pending checkpoint requires reconciliation')
    path = Path(path or read(root / 'checkpoints/latest.json')['path'])
    if path.resolve().parent != (root / 'checkpoints').resolve():
        raise ConfigurationError('Checkpoint is outside candidate directory')
    manifest = load_checkpoint(path, config)
    return str(path), manifest


def _recover(config, record):
    """Only a durable finish in this invocation proves a safe resume boundary."""
    root = Path(config['output'])
    if not root.exists():
        record['status'] = 'pending'
        return
    events = _events(root, record.get('event_offset', 0))
    finishes = [e for e in events if e.get('event') == 'run_finish']
    if not finishes or finishes[-1].get('status') not in {'complete', 'interrupted_at_committed_boundary'}:
        raise ConfigurationError('Interrupted invocation has no proven committed boundary')
    path, manifest = _checkpoint(config)
    if any(e.get('event') == 'update' and e.get('optimizer_step', 0) > manifest['state']['optimizer_step'] for e in events):
        raise ConfigurationError('Acknowledged update is newer than the checkpoint')
    record.update(checkpoint=path, checkpoint_hash=manifest['manifest_hash'],
                  optimizer_step=manifest['state']['optimizer_step'],
                  status='completed' if finishes[-1]['status'] == 'complete' else 'committed')
    record['receipts'].append({'checkpoint':path,'manifest_hash':manifest['manifest_hash'],
        'optimizer_step':record['optimizer_step'],'status':finishes[-1]['status'],'recovered':True,'saved_at':time.time()})


def _summary(config, record):
    events = _events(config['output'])
    evaluations = []
    for event in events:
        if event.get('event') != 'evaluation':
            continue
        report = read(event['artifact'])
        if (report.get('cohort') or {}).get('name') == 'confirmation':
            raise ConfigurationError('Autoresearch must not use confirmation results')
        evaluations.append(report)
    final_step = record['optimizer_step']
    final = next((r for r in reversed(evaluations) if r.get('optimizer_step') == final_step), None)
    initial = next((r for r in evaluations if r.get('optimizer_step') == 0), None)
    def compact(report):
        if report is None:
            return None
        keys = ('checkpoint_id', 'optimizer_step', 'demonstrated_quality', 'scoring_coverage',
                'completion_rate', 'resolved', 'expected', 'cohort', 'data_identity', 'reward_version')
        return {k: report.get(k) for k in keys}
    terminations = Counter(e.get('termination') for e in events
                           if e.get('event') == 'trajectory' and e.get('phase') == 'training')
    batches = [e for e in events if e.get('event') == 'training_batch']
    weighted_reward = 0.0
    resolved_rewards = 0
    for batch in batches:
        mean, count = batch.get('mean_reward'), batch.get('resolved_trajectories', 0)
        if type(mean) in (int, float) and math.isfinite(mean) and type(count) is int and count > 0:
            weighted_reward += mean * count
            resolved_rewards += count
    format_failures = 0
    observed_format_traces = 0
    for event in events:
        if event.get('event') == 'trajectory' and event.get('phase') == 'training' and event.get('artifact'):
            trajectory = read(event['artifact'])
            if isinstance(trajectory.get('events'), list):
                observed_format_traces += 1
                format_failures += sum(e.get('kind') == 'format_failure' for e in trajectory['events'])
    initial, final = compact(initial), compact(final)
    quality = final.get('demonstrated_quality') if final else None
    eligible = (final is not None and isinstance(quality, (int, float)) and math.isfinite(quality)
                and final.get('scoring_coverage', 0) >= .95)
    return {'initial': initial, 'final': final, 'eligible': eligible,
            'quality_change': (quality - initial['demonstrated_quality']
                if eligible and initial and isinstance(initial.get('demonstrated_quality'), (int, float)) else None),
            'training_terminations': dict(terminations), 'optimizer_step': final_step,
            'mean_training_reward': weighted_reward / resolved_rewards if resolved_rewards else None,
            'resolved_training_rewards': resolved_rewards,
            'masked_overlong_trajectories': sum(e.get('masked_overlong_trajectories', 0) for e in batches),
            'width_scaled_trajectories': sum(e.get('width_scaled_trajectories', 0) for e in batches),
            'format_failure_count': format_failures if observed_format_traces else None,
            'format_observed_trajectories': observed_format_traces}


def _pick(configs, state):
    remaining = [i for i, r in enumerate(state['runs']) if r['status'] in {'pending', 'committed'}]
    if not remaining:
        return None, 'All prespecified candidates have completed'
    if 0 in remaining:
        return 0, 'Run the matched control before choosing a diagnostic intervention'
    # Prefer a previously committed interrupted arm before spending on a new arm.
    resumable = next((i for i in remaining if state['runs'][i]['status'] == 'committed'), None)
    if resumable is not None:
        return resumable, 'Resume the exact candidate at its proven committed boundary'
    observations = [r.get('summary', {}) for r in state['runs'] if r['status'] == 'completed']
    failures = Counter()
    for observation in observations:
        failures.update(observation.get('training_terminations', {}))
    # Actual format events are more specific than the broad agent_error label.
    format_counts = [o.get('format_failure_count') for o in observations if o.get('format_failure_count') is not None]
    invalid_count = sum(format_counts) if format_counts else failures['agent_error']
    target = ('invalid' if invalid_count > failures['budget_exhausted']
              else 'overlong' if failures['budget_exhausted'] else 'width')
    def priority(index):
        config = configs[index]
        flags = config['stages'][0].get('stability', {})
        matches = {'invalid': config.get('environment', {}).get('invalid_action_policy') == 'zero-v1',
                   'overlong': flags.get('mask_overlong', False),
                   'width': flags.get('scale_tool_width', False)}
        # Isolate mechanisms before trying their combination.
        complexity = sum(bool(v) for v in matches.values())
        return (not matches[target], complexity, index)
    selected = min(remaining, key=priority)
    return selected, f'Observed training terminations {dict(failures)}, format failures {sum(format_counts) if format_counts else None} prioritize {target}; isolate before combining'


def _decide(state, index):
    run = state['runs'][index]
    result = run['summary']
    if not result['eligible']:
        return {'run_id': run['run_id'], 'decision': 'ineligible', 'reason': 'Final scoring coverage below 95% or final evaluation missing'}
    control = state['runs'][0].get('summary')
    if not control or not control['eligible']:
        return {'run_id': run['run_id'], 'decision': 'inconclusive', 'reason': 'Matched control is not coverage-eligible'}
    final, baseline = result['final'], control['final']
    for key in ('cohort', 'data_identity', 'reward_version', 'expected'):
        if final.get(key) != baseline.get(key):
            return {'run_id': run['run_id'], 'decision': 'incomparable', 'reason': 'Evaluation mismatch: ' + key}
    delta = final['demonstrated_quality'] - baseline['demonstrated_quality']
    previous = state.get('incumbent')
    best = baseline['demonstrated_quality'] if previous is None else previous['quality']
    complete = final.get('completion_rate')
    baseline_complete = baseline.get('completion_rate')
    safe_completion = (isinstance(complete, (int,float)) and isinstance(baseline_complete,(int,float))
                       and complete >= baseline_complete - .02)
    promote = index == 0 or (delta >= .01 and final['demonstrated_quality'] > best and safe_completion)
    if promote:
        state['incumbent'] = {'run_id': run['run_id'], 'checkpoint': run['checkpoint'],
                              'quality': final['demonstrated_quality'], 'provisional': True}
    return {'run_id': run['run_id'], 'decision': 'provisional_incumbent' if promote else 'retain_control_or_incumbent',
            'quality_delta_vs_control': delta, 'completion_gate': safe_completion,
            'reason': 'Selection-only screen; no statistical or confirmation claim'}


def worker(configs, root, *, pipeline_factory=None):
    """Run a finite list of ordinary Pipeline configs under one durable controller.

    First config is control. Returning state records completion or a bounded stop.
    Exceptions with unknown optimizer state always require explicit reconciliation.
    A test may inject an offline pipeline factory; production defaults to Pipeline.
    """
    configs = copy.deepcopy(list(configs))
    if not configs or len(configs) > 16:
        raise ConfigurationError('Autoresearch requires 1 to 16 prespecified candidates')
    if len({c['run_id'] for c in configs}) != len(configs) or len({str(Path(c['output']).resolve()) for c in configs}) != len(configs):
        raise ConfigurationError('Candidate run IDs and output directories must be distinct')
    for config in configs:
        if config['run_id'] == 'auto' or '{run_id}' in config['output']:
            raise ConfigurationError('Autoresearch needs frozen explicit run identities')
        if config.get('evaluation', {}).get('cohort') == 'confirmation':
            raise ConfigurationError('Confirmation is not an autoresearch cohort')
        if any(stage['kind'] not in {'grpo','reinforce'} for stage in config['stages']):
            raise ConfigurationError('Autoresearch accepts bounded RL stages only')
    control = configs[0]
    def content_bound_paths(spec, bindings):
        normalized = copy.deepcopy(spec or {})
        for path_key, hash_key in bindings:
            if path_key not in normalized:
                continue
            identity = normalized.get(hash_key)
            if not isinstance(identity, str) or re.fullmatch(r'[a-f0-9]{64}', identity) is None:
                raise ConfigurationError('Relocatable input requires a verified SHA256: ' + hash_key)
            # Paths differ per bundled arm; the retained hash must match exactly.
            normalized[path_key] = '<content-addressed-input>'
        return normalized
    def matched_science(config):
        environment = content_bound_paths(config.get('environment'), [('release','manifest_sha256')])
        environment.pop('invalid_action_policy',None)
        evaluation = content_bound_paths(config.get('evaluation'), [('cohort_manifest','cohort_sha256')])
        harness = content_bound_paths(config.get('harness'), [('source_manifest','source_manifest_sha256')])
        stages = [{k:v for k,v in stage.items() if k != 'stability'} for stage in config['stages']]
        return {**{k:config.get(k) for k in ('model','seed','limits','judge','training_reward')},
                'evaluation':evaluation,'harness':harness,'environment':environment,'stages':stages}
    if any(matched_science(c) != matched_science(control) for c in configs[1:]):
        raise ConfigurationError('Candidates must match control except registered stability and invalid-action interventions')
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    state_path = root/'autoresearch-state.json'
    lock_path = process_lock_path(root/'.autoresearch.lock')
    with lock_path.open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ConfigurationError('Another controller owns this autoresearch campaign') from None
        hashes = [digest(c) for c in configs]
        if state_path.exists():
            state = read(state_path)
            if state['config_hashes'] != hashes:
                raise ConfigurationError('Campaign configuration changed; create a new campaign')
            if state['status'] not in {'running', 'prepared'}:
                return state
        else:
            state = {'schema_version':'1.0', 'config_hashes':hashes, 'status':'prepared',
                     'created_at':time.time(), 'incumbent':None, 'decisions':[],
                     'runs':[{'run_id':c['run_id'], 'config_hash':h, 'status':'pending', 'receipts':[]}
                             for c,h in zip(configs,hashes)]}
            atomic_json(state_path,state)
        def save():
            state['updated_at'] = time.time()
            atomic_json(state_path,state)
        for index, record in enumerate(state['runs']):
            if record['status'] == 'running':
                try:
                    _recover(configs[index],record)
                except Exception as exc:
                    state.update(status='stopped_ambiguous', error_type=type(exc).__name__, stopped_run=record['run_id'])
                    save(); return state
                save()
        state['status']='running'; save()
        while True:
            # Finish durable receipts that survived a crash before scoring/selection.
            for index, record in enumerate(state['runs']):
                if record['status']=='completed' and 'summary' not in record:
                    try:
                        record['summary']=_summary(configs[index],record)
                        if record['summary']['training_terminations'].get('infrastructure_error',0):
                            state.update(status='stopped_infrastructure',stopped_run=record['run_id'],
                                         stop_reason='Training infrastructure failures require reconciliation')
                            save(); return state
                        state['decisions'].append(_decide(state,index))
                    except Exception as exc:
                        state.update(status='stopped_infrastructure', error_type=type(exc).__name__, stopped_run=record['run_id'])
                        save(); return state
                    save()
            index, reason = _pick(configs,state)
            if index is None:
                state['status']='complete' if state['incumbent'] else 'complete_inconclusive'
                save(); return state
            config, record = configs[index], state['runs'][index]
            state['decisions'].append({'run_id':record['run_id'],'decision':'execute','reason':reason})
            maximum_invocations = 1 + sum(stage['max_batches'] for stage in config['stages'])
            if record.get('invocations',0) >= maximum_invocations:
                state.update(status='stopped_safety',stopped_run=record['run_id'],stop_reason='Bounded resume limit reached')
                save();return state
            record['invocations']=record.get('invocations',0)+1
            resume = record['status']=='committed'
            events_path=Path(config['output'])/'events.jsonl'
            record.update(status='running', event_offset=events_path.stat().st_size if events_path.exists() else 0,
                          invocation_started_at=time.time())
            save()
            try:
                if resume:
                    path, manifest = _checkpoint(config, record['checkpoint'])
                    if manifest['manifest_hash'] != record['checkpoint_hash'] or manifest['state'].get('stop_reason'):
                        raise ConfigurationError('Committed resume receipt is incompatible or stopped')
                pipeline=(pipeline_factory or Pipeline)(config)
                result=pipeline.run(checkpoint=record['checkpoint'],purpose='resume') if resume else pipeline.run()
                path, manifest = _checkpoint(config,result)
                events=_events(config['output'],record['event_offset'])
                finish=next((e.get('status') for e in reversed(events) if e.get('event')=='run_finish'),None)
                if finish not in {'complete','interrupted_at_committed_boundary'}:
                    record['checkpoint']=path
                    state.update(status='stopped_safety', stopped_run=record['run_id'], stop_reason=finish)
                    save(); return state
                record.update(status='completed' if finish=='complete' else 'committed', checkpoint=path,
                              checkpoint_hash=manifest['manifest_hash'],optimizer_step=manifest['state']['optimizer_step'])
                record['receipts'].append({'checkpoint':path,'manifest_hash':manifest['manifest_hash'],
                    'optimizer_step':record['optimizer_step'],'status':finish,'saved_at':time.time()})
                save()
            except AmbiguousUpdate:
                state.update(status='stopped_ambiguous',stopped_run=record['run_id'])
                record['status']='requires_reconciliation';save();return state
            except BudgetLimit:
                state.update(status='stopped_budget',stopped_run=record['run_id'])
                record['status']='stopped_budget';save();return state
            except Exception as exc:
                # Never classify a failed invocation as resumable merely because
                # an older checkpoint exists: an unacknowledged update may exist.
                state.update(status='stopped_infrastructure',stopped_run=record['run_id'],error_type=type(exc).__name__)
                record['status']='requires_reconciliation';save();return state
