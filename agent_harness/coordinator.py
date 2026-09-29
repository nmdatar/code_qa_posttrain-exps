"""Bounded, resumable orchestration independent of remote execution providers.

The journal must have exactly one coordinator writer. ``persist`` synchronizes
local journal writes to remote durable storage before any external submission.
An ambiguous submission is quarantined rather than retried automatically.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
import time

_TERMINAL = {'completed', 'unresolved'}
_STATES = _TERMINAL | {'pending', 'submitting_rollout', 'active_rollout', 'ready',
                       'submitting_grade', 'active_grade'}
_ID = re.compile(r'[A-Za-z0-9][A-Za-z0-9_-]{0,127}')


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _identity(value):
    return isinstance(value, str) and _ID.fullmatch(value) is not None


def _validate(manifest):
    if not isinstance(manifest, dict) or not _identity(manifest.get('run_id')):
        raise ValueError('Invalid run identity')
    for field in ('max_rollouts', 'max_graders'):
        if type(manifest.get(field)) is not int or manifest[field] <= 0:
            raise ValueError('Concurrency limits must be positive integers')
    seconds = manifest.get('coordinator_seconds', 3600)
    if type(seconds) not in (float, int) or not math.isfinite(seconds) or seconds <= 0:
        raise ValueError('coordinator_seconds must be positive and finite')
    jobs = manifest.get('jobs')
    if not isinstance(jobs, list) or not jobs:
        raise ValueError('A nonempty cohort is required')
    seen, groups = set(), {}
    for job in jobs:
        if not isinstance(job, dict) or any(not _identity(job.get(k)) for k in ('episode_id', 'task_id', 'group_id')):
            raise ValueError('Invalid job identity')
        if job.get('run_id', manifest['run_id']) != manifest['run_id']:
            raise ValueError('Job run identity mismatch')
        if job['episode_id'] in seen:
            raise ValueError('Duplicate episode identity')
        seen.add(job['episode_id'])
        if groups.setdefault(job['group_id'], job['task_id']) != job['task_id']:
            raise ValueError('A rollout group must contain one task')
    return hashlib.sha256(_json(manifest).encode()).hexdigest()


def _write(path, journal):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise ValueError('Journal must not be a symlink')
    descriptor, temporary = tempfile.mkstemp(prefix='.' + path.name + '.', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'w') as stream:
            stream.write(_json(journal) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def coordinate(manifest, executor, state_path, persist=lambda: None, poll_interval=0.1, sleep=time.sleep):
    """Execute one complete cohort through separate rollout/grader worker pools.

    Executor.submit(kind, public_job) returns a durable call ID. Executor.poll(ID)
    returns None while pending or a JSON object with all four job identity fields.
    Rollouts return status='completed'; grades return 'resolved' (or 'completed')
    and reward in [0,1]. All other outcomes become terminal unresolved records.
    Executor.cancel(ID), when provided, is requested on active calls at the
    coordinator deadline. Cancellation is best effort and never changes a
    timed-out episode into a policy failure or a completed training sample.
    The caller must hold exclusive ownership of state_path for this run.
    """
    digest = _validate(manifest)
    if type(poll_interval) not in (int, float) or not math.isfinite(poll_interval) or poll_interval <= 0:
        raise ValueError('poll_interval must be positive and finite')
    path = Path(state_path)
    if path.is_symlink():
        raise ValueError('Journal must not be a symlink')
    jobs = [{**job, 'run_id': manifest['run_id']} for job in manifest['jobs']]
    if path.exists():
        journal = json.loads(path.read_text())
        if journal.get('version') != 1 or journal.get('manifest_sha256') != digest or journal.get('run_id') != manifest['run_id']:
            raise ValueError('Journal manifest or run identity mismatch')
        records = journal.get('episodes')
        if not isinstance(records, dict) or set(records) != {job['episode_id'] for job in jobs}:
            raise ValueError('Journal cohort mismatch')
        for job in jobs:
            entry = records[job['episode_id']]
            if (not isinstance(entry, dict) or entry.get('job') != job
                    or entry.get('state') not in _STATES or not isinstance(entry.get('history'), list)):
                raise ValueError('Invalid journal episode')
            if entry['state'].startswith('active_') and (not isinstance(entry.get('call_id'), str) or not entry['call_id']):
                raise ValueError('Active journal episode missing durable call ID')
            if entry['state'] == 'completed':
                reward, result = entry.get('reward'), entry.get('grade_result')
                if (type(reward) not in (int, float) or not math.isfinite(reward) or not 0 <= reward <= 1
                        or not isinstance(result, dict) or result.get('reward') != reward
                        or result.get('status') not in {'resolved', 'completed'}
                        or any(result.get(k) != job[k] for k in ('run_id', 'episode_id', 'task_id', 'group_id'))):
                    raise ValueError('Invalid completed journal result')
        start = journal.get('started_at')
        if type(start) not in (int, float) or not math.isfinite(start):
            raise ValueError('Invalid journal start time')
    else:
        journal = {'version': 1, 'run_id': manifest['run_id'], 'manifest_sha256': digest,
                   'started_at': time.time(), 'status': 'running', 'episodes': {
                       job['episode_id']: {'job': job, 'state': 'pending', 'history': []} for job in jobs}}
        records = journal['episodes']

    def save():
        _write(path, journal)
        persist()

    def transition(entry, state, **fields):
        entry.update(state=state, **fields)
        entry['history'].append({'state': state, 'at': time.time(), **fields})
        save()

    save()
    # A crash after the pre-submit commit may have launched work without saving
    # its handle. Only explicit operator recovery can resolve that ambiguity.
    for entry in records.values():
        if entry['state'].startswith('submitting_'):
            transition(entry, 'unresolved', reason='ambiguous_submission_requires_manual_recovery')

    def count(state):
        return sum(entry['state'] == state for entry in records.values())

    def submit(entry, kind):
        transition(entry, 'submitting_' + kind, call_id=None)
        try:
            call_id = executor.submit(kind, dict(entry['job']))
        except Exception as exc:
            transition(entry, 'unresolved', reason='submission_failed_requires_manual_recovery', error_type=type(exc).__name__)
            return
        if not isinstance(call_id, str) or not call_id or len(call_id) > 4096:
            transition(entry, 'unresolved', reason='invalid_call_id_requires_manual_recovery')
            return
        if any(event.get('call_id') == call_id for other in records.values() for event in other['history']):
            transition(entry, 'unresolved', reason='duplicate_call_id_requires_manual_recovery', call_id=call_id)
            return
        transition(entry, 'active_' + kind, call_id=call_id)

    def poll(entry, kind):
        try:
            result = executor.poll(entry['call_id'])
        except Exception as exc:
            transition(entry, 'unresolved', reason='remote_call_failed', error_type=type(exc).__name__)
            return
        if result is None:
            return
        if not isinstance(result, dict) or any(result.get(k) != entry['job'][k]
                for k in ('run_id', 'episode_id', 'task_id', 'group_id')):
            transition(entry, 'unresolved', reason='remote_result_identity_mismatch')
            return
        try:
            _json(result)
        except (TypeError, ValueError):
            transition(entry, 'unresolved', reason='invalid_remote_result')
            return
        if kind == 'rollout' and result.get('status') == 'completed':
            transition(entry, 'ready', rollout_result=result)
        elif kind == 'grade' and result.get('status') in {'resolved', 'completed'}:
            reward = result.get('reward')
            if type(reward) not in (int, float) or not math.isfinite(reward) or not 0 <= reward <= 1:
                transition(entry, 'unresolved', reason='invalid_grade_reward', grade_result=result)
            else:
                transition(entry, 'completed', reward=reward, grade_result=result)
        else:
            transition(entry, 'unresolved', reason='remote_' + kind + '_unresolved', **{kind + '_result': result})

    deadline = journal['started_at'] + manifest.get('coordinator_seconds', 3600)
    while any(entry['state'] not in _TERMINAL for entry in records.values()):
        if time.time() >= deadline:
            for entry in records.values():
                if entry['state'] not in _TERMINAL:
                    cancellation = {}
                    if entry['state'] in {'active_rollout', 'active_grade'}:
                        cancel = getattr(executor, 'cancel', None)
                        if callable(cancel):
                            try:
                                cancel(entry['call_id'])
                                cancellation['cancellation_status'] = 'requested'
                            except Exception as exc:
                                cancellation.update(cancellation_status='failed',
                                                    cancellation_error_type=type(exc).__name__)
                        else:
                            cancellation['cancellation_status'] = 'unsupported'
                    transition(entry, 'unresolved', reason='coordinator_deadline_exceeded', **cancellation)
            break
        for entry in records.values():
            if entry['state'] == 'active_grade':
                poll(entry, 'grade')
        for entry in records.values():
            if entry['state'] == 'ready' and count('active_grade') < manifest['max_graders']:
                submit(entry, 'grade')
        # Do not drain remote rollout handles when the bounded ready queue is
        # full. Those calls continue to occupy rollout slots and apply pressure.
        for entry in records.values():
            if entry['state'] == 'active_rollout' and count('ready') < manifest['max_graders']:
                poll(entry, 'rollout')
        for entry in records.values():
            if entry['state'] == 'ready' and count('active_grade') < manifest['max_graders']:
                submit(entry, 'grade')
        for entry in records.values():
            if (entry['state'] == 'pending' and count('active_rollout') < manifest['max_rollouts']
                    and count('ready') < manifest['max_graders']):
                submit(entry, 'rollout')
        if any(entry['state'] not in _TERMINAL for entry in records.values()):
            sleep(min(poll_interval, max(0, deadline - time.time())))

    groups = {}
    for job in jobs:
        groups.setdefault(job['group_id'], []).append(records[job['episode_id']])
    summary = {'run_id': manifest['run_id'], 'status': 'finished', 'expected_episodes': len(jobs),
               'completed_episodes': count('completed'), 'unresolved_episodes': count('unresolved'),
               'episodes': [records[job['episode_id']] for job in jobs], 'groups': {}}
    for group_id, members in groups.items():
        ready = all(entry['state'] == 'completed' for entry in members)
        summary['groups'][group_id] = {'status': 'ready' if ready else 'excluded',
            'episode_ids': [entry['job']['episode_id'] for entry in members],
            'rewards': [entry['reward'] for entry in members] if ready else None}
    journal['status'], journal['summary'] = 'finished', summary
    save()
    return summary
