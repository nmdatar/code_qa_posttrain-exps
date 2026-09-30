"""Durable event source; W&B is an optional projection, never training state."""
from __future__ import annotations
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
from datetime import datetime, timezone


def _atomic(path: Path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w') as out:
        json.dump(value, out, sort_keys=True, allow_nan=False)
        out.flush()
        os.fsync(out.fileno())
    os.replace(tmp, path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class EventLog:
    def __init__(self, run_dir):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.run_dir / 'events.jsonl'

    def read(self):
        if not self.path.exists():
            return []
        raw = self.path.read_bytes()
        # An interrupted append may leave a non-committed final line.
        lines = raw.splitlines(keepends=True)
        events = []
        for line in lines:
            if not line.endswith(b'\n'):
                break
            event = json.loads(line)
            if event['event_id'] != len(events) + 1:
                raise ValueError('Non-contiguous event log')
            events.append(event)
        return events

    def emit(self, kind, **fields):
        if not isinstance(kind, str) or not kind:
            raise ValueError('Event kind is required')
        if {'event_id', 'timestamp', 'kind'} & fields.keys():
            raise ValueError('Reserved event field')
        with (self.run_dir / '.events.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            events = self.read()
            event = dict(fields, event_id=len(events) + 1, kind=kind,
                         timestamp=datetime.now(timezone.utc).isoformat())
            encoded = (json.dumps(event, sort_keys=True, allow_nan=False) + '\n').encode()
            with self.path.open('a+b') as out:
                out.seek(0)
                raw = out.read()
                if raw and not raw.endswith(b'\n'):
                    out.truncate(raw.rfind(b'\n') + 1)
                    out.seek(0, os.SEEK_END)
                out.write(encoded)
                out.flush()
                os.fsync(out.fileno())
            descriptor = os.open(self.run_dir, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            return event


def _metrics(event):
    """Do not upload prompts, responses, paths, private grading, or arbitrary strings."""
    result = {'event_id': event['event_id']}
    def visit(obj, prefix=''):
        for key, value in obj.items():
            name = prefix + str(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
                result[name] = value
            elif isinstance(value, dict) and key in {'metrics', 'evaluation', 'usage', 'cost'}:
                visit(value, name + '/')
    visit(event)
    return result


def sync_tracking(run_dir, mode='disabled', project='repo-qa-posttrain', entity=None, wandb_module=None):
    """Replay persisted events. Online monotonic steps tolerate replay after crashes.

    W&B is eventually consistent, not a transactional receipt store. The event log
    is authoritative. Offline writes are queued for `wandb sync`, not uploaded.
    """
    if mode not in {'disabled', 'offline', 'online'}:
        raise ValueError('Unsupported tracking mode')
    directory = Path(run_dir)
    directory.mkdir(parents=True, exist_ok=True)
    if mode == 'disabled':
        return {'status': 'disabled', 'events': 0}
    with (directory / '.tracking.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        state_path = directory / 'tracking-state.json'
        state = json.loads(state_path.read_text()) if state_path.exists() else {}
        identity = {'project': project, 'entity': entity}
        if state and state['destination'] != identity:
            raise ValueError('Tracking destination changed; use a new run')
        run_id = state.get('run_id') or hashlib.sha256(os.urandom(32)).hexdigest()[:24]
        state.update(run_id=run_id, destination=identity)
        # Persist identity before contacting a remote service.
        _atomic(state_path, state)
        cursor_key = mode + '_event_id'
        events = [e for e in EventLog(directory).read() if e['event_id'] > state.get(cursor_key, 0)]
        if not events:
            return {'status': 'up_to_date', 'events': 0, 'run_id': run_id}
        run = None
        try:
            if wandb_module is None:
                import wandb as wandb_module
            run = wandb_module.init(project=project, entity=entity, id=run_id,
                                    resume='allow' if mode == 'online' else None,
                                    mode=mode, dir=str(directory))
            run.define_metric('event_id')
            run.define_metric('*', step_metric='event_id')
            for event in events:
                # W&B ignores replayed historical steps after an interrupted sync.
                run.log(_metrics(event), step=event['event_id'])
            run.finish()
            run = None
            state[cursor_key] = events[-1]['event_id']
            state['last_error'] = None
            _atomic(state_path, state)
            return {'status': 'uploaded' if mode == 'online' else 'queued_offline',
                    'events': len(events), 'run_id': run_id}
        except Exception as exc:
            # Keep the cursor unchanged. Never propagate tracking failures to updates.
            state['last_error'] = type(exc).__name__
            _atomic(state_path, state)
            return {'status': 'pending', 'events': len(events), 'error': type(exc).__name__, 'run_id': run_id}
        finally:
            if run is not None:
                try:
                    run.finish(exit_code=1)
                except Exception:
                    pass
