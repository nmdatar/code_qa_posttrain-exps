"""Local-first records and optional best-effort W&B streaming."""
import hashlib
import json
import os
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from collections.abc import Mapping
from uuid import uuid4
from .checkpoints import _secret_key


def _clean(value):
    if is_dataclass(value) and not isinstance(value, type):
        value = asdict(value)
    if isinstance(value, Mapping):
        return {str(k): '[REDACTED]' if _secret_key(k) else _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    return value


class JsonTracker:
    """Single writer per run; external sink failures never fail local recording.

    Stable IDs support logical dedupe. Ambiguous external delivery can produce
    transport duplicates if a service does not offer idempotent writes.
    """
    def __init__(self, directory, run_id, sink=None, *, sinks=()):
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.artifact_directory = self.directory / 'artifacts'
        self.artifact_directory.mkdir(exist_ok=True)
        self.run_id = run_id
        self.sinks = tuple(sinks) + ((sink,) if sink is not None else ())
        self.events_path = self.directory / 'events.jsonl'
        self._delivered = [set() for _ in self.sinks]
        self._receipt_path = self.directory / 'deliveries.jsonl'
        self._sink_keys = [getattr(sink, 'delivery_key', None) for sink in self.sinks]
        if self._receipt_path.exists():
            with self._receipt_path.open(encoding='utf-8') as stream:
                for line in stream:
                    receipt = json.loads(line)
                    for index, key in enumerate(self._sink_keys):
                        if key is not None and key == receipt['sink']:
                            self._delivered[index].add(receipt['event_id'])
        self.sink_errors = []
        if self.events_path.exists():
            with self.events_path.open(encoding='utf-8') as stream:
                if any(json.loads(line)['run_id'] != run_id for line in stream):
                    raise ValueError('Tracking directory belongs to another run')

    def log(self, kind, payload, *, step=0):
        event = {'event_id': uuid4().hex, 'run_id': self.run_id, 'kind': kind,
                 'step': step, 'payload': _clean(payload)}
        encoded = json.dumps(event, sort_keys=True, allow_nan=False)
        with self.events_path.open('a', encoding='utf-8') as stream:
            stream.write(encoded + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        self._deliver(event)
        return event['event_id']

    def _record_error(self, error):
        name = type(error).__name__
        if name not in self.sink_errors:
            self.sink_errors.append(name)
            del self.sink_errors[:-16]

    def _deliver(self, event):
        for index, sink in enumerate(self.sinks):
            if event['event_id'] in self._delivered[index]:
                continue
            try:
                sink.emit(event)
            except Exception as error:
                self._record_error(error)
            else:
                self._delivered[index].add(event['event_id'])
                key = self._sink_keys[index]
                if key is not None:
                    with self._receipt_path.open('a', encoding='utf-8') as stream:
                        stream.write(json.dumps({'sink': key, 'event_id': event['event_id']}) + '\n')
                        stream.flush()
                        os.fsync(stream.fileno())

    def replay(self):
        if self.events_path.exists():
            with self.events_path.open(encoding='utf-8') as stream:
                for line in stream:
                    self._deliver(json.loads(line))

    def artifact(self, name, payload):
        encoded = (json.dumps(_clean(payload), sort_keys=True, allow_nan=False) + '\n').encode()
        digest = hashlib.sha256(encoded).hexdigest()
        destination = self.artifact_directory / f'{digest}.json'
        temporary = self.artifact_directory / f'.{uuid4().hex}.pending'
        try:
            with temporary.open('xb') as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            try:
                os.link(temporary, destination)
            except FileExistsError:
                if destination.read_bytes() != encoded:
                    raise ValueError('Artifact content hash collision')
        finally:
            temporary.unlink(missing_ok=True)
        self.log('artifact', {'name': name, 'path': str(destination), 'sha256': digest})
        return str(destination)

    def close(self):
        self.replay()
        for sink in self.sinks:
            try:
                sink.close()
            except Exception as error:
                self._record_error(error)


class DeliveryDeferred(ConnectionError):
    """A sink is cooling down; its event remains in the local spool."""


class WandbSink:
    """Lazy W&B initialization isolates remote outages from local recording."""
    def __init__(self, project, run_id, *, mode='offline', config=None, wandb_module=None,
                 max_rollouts_per_step=8, directory=None, retry_seconds=30, clock=None, entity=None,
                 group=None, name=None, job_type=None, tags=None, notes=None):
        if mode not in {'online', 'offline', 'disabled'}:
            raise ValueError('W&B mode must be online, offline, or disabled')
        if max_rollouts_per_step < 0:
            raise ValueError('max_rollouts_per_step must be nonnegative')
        if retry_seconds < 0:
            raise ValueError('retry_seconds must be nonnegative')
        self.directory = Path(directory).resolve() if directory is not None else None
        self.retry_seconds = retry_seconds
        self._clock = clock or time.monotonic
        self._retry_at = 0.0
        self.max_rollouts_per_step = max_rollouts_per_step
        self._rollout_counts = {}
        self.module = wandb_module
        self.project, self.run_id, self.mode = project, run_id, mode
        self.entity = entity
        self.run_options = {key: value for key, value in dict(
            group=group, name=name, job_type=job_type, tags=tags, notes=notes).items()
            if value is not None}
        self.config = _clean(config or {})
        destination = [project, run_id, mode]
        if entity is not None:
            destination.append(entity)
        self.delivery_key = 'wandb:' + hashlib.sha256(json.dumps(destination).encode()).hexdigest()
        self.run = None
        self._seen = set()

    def emit(self, event):
        if event['event_id'] in self._seen or self.mode == 'disabled':
            return
        if self._clock() < self._retry_at:
            raise DeliveryDeferred('W&B delivery deferred during retry cooldown')
        try:
            self._emit(event)
        except Exception:
            self._retry_at = self._clock() + self.retry_seconds
            raise
        self._retry_at = 0.0

    def _emit(self, event):
        if self.module is None:
            import wandb
            self.module = wandb
        if self.run is None:
            options = dict(self.run_options)
            if self.entity is not None:
                options['entity'] = self.entity
            if self.directory is not None:
                self.directory.mkdir(parents=True, exist_ok=True)
                options['dir'] = str(self.directory)
            self.run = self.module.init(project=self.project, id=self.run_id, mode=self.mode,
                                        config=self.config, resume='allow', **options)
        payload = event['payload']
        record = {'event_id': event['event_id'], 'optimizer_step': event['step'],
                  'event_kind': event['kind']}
        if event['kind'] == 'checkpoint' and payload.get('reference'):
            # Link metrics by opaque ID; manifest/configuration stays local.
            record['checkpoint/id'] = Path(payload['reference']).stem
        if payload.get('checkpoint'):
            record['checkpoint/id'] = Path(payload['checkpoint']).stem
        if event['kind'] == 'artifact':
            artifact = self.module.Artifact(name=f"trace-{payload['sha256']}", type='training-record')
            artifact.add_file(payload['path'])
            self.run.log_artifact(artifact)
        else:
            def scalar_fields(prefix, value):
                if isinstance(value, Mapping):
                    for key, item in value.items():
                        scalar_fields(f'{prefix}/{key}', item)
                elif isinstance(value, (float, int)) and not isinstance(value, bool):
                    record[prefix] = value
            scalar_fields(event['kind'], payload)
            if event['kind'] in {'rollout', 'rollouts', 'rollout_table'}:
                # Stable event order, first N per optimizer step; local traces keep all.
                rows = payload.get('rows', [payload])
                remaining = max(0, self.max_rollouts_per_step - self._rollout_counts.get(event['step'], 0))
                rows = rows[:remaining]
                if rows:
                    columns = ['task_id', 'episode_id', 'policy_version', 'answer', 'reward',
                               'status', 'reason', 'components', 'trajectory_ref', 'termination']
                    record['rollouts'] = self.module.Table(columns=columns, data=[
                        [json.dumps(row.get(key, {}), sort_keys=True) if key == 'components'
                         else row.get(key) for key in columns] for row in rows])
        self.run.log(record)
        if 'rollouts' in record:
            self._rollout_counts[event['step']] = self._rollout_counts.get(event['step'], 0) + len(rows)
        self._seen.add(event['event_id'])

    def close(self):
        if self.run is not None:
            self.run.finish()
