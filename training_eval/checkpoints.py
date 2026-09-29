"""Immutable local manifests pointing to verified backend artifacts."""
import json
import os
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
from collections.abc import Mapping


def _secret_key(key):
    key = str(key).lower().replace('-', '_')
    return key in {'password', 'secret', 'authorization', 'credentials', 'access_token', 'refresh_token', 'api_key', 'wandb_api_key', 'tinker_api_key'} or key.endswith(('_api_key', '_password', '_secret'))


def _has_credentials(value):
    if isinstance(value, Mapping):
        return any(_secret_key(k) or _has_credentials(v) for k, v in value.items())
    return isinstance(value, (list, tuple)) and any(_has_credentials(v) for v in value)


class LocalCheckpointStore:
    def __init__(self, directory):
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)

    def commit(self, backend, *, run_spec, state, bindings, parent):
        checkpoint_id = uuid4().hex
        manifest = {'schema_version': 1, 'checkpoint_id': checkpoint_id,
                    'created_at': datetime.now(timezone.utc).isoformat(),
                    'identity': asdict(backend.identity), 'run_spec': asdict(run_spec),
                    'state': asdict(state), 'bindings': dict(bindings), 'parent': parent}
        if _has_credentials(manifest):
            raise ValueError('Credentials must not be stored in checkpoint configuration')
        json.dumps(manifest, allow_nan=False)
        artifacts = backend.save(checkpoint_id)
        manifest['artifacts'] = asdict(artifacts)
        self._validate(manifest)
        backend.verify_artifacts(artifacts)
        data = json.dumps(manifest, sort_keys=True, indent=2, allow_nan=False) + '\n'
        temporary = self.directory / f'.{checkpoint_id}.pending'
        destination = self.directory / f'{checkpoint_id}.json'
        try:
            with temporary.open('x', encoding='utf-8') as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            os.link(temporary, destination)  # Atomic publish, never overwrite.
            descriptor = os.open(self.directory, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        finally:
            temporary.unlink(missing_ok=True)
        return str(destination)

    def read(self, reference):
        path = Path(reference)
        if not path.is_absolute():
            path = self.directory / path
        with path.open(encoding='utf-8') as stream:
            manifest = json.load(stream)
        self._validate(manifest)
        return manifest

    @staticmethod
    def _validate(manifest):
        if manifest.get('schema_version') != 1:
            raise ValueError('Unsupported checkpoint schema version')
        for key in ('checkpoint_id', 'identity', 'run_spec', 'state', 'bindings', 'artifacts'):
            if key not in manifest:
                raise ValueError(f'Checkpoint is missing {key}')
        identity = manifest['identity']
        if not isinstance(identity, dict) or not all(identity.get(k) for k in ('backend', 'base_model', 'tokenizer', 'renderer')):
            raise ValueError('Checkpoint model identity is incomplete')
        artifacts = manifest['artifacts']
        if not isinstance(artifacts, dict) or not all(isinstance(artifacts.get(k), str) and artifacts[k] for k in ('training', 'sampling')):
            raise ValueError('Both training and sampling artifacts are required')
        if artifacts['training'] == artifacts['sampling']:
            raise ValueError('Training and sampling artifacts must be distinct')
        for purpose, reference in artifacts.items():
            if reference.startswith('tinker://'):
                segment = '/weights/' if purpose == 'training' else '/sampler_weights/'
                if segment not in reference:
                    raise ValueError(f'Wrong artifact purpose for {purpose}')
        if _has_credentials(manifest):
            raise ValueError('Checkpoint contains credential configuration')
        json.dumps(manifest, allow_nan=False)
