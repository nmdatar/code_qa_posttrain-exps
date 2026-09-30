"""Server-owned catalogs and hash-bound source access for the local product."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
from datetime import datetime, timezone
from agent_harness.code_tools import GitRepository, _safe_path

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / 'data/releases/repo-qa-automated-rollouts-v1'
BASE_MODEL = 'Qwen/Qwen3.5-4B'

def credentials_available():
    """Mirror SDK auth sources without exposing or copying credential values."""
    if os.environ.get('TINKER_API_KEY') or os.environ.get('TINKER_CREDENTIAL_CMD'):
        return True
    try:
        from tinker.lib.credentials import JsonCredentialStore, default_credentials_path
        record = JsonCredentialStore(default_credentials_path()).get_default_key()
        return record is not None and bool(record.key)
    except (ImportError, OSError, ValueError):
        return False

def read(path):
    return json.loads(Path(path).read_text())

class PinnedRepository(GitRepository):
    """Git blobs plus explicitly materialized, hash-checked dataset source files.

    The dataset builder materializes Pydantic's internal links/submodule. Never
    follow an arbitrary model-supplied path or accept modified source bytes.
    """
    def __init__(self, path, commit, manifest):
        self.path, self.commit = Path(path).resolve(), commit
        if self._git('cat-file', '-t', commit).strip() != b'commit':
            raise ValueError('Pinned commit unavailable')
        self.expected = manifest['snapshot_files']
        self._entries = {}
        for entry in self._git('ls-tree', '-rz', '--full-tree', commit).split(b'\0'):
            if not entry:
                continue
            meta, raw_path = entry.split(b'\t', 1)
            mode, kind, oid = meta.decode().split()
            name = raw_path.decode()
            if mode in ('100644', '100755') and kind == 'blob' and name in self.expected:
                self._entries[name] = oid
        for name in self.expected:
            _safe_path(name)

    def files(self, pattern='*'):
        import fnmatch
        return sorted(p for p in self.expected if fnmatch.fnmatchcase(p, pattern))

    def blob(self, path, *, bounded=True):
        path = _safe_path(path)
        if path not in self.expected:
            raise ValueError('Not in the pinned source catalog')
        if path in self._entries:
            data = super().blob(path, bounded=bounded)
        else:
            target = (self.path / path).resolve()
            if not target.is_relative_to(self.path) or not target.is_file():
                raise ValueError('Materialized source unavailable')
            if bounded and target.stat().st_size > self.max_blob_bytes:
                raise ValueError('Source exceeds read limit')
            data = target.read_bytes()
        if hashlib.sha256(data).hexdigest() != self.expected[path]:
            raise ValueError('Pinned source hash mismatch')
        return data


def repo_catalog():
    rows = []
    catalog = RELEASE / 'public/environments.jsonl'
    if catalog.exists():
        environments = [json.loads(line) for line in catalog.read_text().splitlines() if line]
        environments.sort(key=lambda e: (e['repository']['family_id'] != 'pydantic-ecosystem', e['repository']['url'], e['repository']['commit']))
        for env in environments:
            family = env['repository']['family_id']
            evidence = RELEASE / 'private/environment-evidence' / env['environment_id']
            source_path = evidence / 'source-environment.json'
            manifest_path = evidence / 'result.json'
            source = read(source_path) if source_path.exists() else {}
            repository = env['repository']
            name = repository['url'].removeprefix('https://github.com/')
            candidates = [ROOT / 'artifacts/repos' / (name.replace('/', '--') + '-' + repository['commit'])]
            if source.get('snapshot_path'):
                candidates.append(Path(source['snapshot_path']))
            location = next((p for p in candidates if p.is_dir()), candidates[0])
            ready = location.is_dir() and manifest_path.exists()
            rows.append(dict(id=env['environment_id'], name=name, commit=repository['commit'],
                execution=env['capability'] == 'execution', ready=ready,
                reason=None if ready else 'Pinned source or environment manifest missing locally',
                example='How does Pydantic validate assignment, and how does a frozen model change that behavior?' if family == 'pydantic-ecosystem' else ('Where is header merging implemented when content_type and Content-Type are both specified?' if family == 'getsentry/responses' else f'How is {name.split("/")[-1]} organized, and where are its main entry points?'),
                source_path=str(location), manifest_path=str(manifest_path)))
    return rows


def open_repository(config):
    manifest = read(config['manifest_path'])
    if manifest.get('commit') != config['commit']:
        raise ValueError('Repository and environment commits differ')
    return PinnedRepository(config['source_path'], config['commit'], manifest), manifest


def public_repo(row):
    return {k: v for k, v in row.items() if k not in ('source_path', 'manifest_path')}


def model_catalog(refresh=True):
    """Discover hosted models; checkpoints remain restricted to project manifests."""
    key_available = credentials_available()
    models = [dict(id='base', name='Qwen3.5 · 4B', kind='base', base_model=BASE_MODEL,
                   model_path=None, renderer='hf-chat-no-thinking-v1', ready=False,
                   reason='Hosted model availability has not been verified' if key_available else 'Run tinker auth login or set TINKER_API_KEY on the backend', run='Base model', step=None)]
    known, by_run = {}, {}
    for path in sorted((ROOT / 'artifacts').glob('**/checkpoints/*.json')):
        if path.name == 'latest.json' or '.pending.' in path.name:
            continue
        try:
            manifest = read(path)
            identity = manifest['identity']
            artifacts = manifest['artifacts']
            sampler = artifacts.get('sampler') or artifacts.get('sampling')
            if not sampler or not sampler.startswith('tinker://'):
                continue
            run_id = sampler[9:].split('/')[0]
            step = manifest.get('state', {}).get('optimizer_step', manifest.get('state', {}).get('updates', manifest.get('state', {}).get('step')))
            label = path.stem
            if step is not None:
                label = manifest.get('config', {}).get('run_id', path.stem) + f' · step {step}'
            row = dict(id=hashlib.sha256(sampler.encode()).hexdigest()[:24], name=label,
                kind='checkpoint', base_model=identity['base_model'], model_path=sampler,
                trained_harness=('bash' if manifest.get('config', {}).get('environment', {}).get('solver_tools') == 'bash-only-v1' else 'structured'),
                renderer=identity.get('renderer'), template_hash=identity.get('template_hash'),
                context_tokens=identity.get('context_tokens', 8192), ready=False,
                reason='Checkpoint availability has not been verified',
                run=manifest.get('config', {}).get('run_id', run_id),
                step=step,
                created_at=None)
            known[sampler] = row
            by_run[run_id] = row
        except (KeyError, ValueError, OSError):
            continue
    warning = None
    if refresh and key_available:
        try:
            import tinker
            service = tinker.ServiceClient(timeout=15, max_retries=0)
            from product_api.hosted import discover
            try:
                # Tinker's access probe skips its indefinite billing retries.
                # Run it before capability discovery so a 402 cannot freeze
                # the model picker or block read-only checkpoint listing.
                if hasattr(service, '_check_accessible'):
                    service._check_accessible()
                hosted = discover(service, BASE_MODEL)
                if hosted:
                    models = hosted
            except Exception as exc:
                warning = ('Tinker inference is blocked by billing status. Check your Tinker balance.'
                           if getattr(exc, 'status_code', None) == 402 else
                           'Tinker hosted-model availability could not be verified.')
            rest = service.create_rest_client()
            offset = 0
            found = set()
            while True:
                result = rest.list_user_checkpoints(limit=100, offset=offset).result(timeout=20)
                for cp in result.checkpoints:
                    run_id = cp.tinker_path[9:].split('/')[0]
                    if run_id not in by_run:
                        continue
                    row = known.get(cp.tinker_path, {**by_run[run_id],
                        'id': hashlib.sha256(cp.tinker_path.encode()).hexdigest()[:24],
                        'name': cp.checkpoint_id, 'model_path': cp.tinker_path, 'step': None})
                    expired = cp.expires_at is not None and cp.expires_at <= datetime.now(timezone.utc)
                    compatible = row.get('renderer') == 'hf-chat-no-thinking-v1'
                    row.update(created_at=cp.time.isoformat(), ready=cp.checkpoint_type == 'sampler' and not expired and compatible,
                        reason='Unsupported checkpoint renderer' if not compatible else ('Expired checkpoint' if expired else
                            ('Training-only checkpoint; sampling weights required' if cp.checkpoint_type != 'sampler' else None)))
                    known[cp.tinker_path] = row
                    found.add(cp.tinker_path)
                cursor = result.cursor
                if not cursor or cursor.offset + cursor.limit >= cursor.total_count:
                    break
                offset = cursor.offset + cursor.limit
            for path, row in known.items():
                if path not in found:
                    row.update(ready=False, reason='Checkpoint expired or is no longer available in Tinker')
        except Exception:
            warning = 'Tinker catalog could not be reached; checkpoint availability is unverified.'
            for row in known.values():
                row.update(ready=False, reason=warning)
    elif not key_available:
        for row in known.values():
            row['reason'] = 'Run tinker auth login or set TINKER_API_KEY to verify checkpoints'
    # Local manifests establish project ownership, not checkpoint existence.
    # Only sampling checkpoints confirmed usable by Tinker belong in the picker.
    models.extend(sorted((row for row in known.values() if row['ready']),
                         key=lambda r: r.get('created_at') or '', reverse=True))
    from product_api.hosted import apply_pricing
    try:
        apply_pricing(models)
    except ValueError as exc:
        warning = str(exc)
    return models, warning
