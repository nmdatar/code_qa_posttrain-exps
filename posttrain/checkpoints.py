"""Commit checkpoint references only after backend and client state are durable."""
from pathlib import Path
import time
import uuid
from .storage import atomic, read, digest


def save_checkpoint(run_dir, backend, state, config):
    root = Path(run_dir) / 'checkpoints'
    checkpoint_id = 'step-%06d-%s' % (state['optimizer_step'], uuid.uuid4().hex[:10])
    directory = root / checkpoint_id
    directory.mkdir(parents=True, exist_ok=False)
    atomic(directory / 'pending.json', {'status':'saving','created_at':time.time()})
    refs = backend.save(str(directory / 'backend'))
    if not refs.get('training_state') or not refs.get('sampling_state'):
        raise ValueError('Backend did not return both checkpoint purposes')
    artifacts = {}
    for purpose in ("training_state", "sampling_state"):
        ref = refs[purpose]
        if isinstance(ref, str) and not '://' in ref:
            path = Path(ref)
            if not path.is_file(): raise ValueError('Missing saved backend artifact: ' + purpose)
            import hashlib
            artifacts[str(path.resolve())] = hashlib.sha256(path.read_bytes()).hexdigest()
    body = {'schema_version':'1', 'checkpoint_id':checkpoint_id, 'run_id':config['run_id'],
            'created_at':time.time(), 'backend':config['backend'], 'model':config['model'],
            'state':state, 'config':config, 'references':refs, 'artifact_hashes':artifacts}
    atomic(directory / 'manifest.json', {**body, 'manifest_hash':digest(body)})
    atomic(Path(run_dir)/'latest-checkpoint.json', {'manifest':str((directory/'manifest.json').resolve())})
    return str((directory/'manifest.json').resolve())


def load_checkpoint(path):
    p = Path(path)
    if p.is_dir():
        if (p/'latest-checkpoint.json').exists(): p=Path(read(p/'latest-checkpoint.json')['manifest'])
        else: p=p/'manifest.json'
    m = read(p)
    check = m.pop('manifest_hash', None)
    if digest(m) != check: raise ValueError('Checkpoint manifest integrity mismatch')
    if not m['references'].get('training_state') or not m['references'].get('sampling_state'):
        raise ValueError('Checkpoint is incomplete')
    import hashlib
    for artifact, sha in m['artifact_hashes'].items():
        if hashlib.sha256(Path(artifact).read_bytes()).hexdigest() != sha:
            raise ValueError('Checkpoint artifact integrity mismatch')
    return {**m,'manifest_hash':check}
