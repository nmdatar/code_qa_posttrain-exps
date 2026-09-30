"""Launch only the user-authorized continuation from seed43 checkpoint26."""
import json
import hashlib
from pathlib import Path
import modal
from training_pipeline.storage import atomic_json
from training_pipeline.remote import verify_bundle

original = json.loads(Path('artifacts/expanded-direct-seed43-v1-isolated-bundle.submission.json').read_text())
receipt = Path('artifacts/direct-seed43-continuation.submission.json')
if receipt.exists():
    raise RuntimeError('Continuation submission already recorded; inspect instead of resubmitting')
if original['bundle_id'] != '086f2af8627463d6ea2f047501eb9a543ec95a3c86baf11b8c81d35454bc7603':
    raise RuntimeError('Unexpected seed43 bundle')
verify_bundle('artifacts/expanded-direct-seed43-v1-isolated-bundle')
frozen = Path('artifacts/direct-seed43-continuation')
for name, expected in json.loads((frozen/'checksums.json').read_text()).items():
    if hashlib.sha256((frozen/name).read_bytes()).hexdigest() != expected:
        raise RuntimeError('Frozen continuation source changed: ' + name)
old = modal.Sandbox.from_id(original['sandbox_id'])
try:
    if old.poll() is None:
        raise RuntimeError('Original controller still running')
finally:
    old.detach()
volume = modal.Volume.from_name(original['volume'])
# A successful prior continuation or an interrupted launch must be inspected,
# never silently re-executed under a fresh controller name.
try:
    prior_status = b''.join(volume.read_file('continuations/direct-seed43-c1/status.json'))
except FileNotFoundError:
    prior_status = None
if prior_status is not None:
    raise RuntimeError('Remote continuation already exists; inspect status')
image = (modal.Image.debian_slim(python_version='3.11').apt_install('git')
    .pip_install('modal==1.5.5', 'tinker==0.30.4', 'transformers==5.17.0', 'jinja2==3.1.6', 'wandb==0.30.0')
    .add_local_dir('artifacts/expanded-direct-seed43-v1-isolated-bundle', '/bundle', copy=True)
    .add_local_dir('artifacts/direct-seed43-continuation', '/continuation', copy=True)
    .env({'PYTHONPATH': '/continuation/code', 'QA_MODAL_WORKER': '1',
          'PYTHONUNBUFFERED': '1', 'PYTHONDONTWRITEBYTECODE': '1'}))
atomic_json(receipt, {'status': 'submission_pending', 'volume': original['volume'],
                      'purpose': 'resume_confirmed_checkpoint26'})
sandbox = modal.Sandbox.create('python', '/continuation/worker.py',
    app=modal.App.lookup('repository-qa-training'), name='qa-direct-seed43-continue-v1',
    image=image, timeout=21600, cpu=(2, 2), memory=(4096, 4096),
    secrets=[modal.Secret.from_name('repository-qa-training-credentials')],
    volumes={'/state': volume}, workdir='/state')
result = {'status': 'submitted', 'sandbox_id': sandbox.object_id, 'volume': original['volume'],
          'controller': 'qa-direct-seed43-continue-v1', 'run_id': 'expanded-direct-seed43-v1-continued',
          'purpose': 'resume_confirmed_checkpoint26'}
atomic_json(receipt, result)
sandbox.detach()
print(json.dumps(result))
