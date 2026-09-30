"""Submit the frozen grading-only gate; never launches aligned RL."""
import json,hashlib
from pathlib import Path
import modal
from modal.exception import NotFoundError
from training_pipeline.storage import atomic_json
from training_pipeline.remote import verify_bundle,isolated_names

root=Path('artifacts/aligned-live-controls-v1');manifest=json.loads((root/'manifest.json').read_text())
receipt=Path('artifacts/aligned-live-controls-v1.submission.json')
if receipt.exists():raise RuntimeError('Control submission already recorded; inspect instead of repeat')
for name,expected in json.loads((root/'checksums.json').read_text()).items():
    if hashlib.sha256((root/name).read_bytes()).hexdigest()!=expected:raise RuntimeError('Frozen gate changed: '+name)
verify_bundle('artifacts/expanded-direct-seed44-v1-isolated-bundle')
controller,volume_name=isolated_names(manifest['control_bundle_id'])
try:
    modal.Sandbox.from_name('repository-qa-training', controller)
except NotFoundError:
    pass
else:
    raise RuntimeError('Named control controller exists; inspect instead of resubmit')
# A genuinely fresh volume is required, including after an unrecorded failed launch.
try:
    existing = modal.Volume.from_name(volume_name)
    list(existing.iterdir('/'))
except NotFoundError:
    pass
else:
    raise RuntimeError('Control volume already exists; inspect instead of reusing output')
image=(modal.Image.debian_slim(python_version='3.11').apt_install('git')
 .pip_install('modal==1.5.5','tinker==0.30.4','transformers==5.17.0','jinja2==3.1.6','wandb==0.30.0')
 .add_local_dir('artifacts/expanded-direct-seed44-v1-isolated-bundle','/bundle',copy=True)
 .add_local_dir(str(root),'/control',copy=True)
 .env({'PYTHONPATH':'/control/code','QA_MODAL_WORKER':'1','PYTHONUNBUFFERED':'1','PYTHONDONTWRITEBYTECODE':'1'}))
volume=modal.Volume.from_name(volume_name,create_if_missing=True)
atomic_json(receipt,{'status':'submission_pending','volume':volume_name,'controller':controller,
                     'control_bundle_id':manifest['control_bundle_id'],'purpose':'training_only_live_reward_controls'})
sandbox=modal.Sandbox.create('python','/control/worker.py',app=modal.App.lookup('repository-qa-training'),
 name=controller,image=image,timeout=7200,cpu=(2,2),memory=(4096,4096),
 secrets=[modal.Secret.from_name('repository-qa-training-credentials')],volumes={'/state':volume},workdir='/state')
result={'status':'submitted','sandbox_id':sandbox.object_id,'volume':volume_name,'controller':controller,
 'control_bundle_id':manifest['control_bundle_id'],'fixture_hash':manifest['fixture_hash'],
 'purpose':'training_only_live_reward_controls','private_cap_usd':100,'optimizer_calls':0}
atomic_json(receipt,result);sandbox.detach();print(json.dumps(result,indent=2))
