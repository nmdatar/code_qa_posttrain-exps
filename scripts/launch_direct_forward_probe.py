import json,modal
from pathlib import Path
from training_pipeline.storage import atomic_json
r=json.loads(Path('artifacts/expanded-direct-seed42-v2-isolated-bundle.submission.json').read_text());receipt=Path('artifacts/direct-v2-forward-diagnostic.submission.json');assert not receipt.exists()
s=modal.Sandbox.from_id(r['sandbox_id']);assert s.poll() is not None;s.detach()
image=(modal.Image.debian_slim(python_version='3.11').apt_install('git').pip_install('modal==1.5.5','tinker==0.30.4','transformers==5.17.0','jinja2==3.1.6','wandb==0.30.0').add_local_dir('artifacts/direct-v2-forward-diagnostic','/probe',copy=True).env({'PYTHONPATH':'/probe/code','QA_MODAL_WORKER':'1','PYTHONUNBUFFERED':'1'}))
atomic_json(receipt,{'status':'submission_pending','kind':'forward_only','volume':r['volume']})
s=modal.Sandbox.create('python','/probe/probe.py',app=modal.App.lookup('repository-qa-training'),name='qa-direct-v2-forward-v1',image=image,timeout=600,cpu=2,memory=4096,secrets=[modal.Secret.from_name('repository-qa-training-credentials')],volumes={'/state':modal.Volume.from_name(r['volume'])},workdir='/state')
result={'status':'submitted','sandbox_id':s.object_id,'volume':r['volume'],'kind':'forward_only','additional_bound_usd':2};atomic_json(receipt,result);s.detach();print(json.dumps(result))
