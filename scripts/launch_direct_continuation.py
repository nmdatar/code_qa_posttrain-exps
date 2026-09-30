import json,modal
from pathlib import Path
from training_pipeline.storage import atomic_json
r=json.loads(Path('artifacts/expanded-direct-seed42-v2-isolated-bundle.submission.json').read_text());receipt=Path('artifacts/direct-v2-continuation.submission.json');assert not receipt.exists()
for source in ['artifacts/expanded-direct-seed42-v2-isolated-bundle.submission.json','artifacts/direct-v2-forward-diagnostic-v3.submission.json']:
 prior=json.loads(Path(source).read_text());s=modal.Sandbox.from_id(prior['sandbox_id']);assert s.poll() is not None;s.detach()
v=modal.Volume.from_name(r['volume']);probe=json.loads(b''.join(v.read_file('diagnostics/direct-v2-forward-v3/result.json')));assert probe['status']=='passed',probe
image=(modal.Image.debian_slim(python_version='3.11').apt_install('git').pip_install('modal==1.5.5','tinker==0.30.4','transformers==5.17.0','jinja2==3.1.6','wandb==0.30.0').add_local_dir('artifacts/expanded-direct-seed42-v2-isolated-bundle','/bundle',copy=True).add_local_dir('artifacts/direct-v2-continuation','/continuation',copy=True).env({'PYTHONPATH':'/continuation/code','QA_MODAL_WORKER':'1','PYTHONUNBUFFERED':'1','PYTHONDONTWRITEBYTECODE':'1'}))
atomic_json(receipt,{'status':'submission_pending','volume':r['volume'],'purpose':'resume_confirmed_checkpoint2'})
s=modal.Sandbox.create('python','/continuation/worker.py',app=modal.App.lookup('repository-qa-training'),name='qa-direct-seed42-continue-v1',image=image,timeout=21600,cpu=(2,2),memory=(4096,4096),secrets=[modal.Secret.from_name('repository-qa-training-credentials')],volumes={'/state':v},workdir='/state')
x={'status':'submitted','sandbox_id':s.object_id,'volume':r['volume'],'controller':'qa-direct-seed42-continue-v1','run_id':'expanded-direct-seed42-v2-continued','purpose':'resume_confirmed_checkpoint2'};atomic_json(receipt,x);s.detach();print(json.dumps(x))
