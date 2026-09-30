"""Archive campaign receipt/logs and reconcile spending only after both arms finish."""
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import modal
from modal.volume import FileEntryType
from training_pipeline.remote import reconcile_ledger
from training_pipeline.storage import atomic_json, read

p=Path('reports/grpo-v6-parallel');receipt=read(p/'submission.json')
s=modal.Sandbox.from_id(receipt['sandbox_id']);status=s.poll();s.detach()
assert status==0,('Controller not successfully finished',status)
v=modal.Volume.from_name(receipt['volume']);prefix='campaigns/'+receipt['bundle_id']
result=json.loads(b''.join(v.read_file(prefix+'/status.json')))
assert result['status']=='complete' and all(r['exit_code']==0 for r in result['results']),result
atomic_json(p/'campaign-status.json',result)
paths=[e.path for e in v.listdir(prefix,recursive=True) if e.type==FileEntryType.FILE]
def get(path):
 dest=Path('artifacts/grader-paraphrase-validation-results')/path.lstrip('/');dest.parent.mkdir(parents=True,exist_ok=True)
 data=b''.join(v.read_file(path))
 if dest.suffix=='.json':json.loads(data)
 dest.write_bytes(data)
with ThreadPoolExecutor(max_workers=8) as pool:list(pool.map(get,paths))
ledger=Path('artifacts/project-budget/02-direct-grpo.json');r=json.loads(b''.join(v.read_file(str(ledger))))
reconcile_ledger(r,read(ledger));atomic_json(ledger,r)
before=read(p/'budget-before.json')['reserved_usd']
summary={'reserved_usd':r['reserved_usd'],'cap_usd':r['cap'],'campaign_reservations_usd':r['reserved_usd']-before,
         'actual_provider_billing_usd':None,'active_training_experiments':0}
atomic_json(p/'budget-after-training.json',summary);print(summary)
