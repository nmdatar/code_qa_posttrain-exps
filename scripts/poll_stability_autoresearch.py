"""Read-only campaign status and checkpoint progress; never submit or retry."""
import os,json,datetime
from pathlib import Path
import certifi
os.environ.setdefault('SSL_CERT_FILE',certifi.where())
import modal
root=Path(__file__).resolve().parents[1]
receipt=json.loads((root/'artifacts/rl-stability-v1-final-bundle.submission.json').read_text())
sb=modal.Sandbox.from_id(receipt['sandbox_id'])
code=sb.poll();sb.detach()
v=modal.Volume.from_name(receipt['volume'])
def get(path):
 try:return json.loads(b''.join(v.read_file(path)))
 except (json.JSONDecodeError,UnicodeDecodeError) as exc:
  # A volume snapshot can overlap publication. Read the same atomic file inside
  # the live controller, never infer a zero balance from unreadable JSON.
  if code is None:
   live=modal.Sandbox.from_id(receipt['sandbox_id'])
   try:
    proc=live.exec('python','-c',
      'import json,sys; print(json.dumps(json.load(open(sys.argv[1]))))',
      '/state/'+path,timeout=30)
    output=proc.stdout.read()
    if proc.wait()==0:return json.loads(output)
   except Exception as inner:return {'unavailable':type(inner).__name__}
   finally:live.detach()
  return {'unavailable':type(exc).__name__}
 except Exception as exc:return {'unavailable':type(exc).__name__}
base='campaigns/'+receipt['bundle_id']
state=get(base+'/autoresearch-state.json')
result={'observed_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'receipt':receipt,'exit_code':code,
        'remote_status':get(base+'/status.json'),'state':state,'ledger':get('artifacts/project-budget/rl-stability-v1.json')}
ledger=result['ledger'];result['ledger']={k:ledger.get(k) for k in ['cap','reserved_usd','unavailable']}
result['runs']=[]
for run in state.get('runs',[]):
 path='artifacts/experiments/'+run['run_id']
 latest=get(path+'/checkpoints/latest.json')
 details={'run_id':run['run_id'],'status':run['status'],'tracking':get(path+'/tracking-url.json'),'latest':latest}
 if 'path' in latest:
  checkpoint=get(path+'/checkpoints/'+Path(latest['path']).name)
  details['checkpoint']={k:checkpoint.get(k) for k in ['id','state']}
  if isinstance(details['checkpoint'].get('state'),dict):
   details['checkpoint']['state']={k:details['checkpoint']['state'].get(k) for k in ['optimizer_step','attempted_batches','stage','stop_reason']}
 result['runs'].append(details)
(root/'reports/rl-stability/live-status.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
