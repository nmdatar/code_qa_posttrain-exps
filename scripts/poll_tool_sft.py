"""Read a compact Modal campaign snapshot; never mutates the running job."""
import argparse,json
from pathlib import Path
import modal
p=argparse.ArgumentParser();p.add_argument('receipt');p.add_argument('--run-id');a=p.parse_args()
r=json.loads(Path(a.receipt).read_text());s=modal.Sandbox.from_id(r['sandbox_id']);code=s.poll();print('controller_exit',code,flush=True)
v=modal.Volume.from_name(r.get('volume','repository-qa-training-state-v2'))
paths=['campaigns/'+r['bundle_id']+'/status.json']
if a.run_id:paths+=['artifacts/experiments/'+a.run_id+'/events.jsonl']
for path in paths:
 try:
  raw=b''.join(v.read_file(path)).decode('utf-8')
  if '\0' in raw: print(path,'being rewritten; snapshot skipped');continue
  if path.endswith('.jsonl'):
   rows=[json.loads(l) for l in raw.splitlines() if l.strip()];print('events',len(rows),'trajectories',sum(x['event']=='trajectory' for x in rows))
   for x in rows:
    if x['event'] in ['update','evaluation','run_end','stage_complete','experiment_stop']:
     print(json.dumps({k:x[k] for k in ['event','optimizer_step','status','demonstrated_quality','scoring_coverage','updates','attempted_batches','metrics'] if k in x}))
  else: print(raw[:1500])
 except Exception as exc: print(path,type(exc).__name__)

s.detach()
