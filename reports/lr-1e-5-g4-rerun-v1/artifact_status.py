import json,modal
from pathlib import Path
r=json.loads(Path('artifacts/lr-1e-5-g4-rerun-v1-bundle.submission.json').read_text());v=modal.Volume.from_name(r['volume']);base='artifacts/experiments/lr-1e-5-group-4-v6-rerun-v1-seed42'
s=modal.Sandbox.from_id(r['sandbox_id']);print('controller_exit',s.poll());s.detach()
for part in ['evaluations','trajectories','checkpoints']:
 try:
  es=v.listdir(base+'/'+part,recursive=True);print(part,len(es))
  if part=='evaluations':
   for e in es:
    if e.path.endswith('.json'):
     d=json.loads(b''.join(v.read_file(e.path)));print({k:d.get(k) for k in ['optimizer_step','demonstrated_quality','scoring_coverage','completion_rate']})
  if part=='checkpoints':
   print([e.path.rsplit('/',1)[-1] for e in es])
 except Exception as exc:print(part,type(exc).__name__)
