import json,modal
from pathlib import Path
receipt=json.loads(Path('artifacts/lr-1e-5-g4-rerun-v1-bundle.submission.json').read_text())
s=modal.Sandbox.from_id(receipt['sandbox_id'])
script="""
import json
from pathlib import Path
r=Path('/state/artifacts/experiments/lr-1e-5-group-4-v6-rerun-v1-seed42')
p=r/'events.jsonl';events=[]
if p.exists():
 for line in p.read_text().splitlines():
  try:events.append(json.loads(line))
  except ValueError:pass
out={'trajectories':sum(e.get('event')=='trajectory' for e in events),'batches':sum(e.get('event')=='training_batch' for e in events),'updates':sum(e.get('event')=='update' and e.get('acknowledged',False) for e in events),'infrastructure_errors':sum(e.get('event')=='trajectory' and e.get('termination')=='infrastructure_error' for e in events),'milestones':[{k:e[k] for k in ['event','status','demonstrated_quality','scoring_coverage','optimizer_step'] if k in e} for e in events if e.get('event') in ['evaluation','run_end','run_finished']]}
if (r/'tracking-url.json').exists():out['tracking']=json.loads((r/'tracking-url.json').read_text())
print(json.dumps(out))
"""
try:
 code=s.poll()
 print('controller_exit',code)
 if code is None:
  p=s.exec('python','-c',script,timeout=15)
  print(p.stdout.read());print(p.stderr.read())
finally:s.detach()
