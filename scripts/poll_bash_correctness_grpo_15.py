"""Read-only status for the approved isolated bash comparison."""
import json
from pathlib import Path
import os
import certifi
os.environ.setdefault('SSL_CERT_FILE', certifi.where())
import modal
r=json.loads(Path('artifacts/bash-correctness-grpo-15-v1-bundle.submission.json').read_text())
sb=modal.Sandbox.from_id(r['sandbox_id'])
try:
 result={'sandbox_id':r['sandbox_id'],'exit_code':sb.poll()}
 if result['exit_code'] is None:
  source="""
import json
from pathlib import Path
out={}
for arm in ['run']:
 root=Path('/state/artifacts/experiments/bash-correctness-grpo-15-v1')
 traces=[]
 for p in (root/'trajectories').glob('*.json'):
  try:traces.append(json.loads(p.read_text()))
  except (ValueError,OSError):pass
 out[arm]={'traces':len(traces),'graded':sum(bool(t.get('verification')) for t in traces),'evaluations':len(list((root/'evaluations').glob('*.json')))}
events=[]
if (root/'events.jsonl').exists():
 for line in (root/'events.jsonl').read_text().splitlines():
  try:events.append(json.loads(line))
  except ValueError:pass
out['unresolved']=[{'episode':t['episode_id'],'task':t['task_id'],'reasons':t['verification']['reasons']} for t in traces if t.get('verification') and t['verification']['status']=='unresolved'][-4:]
out['phases']={}
for phase in ['train','development']:
 ts=[t for t in traces if t.get('split')==phase]
 vs=[t['verification']['reward'] for t in ts if t.get('verification') and t['verification']['status']=='resolved']
 out['phases'][phase]={'attempts':len(ts),'resolved':len(vs),'correctness_mean':sum(vs)/len(vs) if vs else None}
out['checkpoint_steps']=[{'file':p.name,'step':json.loads(p.read_text()).get('state',{}).get('optimizer_step')} for p in (root/'checkpoints').glob('ckpt-*.json')]
out['events']=[{k:v for k,v in e.items() if k in ('event','optimizer_step','attempted_batches','expected','resolved','mean_reward','mean_correctness_score','mean_citation_score','excluded_groups','zero_variance_groups','contributing_trajectories','acknowledged','error_type','reason','update_diagnostics','status')} for e in events if e.get('event') not in ('trajectory','run_metadata','resolved_models')][-5:]
for p in Path('/state/artifacts/experiments/bash-correctness-grpo-15-v1').glob('tracking-url.json'):
 out['wandb']=json.loads(p.read_text())
for p in Path('/state/campaigns').glob('*/status.json'):
 out['campaign']=json.loads(p.read_text())
for p in Path('/state/campaigns').glob('*/*.log'):
 out[p.name]=p.read_text()[-1200:]
print(json.dumps(out))
"""
  p=sb.exec('python','-c',source,timeout=20);raw=p.stdout.read();code=p.wait()
  if code:raise RuntimeError('status read failed')
  result['progress']=json.loads(raw)
 Path('reports/bash-correctness-grpo-15-v1/live-status.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps(result,indent=2))
finally:sb.detach()
