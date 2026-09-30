"""Read-only live diagnostics; retain no credentials or unrelated state."""
import json,os
from pathlib import Path
import certifi
os.environ['SSL_CERT_FILE']=certifi.where()
import modal
r=json.loads(Path('artifacts/bash-correctness-grpo-v1-bundle.submission.json').read_text())
sb=modal.Sandbox.from_id(r['sandbox_id'])
source='''
import json
from pathlib import Path
from collections import Counter
root=Path('/state/artifacts/experiments/bash-correctness-grpo-v1')
ts=[json.loads(p.read_text()) for p in (root/'trajectories').glob('*.json')]
out={}
for split in ['train','development']:
 rows=[t for t in ts if t['split']==split]
 out[split]={'terminations':dict(Counter(t['termination'] for t in rows)),'tool_calls':dict(Counter(t['usage'].get('tool_calls') for t in rows))}
errors=[]
for t in ts:
 if t.get('verification') and t['verification']['status']=='unresolved':
  detail=[]
  for p in (root/'private').glob(t['episode_id']+'*'):
   if p.suffix=='.json':
    d=json.loads(p.read_text())
    if d.get('validation_error'):detail.append({'file':p.name,'error':d['validation_error']})
  errors.append({'episode':t['episode_id'],'details':detail})
out['training_sequence_sizes']={}
for policy in sorted({t['policy_id'] for t in ts if t['split']=='train'}):
 rows=[len(g['prompt'])+len(g['tokens'])-1 for t in ts if t['split']=='train' and t['policy_id']==policy for g in t['generations']]
 out['training_sequence_sizes'][policy]={'rows':len(rows),'total_input_tokens':sum(rows),'max_sequence_tokens':max(rows) if rows else 0}
out['errors']=errors
print(json.dumps(out))
'''
try:
 p=sb.exec('python','-c',source,timeout=30);raw=p.stdout.read();assert p.wait()==0
 result=json.loads(raw)
 Path('reports/bash-correctness-grpo-v1/live-diagnostics.json').write_text(json.dumps(result,indent=2))
 print(json.dumps(result,indent=2))
finally:sb.detach()
