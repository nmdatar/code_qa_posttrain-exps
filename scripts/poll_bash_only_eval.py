"""Read-only status for the approved isolated bash comparison."""
import json
from pathlib import Path
import os
import certifi
os.environ.setdefault('SSL_CERT_FILE', certifi.where())
import modal
r=json.loads(Path('artifacts/bash-only-eval-v1-funded-bundle.submission.json').read_text())
sb=modal.Sandbox.from_id(r['sandbox_id'])
try:
 result={'sandbox_id':r['sandbox_id'],'exit_code':sb.poll()}
 if result['exit_code'] is None:
  source="""
import json
from pathlib import Path
out={}
for arm in ['structured','bash']:
 root=Path('/state/artifacts/experiments/bash-only-eval-v1-'+arm)
 traces=[]
 for p in (root/'trajectories').glob('*.json'):
  try:traces.append(json.loads(p.read_text()))
  except (ValueError,OSError):pass
 out[arm]={'traces':len(traces),'graded':sum(bool(t.get('verification')) for t in traces),'evaluations':len(list((root/'evaluations').glob('*.json')))}
for p in Path('/state/campaigns').glob('*/status.json'):
 out['campaign']=json.loads(p.read_text())
for p in Path('/state/campaigns').glob('*/*.log'):
 out[p.name]=p.read_text()[-1200:]
print(json.dumps(out))
"""
  p=sb.exec('python','-c',source,timeout=20);raw=p.stdout.read();code=p.wait()
  if code:raise RuntimeError('status read failed')
  result['progress']=json.loads(raw)
 Path('reports/bash-only-eval-v1/live-status.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps(result,indent=2))
finally:sb.detach()
