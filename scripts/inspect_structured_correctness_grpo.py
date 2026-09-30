"""Read live controller state; never resubmit or replay training updates."""
import json
import os
from pathlib import Path

import certifi
os.environ.setdefault('SSL_CERT_FILE', certifi.where())
import modal

RUN = 'structured-correctness-grpo-15-v1'
receipt = json.loads(Path(f'artifacts/{RUN}-launch-bundle.submission.json').read_text())
sb = modal.Sandbox.from_id(receipt['sandbox_id'])
try:
    result = {'sandbox_id': receipt['sandbox_id'], 'exit_code': sb.poll()}
    if result['exit_code'] is None:
        source = '''
import json
from pathlib import Path
root=Path('/state/artifacts/experiments/structured-correctness-grpo-15-v1')
events=[]
if (root/'events.jsonl').exists():
 for line in (root/'events.jsonl').read_text().splitlines():
  try: events.append(json.loads(line))
  except ValueError: pass
out={'trajectories':sum(e['event']=='trajectory' for e in events),
     'attempted_batches':sum(e['event']=='training_batch' for e in events),
     'optimizer_updates':sum(e['event']=='update' for e in events),
     'tracking_failures':sum(e['event']=='tracking_failure' for e in events),
     'evaluations':[{k:v for k,v in e.items() if k in ('optimizer_step','mean_reward','resolved','expected','wall_seconds')} for e in events if e['event']=='evaluation'],
     'recent_events':[e['event'] for e in events[-5:]]}
if (root/'tracking-url.json').exists(): out['tracking']=json.loads((root/'tracking-url.json').read_text())
for p in Path('/state/campaigns').glob('*/status.json'): out['campaign']=json.loads(p.read_text())
print(json.dumps(out))
'''
        proc = sb.exec('python', '-c', source, timeout=30)
        raw = proc.stdout.read()
        if proc.wait():
            raise RuntimeError('Remote status read failed')
        result['progress'] = json.loads(raw)
    out = Path('reports') / RUN / 'live-status.json'
    out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
finally:
    sb.detach()
