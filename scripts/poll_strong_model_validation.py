"""Read comparison progress from the existing controller without modifying it."""
import json
import os
from pathlib import Path

import certifi
os.environ.setdefault('SSL_CERT_FILE', certifi.where())
import modal

from training_pipeline.storage import atomic_json, read


receipt = read('artifacts/strong-model-validation-v1-concurrent-bundle.submission.json')
sandbox = modal.Sandbox.from_id(receipt['sandbox_id'])
try:
    code = sandbox.poll()
    result = {'sandbox_id': receipt['sandbox_id'], 'exit_code': code}
    if code is None:
        source = """
import json
from pathlib import Path
out={}
for arm in ('student','strong'):
 root=Path('/state/artifacts/experiments/strong-model-validation-v1-'+arm)
 rows=[]
 for p in (root/'trajectories').glob('*.json'):
  try: rows.append(json.loads(p.read_text()))
  except (ValueError,OSError): pass
 events=[]
 if (root/'events.jsonl').exists():
  for line in (root/'events.jsonl').read_text().splitlines():
   try: events.append(json.loads(line))
   except ValueError: pass
 track=root/'tracking-url.json'
 out[arm]={'recorded_traces':len(rows),'graded_traces':sum(isinstance(r.get('verification'),dict) for r in rows),'events':len(events),'last_event':events[-1].get('event') if events else None,'tracking':json.loads(track.read_text()) if track.exists() else None,'benchmark_complete':(root/'benchmark.json').exists(),'tracking_failures':sum(e.get('event')=='tracking_failure' for e in events)}
print(json.dumps(out))
"""
        process = sandbox.exec('python', '-c', source, timeout=30)
        stdout = process.stdout.read()
        if process.wait() != 0:
            raise RuntimeError('Progress read failed')
        result['arms'] = json.loads(stdout)
    atomic_json('reports/strong-model-validation-v1/live-status.json', result)
    print(json.dumps(result, indent=2))
finally:
    sandbox.detach()
