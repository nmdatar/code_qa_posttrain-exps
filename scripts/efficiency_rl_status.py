"""Read the isolated efficiency pilot without changing remote jobs."""
import json
import os
from pathlib import Path
import certifi
os.environ.setdefault('SSL_CERT_FILE', certifi.where())
import modal

receipt = json.loads(Path('artifacts/efficiency-rl-v1-bundle.submission.json').read_text())
sandbox = modal.Sandbox.from_id(receipt['sandbox_id'])
try:
    code = sandbox.poll()
    print(json.dumps({'controller_exit_code': code}))
    if code is None:
        script = '''
import json
from pathlib import Path
for root in sorted(Path('/state/artifacts/experiments').glob('efficiency-rl-v1-*')):
 events=[]
 if (root/'events.jsonl').exists():
  for line in (root/'events.jsonl').read_text().splitlines():
   try: events.append(json.loads(line))
   except ValueError: pass
 out={'run':root.name,'trajectories':sum(e['event']=='trajectory' for e in events),
      'batches':sum(e['event']=='training_batch' for e in events),
      'updates':sum(e['event']=='update' for e in events),
      'milestones':[{k:v for k,v in e.items() if k in ('event','optimizer_step','status','error_type','accepted_answers','scoring_coverage','mean_output_tokens','mean_compute_units')}
       for e in events if e['event'] in ('evaluation','run_finish','experiment_stop','tracking_failure')],
      'last_event':events[-1].get('event') if events else None,
      'accepted_training_answers':sum(e.get('accepted_answers',0) for e in events if e['event']=='training_batch'),
      'last_batch':next(({k:v for k,v in e.items() if k in ('attempted_batches','mean_reward','accepted_rate','mean_output_tokens','mean_accepted_efficiency_bonus')} for e in reversed(events) if e['event']=='training_batch'),None)}
 if (root/'tracking-url.json').exists(): out['tracking']=json.loads((root/'tracking-url.json').read_text())
 print(json.dumps(out))
for p in Path('/state/campaigns').glob('*/status.json'): print(p.read_text())
'''
        p = sandbox.exec('python', '-c', script, timeout=30)
        print(p.stdout.read())
        print(p.stderr.read())
    else:
        volume = modal.Volume.from_name(receipt['volume'])
        print(b''.join(volume.read_file('campaigns/'+receipt['bundle_id']+'/status.json')).decode())
finally:
    sandbox.detach()
