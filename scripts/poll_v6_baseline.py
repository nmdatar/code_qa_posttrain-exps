"""Read-only progress for the frozen baseline; does not mutate shared ledgers."""
import json
from pathlib import Path
import modal
p=Path('reports/grpo-v6-parallel/baseline');receipt=json.loads(Path('reports/grpo-v6-parallel/submission.json').read_text())
v=modal.Volume.from_name(receipt['volume'])
s=modal.Sandbox.from_id(receipt['sandbox_id']);print('controller_exit',s.poll());s.detach()
root='artifacts/experiments/grpo-v6-baseline-seed42'
for name in ['events.jsonl','tracking-url.json']:
 try:
  data=b''.join(v.read_file(root+'/'+name))
  if name.endswith('.jsonl'):
   events=[json.loads(line) for line in data.decode().splitlines() if line.strip()]
   (p/name).write_bytes(data)
   print('trajectories',sum(e['event']=='trajectory' for e in events))
   for e in events:
    if e['event'] in ('evaluation','run_finish'):print({k:e.get(k) for k in ['event','attempted','resolved','demonstrated_quality','scoring_coverage','status','optimizer_step']})
  else:
   obj=json.loads(data);(p/name).write_bytes(data);print('tracking',obj)
 except (FileNotFoundError,UnicodeDecodeError,json.JSONDecodeError):print(name,'not ready')
