"""Archive completed baseline arm only; never reconcile a live shared ledger."""
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import modal
from modal.volume import FileEntryType
root='artifacts/experiments/grpo-v6-baseline-seed42'
out=Path('artifacts/grader-paraphrase-validation-results')
v=modal.Volume.from_name('repository-qa-training-state-v2')
events=[json.loads(s) for s in b''.join(v.read_file(root+'/events.jsonl')).decode().splitlines()]
assert any(e['event']=='run_finish' and e['status']=='complete' for e in events)
paths=[e.path for e in v.listdir(root,recursive=True) if e.type==FileEntryType.FILE]
def get(p):
 data=b''.join(v.read_file(p));dest=out/p.lstrip('/');dest.parent.mkdir(parents=True,exist_ok=True)
 if dest.suffix=='.json':json.loads(data)
 dest.write_bytes(data)
with ThreadPoolExecutor(max_workers=8) as pool:list(pool.map(get,paths))
print('Archived',len(paths),'files')
