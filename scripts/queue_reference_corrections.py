"""Create stable, blind correction-review batches assigned to a different worker."""
import json
from pathlib import Path
from dataset_builder.build import read_jsonl, write_json
from dataset_builder.collection import file_sha, load
import hashlib

root=Path.cwd();folder=root/'reports/task-generation-1000';tasks={}
for b in (root/'data/generated/collection-1000-v1').iterdir():
 env=load(b/'public/environment.json')
 for ref in read_jsonl(b/'private/references.jsonl'):
  tasks[ref['id']]={'task_id':ref['id'],'question':ref['question'],'question_sha256':hashlib.sha256(ref['question'].encode()).hexdigest(),'repository':ref['repository'],'snapshot_path':env['snapshot_path'],'source_scope':env.get('source_scope','Primary pinned repository snapshot'),'source_inventory_mode':env.get('source_inventory_mode','checkout'),'bundle':str(b.relative_to(root))}
override_path=folder/'correction-assignment-overrides.json'
overrides=load(override_path) if override_path.exists() else {}
count=0
for path in sorted([*folder.glob('semantic-review-[0-9]*.json'),*(folder/'full-review').glob('batch-*.json')]):
 report=load(path);name=path.stem
 # Full batches modulo3 reflect author, except three appended recovery batches.
 n=int(name.rsplit('-',1)[1]);author=n if name.startswith('semantic') else (n-83 if n>=83 else n%3);worker=overrides.get(name,(author+1)%3)
 rows=[]
 for r in report['tasks']:
  answer=r.get('corrected_reference_answer')
  if not answer:continue
  rows.append({**tasks[r['task_id']],'reference_answer':answer,'reference_sha256':hashlib.sha256(answer.encode()).hexdigest(),'original_reference_sha256':r['reference_sha256']})
 if not rows:continue
 dest=folder/'correction-queue'/f'worker-{worker}'/(name+'.json')
 record={'author_worker':author,'reviewer_worker':worker,'source_review_report':str(path.relative_to(root)),'tasks':rows}
 if dest.exists():
  if load(dest)!=record:raise ValueError('Correction queue changed: '+str(dest))
 else:write_json(dest,record)
 count+=len(rows)
print(json.dumps({'corrections_queued':count}))
