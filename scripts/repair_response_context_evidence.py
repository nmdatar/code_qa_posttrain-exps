"""Add a missing training reference excerpt, preserving prior releases and scores."""
import ast
import copy
from pathlib import Path
from training_pipeline.admission import blob, sha, verified_source
from training_pipeline.atomic_claim_experiment import rewrite_release
from training_pipeline.collection import load_collection
from training_pipeline.storage import read, atomic_json

report=Path('reports/grpo-autoresearch/fixes-v1')
c=read('configs/experiments/grpo-autoresearch/fixes-v1-training.json')
row=next(r for r in load_collection(c['environment'])['tasks'] if r['id']=='import-43544c025fdaa9cc2029afcd')
assert row['split']=='train'
path='src/requests/models.py'
content=blob(row['snapshot_root'],row['public']['repository']['commit'],path)
assert sha(content)==row['image_result']['snapshot_files'][path]
cls=next(n for n in ast.parse(content).body if isinstance(n,ast.ClassDef) and n.name=='Response')
node=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='__exit__')
assert ast.dump(node.body[0])==ast.dump(ast.parse('self.close()').body[0]) and len(node.body)==1
ref={'path':path,'start_line':node.lineno,'end_line':node.end_lineno,'file_sha256':sha(content)}
source=Path(c['environment']['release']);target=source.parent/'repo-qa-training-claims-v6'
if target.exists():raise ValueError('Never overwrite a frozen release')
verified_source(source)
import json
records=[json.loads(s) for s in (source/'private/grading.jsonl').read_text().splitlines()]
original=next(r for r in records if r['task_id']==row['id'])
revised=copy.deepcopy(original);revised['verified_evidence'].append(ref)
# Bind the explicit context-exit claim to its own verified implementation.
claim=next(cl for cl in revised['reviewed_claims'] if cl['claim']=='Response context exit calls close.')
claim['evidence']={k:ref[k] for k in ('path','start_line','end_line')}
claim['supporting_evidence']=[]
revised['evidence_revision']='training-context-exit-evidence-v1'
records=[revised if r['task_id']==row['id'] else r for r in records]
audit={'task_id':row['id'],'split':'train','reason':'Strict training control exposed missing __exit__ source.',
       'added_evidence':ref,'verified_text':'\n'.join(content.decode().splitlines()[node.lineno-1:node.end_lineno]),
       'old_reference':original,'scores_relabelled':False,'human_reviewed':False}
rewrite_release(source,target,records,[audit])
m=read(target/'manifest.json');m['claim_revision']='training-placeholder-facts-v1-context-evidence-v1';m['claim_revision_note']='Same 316 explicit training claims as v5; add verified Response.__exit__ excerpt to one training rubric. All evaluation records preserved.';atomic_json(target/'manifest.json',m)
verified_source(target)
summary=read(report/'release-validation.json');atomic_json(report/'release-validation-v5.json',summary)
summary.update(release=str(target.resolve()),manifest_sha256=sha((target/'manifest.json').read_bytes()),evidence_repairs=1,parent_release=str(source))
atomic_json(report/'release-validation.json',summary);atomic_json(report/'context-exit-evidence-repair.json',audit)
print(summary)
