"""Grade saved v6 selection answers symmetrically; never generate training data."""
import copy
import hashlib
import random
from pathlib import Path
from training_pipeline.storage import read, atomic_json, digest
from training_pipeline.config import inputs
from training_pipeline.admission import blob, sha
from training_pipeline.remote import prepare

out=Path('reports/grpo-v6-parallel/factual');out.mkdir(exist_ok=True)
if (out/'submission.json').exists():raise ValueError('Existing receipt: never resubmit')
c=read('configs/experiments/grpo-v6-parallel/baseline-v1.json')
rows={r['id']:r for r in inputs(c)['development']}
archive=Path('artifacts/grader-paraphrase-validation-results/artifacts/experiments')
def evaluations(run):
 root=archive/run
 events=[__import__('json').loads(s) for s in (root/'events.jsonl').read_text().splitlines()]
 assert any(e['event']=='run_finish' and e['status']=='complete' for e in events)
 return root,[read(root/'evaluations'/Path(e['artifact']).name) for e in events if e['event']=='evaluation']
broot,be=evaluations('grpo-v6-baseline-seed42')
sroot,se=evaluations('grpo-v6-smoke-seed42-v1')
assert len(be)==1 and len(se)>=2 and se[0]['optimizer_step']==0
selections=[('baseline',broot,be[0]),('smoke_initial',sroot,se[0]),('smoke_final',sroot,se[-1])]
for key in ('data_identity','environment','reward_version'):
 assert all(e[key]==be[0][key] for _,_,e in selections), 'Mismatched comparison condition: '+key
expected=set(read(c['evaluation']['cohort_manifest'])['selection'])
cases=[];automatic=[]
for phase,root,evaluation in selections:
 assert {r['task_id'] for r in evaluation['results']}==expected
 for result in evaluation['results']:
  path=root/'trajectories'/(result['episode_id']+'.json');t=read(path)
  row=copy.deepcopy(rows[t['task_id']]);a=copy.deepcopy(t['submission'])
  entry={'name':phase+'-'+t['episode_id'],'phase':phase,'task_id':t['task_id'],'episode_id':t['episode_id'],
         'termination':t['termination'],'original_verification':t['verification'],'original_split':'development',
         'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'source_trajectory':str(path)}
  if not a or not a['text'].strip():
   automatic.append({**entry,'reward':0.,'status':'resolved','reason':'No submitted answer'});continue
  valid=[];observed={}
  # Reconstruct successfully observed files even when strict evaluation stopped
  # before extraction (for example because a submitted citation was invalid).
  pending=None
  for event in t['events']:
   if event['kind']=='generation':pending=None
   elif event['kind']=='parsed_action':pending=event['value']
   elif event['kind']=='observation' and isinstance(pending,dict):
    if pending.get('tool')=='read_file' and event['value'].get('exit_code')==0:
     path_read=pending['arguments']['path']
     observed[path_read]=row['image_result']['snapshot_files'][path_read]
    pending=None
  for ref in a['citations']:
   if row['image_result']['snapshot_files'].get(ref['path'])!=ref['file_sha256']:continue
   content=blob(row['snapshot_root'],row['public']['repository']['commit'],ref['path'])
   if sha(content)==ref['file_sha256'] and 1<=ref['start_line']<=ref['end_line']<=len(content.splitlines()) and ref['end_line']-ref['start_line']<120:
    valid.append(ref);observed[ref['path']]=ref['file_sha256']
  raw=root/'private'/(t['episode_id']+'.extract.judge-raw.json')
  if raw.exists():
   for ref in read(raw)['request']['untrusted']['repository_catalog']:
    assert row['image_result']['snapshot_files'][ref['path']]==ref['file_sha256']
    observed[ref['path']]=ref['file_sha256']
  a['citations']=valid
  row['split']='train' # invoke training reward for diagnosis only; original split recorded above
  cases.append({**entry,'request':{'episode_id':entry['name'],'question':row['public']['user_prompt'],'answer':a,
      'rubric':row['rubric'],'source_row':row,'observed_files':observed,'answer_evidence':valid}})
random.Random(42).shuffle(cases)
f={'purpose':'Paired factual diagnostic on saved selection answers: independent base, smoke initial and smoke final. No new rollouts, optimization, rubric tuning, or confirmation use. Original strict results retained.',
   'cases':cases,'automatic_results':automatic,'original_evaluations':{phase:e for phase,_,e in selections},'optimizer_updates':0}
f['fixture_hash']=digest(f);atomic_json(out/'fixture.json',f)
c.update(run_id='grpo-v6-paired-factual-v1',output='artifacts/experiments/grpo-v6-paired-factual-v1')
c['execution']['timeout_seconds']=1800;c['tracking']['notes']=f['purpose']
cp=Path('configs/experiments/grpo-v6-parallel/factual-v1.json');atomic_json(cp,c)
r=prepare([str(cp)],'artifacts/grpo-v6-paired-factual-v1-bundle',budget_plan='configs/experiments/project-budget-v4.json')
b=Path(r['bundle']);atomic_json(b/'comparison-fixture.json',f);m=read(b/'bundle.json');m.pop('bundle_id')
m['files']['comparison-fixture.json']=hashlib.sha256((b/'comparison-fixture.json').read_bytes()).hexdigest()
m['atomic_claim_experiment']={'mode':'qwen_validation','fixture':'comparison-fixture.json','reservation_limit_usd':160}
m['note']=f['purpose'];m['bundle_id']=digest(m);atomic_json(b/'bundle.json',m)
r.update(bundle_id=m['bundle_id'],live_cases=len(cases),automatic_cases=len(automatic),reservation_cap=160)
atomic_json(out/'launch.json',r);print(r)
