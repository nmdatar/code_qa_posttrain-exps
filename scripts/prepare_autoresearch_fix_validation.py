"""Freeze training-only factual controls; no policy rollouts or optimizer calls."""
import copy
import argparse
import hashlib
import random
from pathlib import Path
from training_pipeline.storage import read, atomic_json, digest
from training_pipeline.config import inputs
from training_pipeline.remote import prepare

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--evidence-followup', action='store_true')
args = parser.parse_args()
suffix = '-v2' if args.evidence_followup else ''
out = Path('reports/grpo-autoresearch/fixes-v1') / ('control-v2' if args.evidence_followup else '.')
out.mkdir(parents=True, exist_ok=True)
if (out/'submission.json').exists():
    raise ValueError('Existing submission; do not overwrite its fixture or resubmit')
c = read('configs/experiments/grpo-autoresearch/fixes-v1-training.json')
data = inputs(c); rows = {r['id']:r for r in data['tasks']}
drafts = read('reports/grpo-autoresearch/placeholder-repair-inputs.json')['tasks']
cases = []
for index in ([15] if args.evidence_followup else [0, 2, 15, 55]):
    row = rows[drafts[index]['task_id']]
    for kind, text in [('complete', row['reference']['reference_answer']),
                       ('one_fact', row['rubric']['claims'][0]['text']),
                       ('unrelated', 'This repository contains source files.')]:
        name = row['id'] + '-' + kind
        refs = [{**e, 'id':'c'+str(i+1)} for i,e in enumerate(row['reference']['verified_evidence'])]
        answer = {'schema_version':'1.0','task_id':row['id'],'text':text,'citations':refs,'diagram':None}
        cases.append({'name':name, 'task_id':row['id'], 'control':kind,
                      'claim_count':len(row['rubric']['claims']),
                      'request':{'episode_id':name,'question':row['public']['user_prompt'],'answer':answer,
                         'rubric':row['rubric'],'source_row':copy.deepcopy(row),
                         'observed_files':{e['path']:e['file_sha256'] for e in refs},'answer_evidence':refs}})
random.Random(42).shuffle(cases)
f = {'purpose':'Training-only synthetic factual controls on repaired placeholder rubrics. Complete reference, single reference fact, unrelated answer. Diagnose signal without policy rollouts, optimization or selection/confirmation data.',
     'cases':cases,'optimizer_updates':0, 'expected_order':'complete > one_fact > unrelated; complete near 1 and unrelated 0. Single-fact exact credit can overlap other criteria; inspect per-claim reasons, never relabel.'}
f['fixture_hash'] = digest(f); atomic_json(out/'fixture.json',f)
c.update(run_id='autoresearch-fixes-v1-grader-controls'+suffix, output='artifacts/experiments/autoresearch-fixes-v1-grader-controls'+suffix)
c['execution'].update(operation='baseline',timeout_seconds=1200)
c.pop('best_checkpoint',None);c.pop('stopping',None)
c['stages'][0].update(max_batches=1,max_updates=1)
c['tracking']['notes']=f['purpose']
cp=Path('configs/experiments/grpo-autoresearch/fixes-v1-validation'+suffix+'.json');atomic_json(cp,c)
r=prepare([str(cp)],'artifacts/grpo-autoresearch-fixes-v1-bundle'+suffix,budget_plan='configs/experiments/project-budget-v4.json')
b=Path(r['bundle']);atomic_json(b/'comparison-fixture.json',f)
m=read(b/'bundle.json');m.pop('bundle_id')
m['files']['comparison-fixture.json']=hashlib.sha256((b/'comparison-fixture.json').read_bytes()).hexdigest()
cap = 12 if args.evidence_followup else 35
m['atomic_claim_experiment']={'mode':'qwen_validation','fixture':'comparison-fixture.json','reservation_limit_usd':cap}
m['note']=f['purpose'];m['bundle_id']=digest(m);atomic_json(b/'bundle.json',m)
r.update(bundle_id=m['bundle_id'],live_cases=len(cases),reservation_cap=cap)
atomic_json(out/'launch.json',r);print(r)
