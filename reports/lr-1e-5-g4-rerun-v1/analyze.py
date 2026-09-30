import json,math
from collections import Counter
from pathlib import Path
base=Path('artifacts/lr-1e-5-g4-rerun-v1-results');name='lr-1e-5-group-4-v6-rerun-v1-seed42';r=base/'artifacts/experiments'/name
rows=[json.loads(x) for x in (r/'events.jsonl').read_text().splitlines() if x.strip()]
evals=[x for x in rows if x['event']=='evaluation'];assert len(evals)==2
tr=[x for x in rows if x['event']=='trajectory' and x['phase']=='training'];batches=[x for x in rows if x['event']=='training_batch']
ledger=json.loads((base/'artifacts/project-budget/lr-1e-5-g4-rerun-v1'/(name+'.json')).read_text())
assert math.isclose(sum(x['upper_estimate_usd'] for x in ledger['reservations']),ledger['reserved_usd'])
result={'run_id':name,'learning_rate':1e-5,'group_size':4,'batches':len(batches),'training_attempts':len(tr),'updates':sum(x['event']=='update' and x.get('acknowledged',False) for x in rows),'training_terminations':dict(Counter(x['termination'] for x in tr)),'resolved_training':sum(x['resolved_trajectories'] for x in batches),'excluded_groups':sum(x['excluded_groups'] for x in batches),'reserved_usd':ledger['reserved_usd'],'actual_billing_usd':None,'tracking':json.loads((r/'tracking-url.json').read_text())}
for label,e in zip(['initial','final'],evals):
 result[label]={k:e[k] for k in ['optimizer_step','checkpoint_id','demonstrated_quality','scoring_coverage','completion_rate','resolved','expected']}
for label in ['initial','final']:
 matching=[json.loads(p.read_text()) for p in (r/'evaluations').glob('*.json') if json.loads(p.read_text())['checkpoint_id']==result[label]['checkpoint_id']]
 assert len(matching)==1
 result[label]['strict_passes']=sum(row['reward']==1 for row in matching[0]['results'])
 result[label]['task_ids']=sorted(row['task_id'] for row in matching[0]['results'])
assert result['initial']['task_ids']==result['final']['task_ids']
path=Path('reports/lr-1e-5-g4-rerun-v1/summary.json');path.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
