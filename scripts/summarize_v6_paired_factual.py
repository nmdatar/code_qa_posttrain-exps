"""Report every paired result, including unknown grades, without relabeling."""
import csv
import random
from pathlib import Path
from training_pipeline.storage import read, atomic_json

out=Path('reports/grpo-v6-parallel/factual');fixture=read(out/'fixture.json');raw=read(out/'results.json')
assert raw['status']=='complete'
rows={p:{} for p in ['baseline','smoke_initial','smoke_final']}
for case in raw['cases']+fixture['automatic_results']:
 rows[case['phase']][case['task_id']]={'reward':case.get('reward'),'episode_id':case['episode_id'],
  'strict_status':case['original_verification']['status'],'strict_reward':case['original_verification']['reward']}
assert all(len(v)==32 for v in rows.values())
summary={'status':'complete','tasks':32,'optimizer_updates':0,'new_policy_rollouts':0,
         'model_reservations_usd':raw['reserved_usd'],'phases':{},'comparisons':{},
         'limitations':'Selection-set diagnostic, one answer per phase/task. Same-family judge. Task bootstrap does not capture repeated generation/judge variance. Confirmation untouched. New rubric/environment reward curves are not directly comparable to old runs.'}
for phase,values in rows.items():
 rewards=[v['reward'] for v in values.values() if v['reward'] is not None]
 e=fixture['original_evaluations'][phase]
 summary['phases'][phase]={'factual_resolved':len(rewards),'factual_attempted':32,
    'factual_mean_over_resolved':sum(rewards)/len(rewards) if rewards else None,
    'factual_full_credit':sum(v==1 for v in rewards),'factual_positive_credit':sum(v>0 for v in rewards),
    'strict_resolved':e['resolved'],'strict_full_passes':sum(v['reward']==1 for v in e['results']),
    'strict_total_credit':sum(v['reward'] for v in e['results'] if v['reward'] is not None),
    'strict_demonstrated_quality':e['demonstrated_quality'],'optimizer_step':e['optimizer_step']}
paired=[]
for first,last in [('smoke_initial','smoke_final'),('baseline','smoke_final'),('baseline','smoke_initial')]:
 key=first+'_to_'+last;deltas=[]
 for ident in sorted(rows[first]):
  a,b=rows[first][ident]['reward'],rows[last][ident]['reward'];delta=None if a is None or b is None else b-a
  if delta is not None:deltas.append(delta)
  paired.append({'comparison':key,'task_id':ident,'before':a,'after':b,'delta':delta})
 rng=random.Random(42)
 boot=sorted(sum(rng.choices(deltas,k=len(deltas)))/len(deltas) for _ in range(10000)) if deltas else []
 summary['comparisons'][key]={'matched_resolved':len(deltas),'mean_paired_delta':sum(deltas)/len(deltas) if deltas else None,
   'bootstrap_95_interval':[boot[250],boot[9749]] if boot else None,
   'improved':sum(d>0 for d in deltas),'worsened':sum(d<0 for d in deltas),'tied':sum(d==0 for d in deltas)}
atomic_json(out/'summary.json',summary)
with (out/'paired-results.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(paired[0]));w.writeheader();w.writerows(paired)
print(summary)
