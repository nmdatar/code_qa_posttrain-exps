"""Paired diagnostic comparison; retain missing grades rather than imputing zero."""
import csv,json,random,statistics,math
from pathlib import Path
p=Path('reports/grpo-factual-comparison');f=json.loads((p/'fixture.json').read_text());r=json.loads((p/'results.json').read_text())
assert r['status']=='complete' and len(r['cases'])==len(f['cases'])
indexed={}
for c in r['cases']+f['automatic_results']:
 status=c.get('grade',{}).get('status',c.get('status','error'));reward=c.get('reward') if status=='resolved' else None
 indexed.setdefault(c['task_id'],{})[c['phase']]={**c,'factual_reward':reward,'factual_status':status}
assert len(indexed)==32 and all(set(v)=={'base','final'} for v in indexed.values())
rows=[]
for task,arms in sorted(indexed.items()):
 a,b=arms['base'],arms['final'];x,y=a['factual_reward'],b['factual_reward']
 rows.append({'task_id':task,'base_reward':x,'final_reward':y,'delta':y-x if x is not None and y is not None else None,'base_status':a['factual_status'],'final_status':b['factual_status'],'base_strict':a['original_verification']['reward'],'final_strict':b['original_verification']['reward'],'base_episode':a['episode_id'],'final_episode':b['episode_id']})
with (p/'paired-results.csv').open('w',newline='') as s:
 w=csv.DictWriter(s,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
paired=[x for x in rows if x['delta'] is not None];d=[x['delta'] for x in paired];rng=random.Random(42)
boot=sorted(statistics.fmean(rng.choices(d,k=len(d))) for _ in range(10000)) if d else []
s={'status':'complete','post_hoc_diagnostic':True,'tasks':32,'matched_resolved_pairs':len(d),'model_reservations_usd':r['reserved_usd'],'optimizer_updates':0,'new_policy_rollouts':0,'grader_identity':r['grader_identity']}
for phase in ['base','final']:
 values=[x[phase+'_reward'] for x in rows if x[phase+'_reward'] is not None]
 s[phase]={'resolved':len(values),'mean_over_resolved':statistics.fmean(values) if values else None,'matched_mean':statistics.fmean(x[phase+'_reward'] for x in paired) if paired else None,'full_credit':sum(v==1 for v in values),'positive_credit':sum(v>0 for v in values),'original_strict_passes':sum(x[phase+'_strict']==1 for x in rows),'original_strict_resolved':sum(x[phase+'_strict'] is not None for x in rows)}
s.update(mean_paired_delta=statistics.fmean(d) if d else None,bootstrap_95_percent_interval=[boot[249],boot[9749]] if boot else None,improved=sum(v>1e-9 for v in d),worsened=sum(v< -1e-9 for v in d),tied=sum(abs(v)<=1e-9 for v in d))
s['limitations']='One stochastic generation per task per checkpoint; post-hoc selection-set diagnostic; same model family judges; bootstrap samples tasks and does not estimate repeated generation/judge variability; missing grades excluded from paired mean; no confirmation-set conclusion.'
(p/'summary.json').write_text(json.dumps(s,indent=2)+'\n');print(json.dumps(s,indent=2))
