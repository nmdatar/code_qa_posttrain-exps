import json,random,statistics,math
from collections import Counter
from pathlib import Path
ROOT=Path('/Users/ndatar/.codex/worktrees/parallel-experiment-campaigns/action-interview')
archive=ROOT/'artifacts/lr-group-v6-sweep-v1-results';report=ROOT/'reports/parallel-experiment-campaigns'
runs={};details={}
for p in sorted(archive.glob('artifacts/experiments/*/events.jsonl')):
 rows=[json.loads(l) for l in p.read_text().splitlines()];assert rows[-1]['status']=='complete'
 name=p.parent.name;label=name.split('-v6-')[0];b=[x for x in rows if x['event']=='training_batch'];tr=[x for x in rows if x['event']=='trajectory' and x['phase']=='training'];ev=[x for x in rows if x['event']=='evaluation'];assert len(b)==16 and len(tr)==128 and len(ev)==2
 outs=[{x['task_id']:x for x in rows if x['event']=='trajectory' and x['phase']=='evaluation' and x['optimizer_step']==e['optimizer_step']} for e in ev]
 assert all(len(o)==32 for o in outs)
 ledger=json.loads((archive/'artifacts/project-budget/lr-group-v6-sweep-v1'/(name+'.json')).read_text())
 reserve_sum=sum(x['upper_estimate_usd'] for x in ledger['reservations']);assert math.isclose(reserve_sum,ledger['reserved_usd'],abs_tol=1e-6)
 r={'run_id':name,'batches':len(b),'training_attempts':len(tr),'unique_training_tasks':len(set(x['task_id'] for x in tr)), 'updates':sum(x['event']=='update' and x.get('acknowledged',False) for x in rows),'resolved_training':sum(x['resolved_trajectories'] for x in b),'contributing_trajectories':sum(x['contributing_trajectories'] for x in b),'zero_variance_groups':sum(x['zero_variance_groups'] for x in b),'excluded_groups':sum(x['excluded_groups'] for x in b),'training_terminations':dict(Counter(x['termination'] for x in tr)),'training_input_tokens':sum(x['usage']['input_tokens'] for x in tr),'training_output_tokens':sum(x['usage']['output_tokens'] for x in tr),'all_policy_output_tokens':sum(x['usage']['output_tokens'] for x in rows if x['event']=='trajectory'),'reserved_usd':ledger['reserved_usd'],'actual_invoice_usd':None}
 for stage,e,o in zip(['initial','final'],ev,outs):
  rewards=[x['reward'] for x in o.values()];known=[x for x in rewards if x is not None]
  assert math.isclose(sum(known)/32,e['demonstrated_quality'],abs_tol=1e-8)
  r[stage]={'quality':e['demonstrated_quality'],'strict_passes':sum(x==1 for x in known),'resolved':e['resolved'],'scoring_coverage':e['scoring_coverage'],'completion_rate':e['completion_rate'],'missing_grade_quality_bounds':[sum(known)/32,(sum(known)+32-len(known))/32],'checkpoint_id':e['checkpoint_id']}
 r['change']=r['final']['quality']-r['initial']['quality'];r['valid_training_comparison']=r['training_terminations'].get('infrastructure_error',0)==0
 runs[label]=r;details[label]={'outs':outs,'evals':ev,'training_ids':[x['task_id'] for x in tr]}
ids=sorted(next(iter(details.values()))['outs'][0]);first=next(iter(details.values()))
for d in details.values():
 for out in d['outs']:assert sorted(out)==ids
 for e in d['evals']:
  for k in ['data_identity','environment','reward_version','cohort']:assert e[k]==first['evals'][0][k],k
# Matched question multiplicities within each group size; different group sizes intentionally differ.
for group in [4,8]:assert Counter(details[f'lr-5e-6-group-{group}']['training_ids'])==Counter(details[f'lr-1e-5-group-{group}']['training_ids'])
def interval(values):
 rng=random.Random(42);samples=sorted(statistics.mean(rng.choices(values,k=len(values))) for _ in range(10000))
 return [samples[249],samples[9749]]
def credit(d,s,t):return d['outs'][s][t]['reward'] or 0
contrasts=[]
for a,b in [('lr-5e-6-group-4','lr-1e-5-group-4'),('lr-5e-6-group-8','lr-1e-5-group-8'),('lr-5e-6-group-4','lr-5e-6-group-8'),('lr-1e-5-group-4','lr-1e-5-group-8')]:
 da,db=details[a],details[b];diff=[credit(db,1,t)-credit(da,1,t) for t in ids];changes=[credit(db,1,t)-credit(db,0,t)-credit(da,1,t)+credit(da,0,t) for t in ids]
 fully=[t for t in ids if all(d['outs'][s][t]['reward'] is not None for d in [da,db] for s in [0,1])]
 ab=runs[a]['final']['missing_grade_quality_bounds'];bb=runs[b]['final']['missing_grade_quality_bounds']
 contrasts.append({'a':a,'b':b,'direction':'B minus A','final_difference':statistics.mean(diff),'final_bootstrap95':interval(diff),'difference_in_changes':statistics.mean(changes),'change_bootstrap95':interval(changes),'missing_grade_difference_bounds':[bb[0]-ab[1],bb[1]-ab[0]],'fully_resolved_all_four_count':len(fully),'fully_resolved_final_difference':statistics.mean(credit(db,1,t)-credit(da,1,t) for t in fully) if fully else None,'valid_training_comparison':runs[a]['valid_training_comparison'] and runs[b]['valid_training_comparison']})
logs={}
for p in archive.glob('campaigns/*/*.log'):
 text=p.read_text(errors='replace');logs[p.stem]={'billing_error_mentions':text.count('Error code: 402'),'billing_status_mentions':text.count('billing status')}
result={'runs':runs,'contrasts':contrasts,'log_audit':logs,'total_reserved_usd':sum(r['reserved_usd'] for r in runs.values()),'decision':'Highest observed final score: LR5e-6 group4. No established LR/group winner: all final scoring coverage below95%; one arm compromised by104 infrastructure errors; single-seed32-task selection only.','limits':['Unknown grades contribute zero demonstrated credit, not proof of error. Bounds impute each unknown anywhere in[0,1].','Bootstrap is paired over32tasks, unadjusted descriptive95%interval; no repeated-seed/judge uncertainty or multiplicity correction.','Group8 sees16unique training questions versus32for group4 at fixed128attempts.','Reward is strict evaluator aggregate and can include partial credit; strict passes reported separately.','No confirmation evaluation or extra judge calls.']}
(report/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
