"""Summarize strict baseline rescoring without mixing repeats or training rewards."""
import csv,json
from pathlib import Path
root=Path('reports/grader-paraphrase-validation')
f=json.loads((root/'baseline-fixture.json').read_text());r=json.loads((root/'baseline-final-results.json').read_text())
if r['status']!='complete' or len(r['cases'])!=len(f['cases']):raise ValueError('Baseline rescoring incomplete')
rows=[]
def cohort(run):return 'confirmation' if 'confirmation' in run else 'selection-repeat' if 'repeat' in run else 'selection'
for c in r['cases']+f['automatic_strict_results']:
 g=c.get('grade',{});old=c['original_verification']
 rows.append({'episode_id':c['name'],'task_id':c['task_id'],'cohort':cohort(c['original_run_id']),
              'old_status':old['status'],'old_strict_score':old.get('reward'),
              'new_status':g.get('status',c.get('status','error')),
              'new_strict_score':g.get('strict_score',c.get('strict_score')),
              'reason':g.get('reason',c.get('error',c.get('reason'))),
              'termination':c['termination'],'source_trajectory':c['source_trajectory']})
assert len(rows)==149 and len({r['episode_id'] for r in rows})==149
with (root/'baseline-episode-results.csv').open('w') as stream:
 w=csv.DictWriter(stream,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def stats(subset):
 n=len(subset);new=sum(x['new_status']=='resolved' for x in subset);old=sum(x['old_status']=='resolved' for x in subset)
 return {'attempts':n,'old_resolved':old,'new_resolved':new,'old_strict_passes':sum(x['old_strict_score']==1 for x in subset),'new_strict_passes':sum(x['new_strict_score']==1 for x in subset),'new_scoring_coverage':new/n,'completed_answers':sum(x['termination']=='completed' for x in subset),'unresolved_or_error':n-new}
s={k:stats([x for x in rows if x['cohort']==k]) for k in ['selection','confirmation','selection-repeat']};s['primary_117']=stats([x for x in rows if x['cohort']!='selection-repeat']);s['all_149_including_repeat']=stats(rows)
result={'status':'complete','cohorts':s,'baseline_scoring_gate_passed':s['selection']['new_resolved']>=31 and s['confirmation']['new_resolved']>=81,'answer_completion_gate_passed':s['primary_117']['completed_answers']/117>=.9,'optimizer_updates':0,'new_policy_rollouts':0,'model_reservations_usd':r['reserved_usd'],'elapsed_seconds':r['elapsed_seconds'],'note':'Strict evaluation routing; repeated selection reported separately. Changed scores reflect grader/rubric changes, not policy improvement.'}
(root/'baseline-summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
