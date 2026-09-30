"""Compare all frozen validation versions, retaining every failure and repeat."""
import json,sys,statistics,csv
from pathlib import Path
root=Path('reports/grader-paraphrase-validation')
version=sys.argv[1] if len(sys.argv)>1 else 'v3'
p=root/('live-results.json' if version=='v1' else f'live-results-{version}.json');report=json.loads(p.read_text());cases=report['cases'];indexed={c['name']:c for c in cases}
def matches(c):
 e=c['expected_reward_range'];r=c.get('reward');return r is not None and e[0]-1e-8<=r<=e[1]+1e-8
rows=[]
for c in cases:
 g=c.get('grade',{});coverage=g.get('coverage_assessment',{})
 rows.append({'case':c['name'],'task_id':c['task_id'],'family':c['family'],'kind':c['kind'],'reward':c.get('reward'),'within_expected':matches(c),'strict_score':g.get('strict_score'),'strict_status':g.get('strict_status',g.get('status')),'reward_status':g.get('status','error'),'error':c.get('error','')})
repeats={}
for c in cases:
 if c.get('repeat_of'):
  name=c['repeat_of'];repeats.setdefault(name,[indexed[name].get('reward')]);repeats[name].append(c.get('reward'))
spreads={k:None if any(x is None for x in vals) else max(vals)-min(vals) for k,vals in repeats.items()}
allfacts=[c for c in cases if c['kind']=='all-facts'];one=[c for c in cases if c['kind']=='one-fact'];zero=[c for c in cases if c['kind']=='abstention'];reg=[c for c in cases if c['kind']=='regression']
ordered=sum(indexed[c['name'].replace('one-fact','all-facts')].get('reward') is not None and c.get('reward') is not None and indexed[c['name'].replace('one-fact','all-facts')]['reward']>c['reward']+1e-8 for c in one)
summary={'status':report['status'],'attempts':len(cases),'resolved_rewards':sum(c.get('reward') is not None for c in cases),'all_facts_full_credit':sum(matches(c) for c in allfacts),'all_facts_cases':len(allfacts),'one_fact_positive':sum(c.get('reward') is not None and c['reward']>0 for c in one),'one_fact_cases':len(one),'zero_credit_abstentions':sum(matches(c) for c in zero),'abstention_cases':len(zero),'full_strictly_above_one_fact':ordered,'regression_cases_within_expected':sum(matches(c) for c in reg),'regression_cases':len(reg),'repeat_rewards':repeats,'repeat_spreads':spreads,'outside_expectation':[c['name'] for c in cases if not matches(c)],'reserved_usd':report['reserved_usd'],'elapsed_seconds':report['elapsed_seconds']}
summary['engineering_gate_passed']=report['status']=='complete' and len(cases)==92 and summary['resolved_rewards']/92>=.95 and summary['all_facts_full_credit']>=24 and summary['one_fact_positive']>=24 and summary['zero_credit_abstentions']==25 and ordered>=20 and summary['regression_cases_within_expected']==7
with (root/f'case-results-{version}.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(root/f'summary-{version}.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
