"""Combine unchanged broad cases with explicitly rerun affected cases; never best-of."""
import json
from pathlib import Path
root=Path('reports/grader-paraphrase-validation')
broad=json.loads((root/'live-results-v4.json').read_text());target=json.loads((root/'live-results-v6.json').read_text())
assert broad['status']==target['status']=='complete'
by_name={c['name']:c for c in broad['cases']}
replaced=[]
for case in target['cases']:
    if case['name'] in by_name:
        by_name[case['name']]={**case,'result_source':'targeted-v6'};replaced.append(case['name'])
for name,case in by_name.items():case.setdefault('result_source','broad-v4-unchanged')
# Every actual/conservative input difference identified by the audit was rerun.
impact=json.loads((root/'prompt-impact.json').read_text())
assert {x['case'] for x in impact if x['changed']} <= set(replaced)
assert {c['name'] for c in broad['cases'] if c['task_id']=='import-83110e323c2efed311cc105b'} <= set(replaced)
controls=[c for c in target['cases'] if c['name'] not in by_name]
def passed(c):
    lo,hi=c['expected_reward_range'];r=c.get('reward')
    return r is not None and lo-1e-8<=r<=hi+1e-8
combined={'status':'complete','cases':list(by_name.values()),'purpose':'Final validation assembled from the broad run and mandatory affected-case reruns, not best-of selection. No unchanged case was rerun for a favorable score.','reserved_usd':sum(json.loads((root/filename).read_text())['reserved_usd'] for filename in ['live-results.json','live-results-v2.json','live-results-v3.json','live-results-v4.json','live-results-v5.json','live-results-v6.json']),'elapsed_seconds':None,'optimizer_updates':0,'targeted_extra_controls':controls,'all_targeted_extra_controls_passed':all(passed(c) for c in controls),'replaced_cases':replaced}
(root/'live-results-final.json').write_text(json.dumps(combined,indent=2)+'\n')
print('Final cases:',len(by_name),'replaced:',len(replaced),'extra controls:',len(controls),'extra controls passed:',combined['all_targeted_extra_controls_passed'])
