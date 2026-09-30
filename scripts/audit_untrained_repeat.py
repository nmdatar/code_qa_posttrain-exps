import json
from pathlib import Path
base=Path('artifacts/grader-paraphrase-validation-results/artifacts/experiments');arms={}
for name in ['autoresearch-grpo-paginate-v1-seed42','autoresearch-grpo-paginate-v2-seed42']:
 e=next(e for p in (base/name/'evaluations').glob('*.json') if (e:=json.loads(p.read_text()))['optimizer_step']==0)
 arms[name]={r['task_id']:r for r in e['results']}
a,b=arms.values();pairs=[]
for task in a:
 x,y=a[task],b[task]
 ta=json.loads((base/list(arms)[0]/'trajectories'/(x['episode_id']+'.json')).read_text());tb=json.loads((base/list(arms)[1]/'trajectories'/(y['episode_id']+'.json')).read_text())
 pairs.append({'task_id':task,'first_score':x['reward'],'second_score':y['reward'],'same_answer_text':(ta.get('submission') or {}).get('text')==(tb.get('submission') or {}).get('text')})
r={'first_passes':sum(x['reward']==1 for x in a.values()),'second_passes':sum(x['reward']==1 for x in b.values()),'same_answer_text_count':sum(x['same_answer_text'] for x in pairs),'changed_score_count':sum(x['first_score']!=x['second_score'] for x in pairs),'same_text_changed_score_count':sum(x['same_answer_text'] and x['first_score']!=x['second_score'] for x in pairs),'pairs':pairs,'note':'Two fresh pre-update trainers, same policy config, tasks, temp0, environment and grader. Combined generation/judge variation; not independent replication of RL effect.'}
Path('reports/grpo-autoresearch/baseline-repeat-variability.json').write_text(json.dumps(r,indent=2)+'\n');print({k:v for k,v in r.items() if k!='pairs'})
