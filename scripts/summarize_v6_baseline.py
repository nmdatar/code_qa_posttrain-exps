from pathlib import Path
from collections import Counter
from training_pipeline.storage import read,atomic_json
root=Path('artifacts/grader-paraphrase-validation-results/artifacts/experiments/grpo-v6-baseline-seed42')
files=list((root/'evaluations').glob('*.json'));assert len(files)==1
r=read(files[0]);unknown=[];traces=[read(root/'trajectories'/(v['episode_id']+'.json')) for v in r['results']]
for t in traces:
 if t['verification']['status']!='resolved':unknown.append({'task_id':t['task_id'],'episode_id':t['episode_id'],'reason':t['verification'].get('reasons',[])})
s={'status':'complete','attempted':r['attempted'],'resolved':r['resolved'],'strict_full_passes':sum(v['reward']==1 for v in r['results']),
 'strict_total_credit':sum(v['reward'] for v in r['results'] if v['reward'] is not None),'strict_demonstrated_quality':r['demonstrated_quality'],
 'strict_mean_over_resolved':r['mean_reward'],'scoring_coverage':r['scoring_coverage'],'terminations':dict(Counter(t['termination'] for t in traces)),
 'optimizer_updates':0,'checkpoints':0,'policy_id':r['policy_id'],'evaluation_artifact':str(files[0]),'wandb':read(root/'tracking-url.json'),
 'unresolved':unknown,'note':'Strict scores; independent factual coverage pending. Arm reservation counter overlaps concurrent smoke and is not isolated cost.'}
atomic_json('reports/grpo-v6-parallel/baseline/summary.json',s);print(s)
