"""Offline replay of saved TRAINING alias actions through parser/argument validation."""
import json
from pathlib import Path
from types import SimpleNamespace
from training_pipeline.collection import CollectionEpisode, load_collection
from training_pipeline.storage import read, atomic_json
from agent_harness.repository_tools import command

config=read('configs/experiments/grpo-autoresearch/fixes-v1-training.json')
rows={r['id']:r for r in load_collection(config['environment'])['tasks']}
root=Path('artifacts/grader-paraphrase-validation-results/artifacts/experiments/autoresearch-grpo-paginate-v2-seed42/trajectories')
results=[]
for path in sorted(root.glob('*.json')):
    trace=read(path)
    if trace['split']!='train':continue
    for i,generation in enumerate(trace['generations']):
        text=generation['text'].strip()
        if text.startswith('```json') and text.endswith('```'):text=text[7:-3].strip()
        try:action=json.loads(text)
        except (ValueError,TypeError):continue
        if not isinstance(action,dict) or set(action)!={'action','arguments'}:continue
        episode=object.__new__(CollectionEpisode);episode.factory=SimpleNamespace(config=config);episode.row=rows[trace['task_id']]
        revised=episode.parse_action(text)
        entry={'episode_id':trace['episode_id'],'generation':i,'original':action,'normalized':revised}
        try:
            command(revised['tool'],revised['arguments'],episode.row['image_result']['snapshot_files'],paginate_reads=True)
            entry['argument_validation']='passed'
        except (KeyError,TypeError,ValueError) as exc:entry['argument_validation']=str(exc)
        results.append(entry)
report={'training_only':True,'saved_actions':len(results),'argument_validation_passed':sum(r['argument_validation']=='passed' for r in results),
        'policy_rollouts':0,'optimizer_updates':0,'limitation':'Offline parse/argument validation only; does not claim rollout or reward improvement.','results':results}
atomic_json('reports/grpo-autoresearch/fixes-v1/alias-replay.json',report)
print(report['saved_actions'],report['argument_validation_passed'])
