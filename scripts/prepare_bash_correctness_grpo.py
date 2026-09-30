import json
from pathlib import Path
from training_pipeline.storage import read, atomic_json
from training_pipeline.config import validate_config, inputs
from training_pipeline.remote import operation_estimate, controller_reservation
root=Path('configs/experiments/bash-correctness-grpo-v1')
c=read('configs/experiments/fast-grpo-selected-v1/run.json')
c['run_id']='bash-correctness-grpo-v1'
c['output']='artifacts/experiments/'+c['run_id']
c['environment'].update(solver_tools='bash-only-v1',scoring_policy='correctness-only-v1')
c['limits'].update(max_tool_calls=30,max_generations=31,max_output_tokens=6000,context_tokens=65536)
c['stages'][0].update(batch_size=8,group_size=8,max_batches=3,max_updates=3,learning_rate=1e-5)
c['evaluation']['max_tasks']=32
subset=read(c['task_subset']['manifest'])
subset['evaluation_ids']=read(c['evaluation']['cohort_manifest'])['selection']
atomic_json(root/'subset.json',subset)
import hashlib
c['task_subset']={'manifest':str(root/'subset.json'),'sha256':hashlib.sha256((root/'subset.json').read_bytes()).hexdigest()}
c['tracking'].update(mode='online',flush_every=0,experiment_id='bash-correctness-grpo-v1',
 notes='Correctness-only GRPO: 3 batches, 8 tasks x8 attempts, LR1e-5. Unrestricted bash, 30 tool calls. Independent required-fact coverage drives train and eval reward. Citation diagnostics do not affect reward. Before/after 32-task validation, no confirmation.',tags=['bash','correctness-only','grpo','b8-g8','lr1e-5'])
c['spend'].update(ledger='artifacts/project-budget/bash-correctness-grpo-v1/run.json',cap_usd=850)
validate_config(c);data=inputs(c)
assert len(data['tasks'])>=24 and len(data['evaluation_subset']['task_ids'])==32
atomic_json(root/'run.json',c)
e=operation_estimate(c);e['controller_reservation']=controller_reservation(c)
atomic_json('reports/bash-correctness-grpo-v1/preparation.json',e)
print(json.dumps(e,indent=2))
