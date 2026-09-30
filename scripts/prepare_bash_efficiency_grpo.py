import os
import certifi
os.environ.setdefault('SSL_CERT_FILE', certifi.where())
from training_pipeline.storage import read, atomic_json
from training_pipeline.config import validate_config, inputs
from training_pipeline.budget import current_prices
from training_pipeline.remote import operation_estimate, controller_reservation
name='bash-efficiency-grpo-15-v1'
c=read('configs/experiments/bash-correctness-grpo-15-v1/run.json')
c['run_id']=name
c['output']='artifacts/experiments/'+name
c['training_reward']['efficiency_penalty']='output-token-fraction-v1'
c['tracking'].update(experiment_id=name, notes='Matched against bash-correctness-grpo-15-v1. Fresh base, same seed, b8 g8, 15 batches, LR1e-5, fixed 32-task eval initially/every 3 updates/final. Training reward=correctness*(1-0.1*min(total trajectory output tokens/6000,1)); eval remains correctness only. Latency diagnostic, not rewarded.', tags=['bash','efficiency','correctness-first','grpo','b8-g8','15-iterations'])
c['spend']['ledger']='artifacts/project-budget/'+name+'/run.json'
c['spend']['prices']=current_prices(c['model']['base_model'])
c['judge']['prices']=current_prices(c['judge']['base_model'],sampling_only=True)
validate_config(c)
data=inputs(c)
assert len(data['evaluation_subset']['task_ids'])==32
atomic_json('configs/experiments/'+name+'/run.json',c)
atomic_json('configs/experiments/'+name+'/budget.json',{'ledger_caps':{c['spend']['ledger']:c['spend']['cap_usd']},'project_ceiling_usd':c['spend']['cap_usd'],'reserves_usd':{}})
e=operation_estimate(c);e['controller_reservation']=controller_reservation(c)
atomic_json('reports/'+name+'/preparation.json',e)
print({'run_id':name,'estimated_upper_usd':e['upper_estimate_usd']+e['controller_reservation'],'cap_usd':c['spend']['cap_usd']})
