"""Prepare the question-subsection treatment against the existing bash control."""
import os
import certifi
os.environ.setdefault('SSL_CERT_FILE', certifi.where())
from training_pipeline.storage import read, atomic_json
from training_pipeline.config import validate_config, inputs
from training_pipeline.budget import current_prices
from training_pipeline.remote import operation_estimate, controller_reservation
from training_pipeline.prompt_decomposition import VERSION

name = 'bash-prompt-decomposition-grpo-7-v1'
c = read('configs/experiments/bash-correctness-grpo-15-v1/run.json')
c['run_id'] = name
c['output'] = 'artifacts/experiments/' + name
c['stages'][0].update(max_batches=7, max_updates=7)
c['prompt_decomposition'] = {'version':VERSION}
c['tracking'].update(experiment_id=name,
    notes='Question-only Qwen397 subsections generated once for 32 train and 32 selection questions. Original question retained. Fresh 4B seed42 GRPO, b8 g8 LR1e-5, 7 attempted batches. All other scientific settings match bash-correctness-grpo-15-v1. Compare baseline-adjusted correctness at steps 3 and 6; final eval at step7 if reached. No confirmation.',
    tags=['bash','question-decomposition','qwen397','correctness-only','grpo','7-iterations'])
c['spend']['ledger'] = 'artifacts/project-budget/' + name + '/run.json'
c['spend']['prices'] = current_prices(c['model']['base_model'])
c['judge']['prices'] = current_prices(c['judge']['base_model'], sampling_only=True)
validate_config(c)
data = inputs(c)
assert len(data['tasks']) == 32 and len(data['evaluation_subset']['task_ids']) == 32
atomic_json('configs/experiments/'+name+'/run.json', c)
atomic_json('configs/experiments/'+name+'/budget.json', {
    'ledger_caps':{c['spend']['ledger']:c['spend']['cap_usd']},
    'project_ceiling_usd':c['spend']['cap_usd'], 'reserves_usd':{}})
e = operation_estimate(c)
e['controller_reservation'] = controller_reservation(c)
atomic_json('reports/'+name+'/preparation.json', e)
print({'run_id':name, 'upper_reservation_usd':e['upper_estimate_usd']+e['controller_reservation'], 'cap_usd':c['spend']['cap_usd']})
