"""Prepare matched, frozen-policy validation arms; no training or confirmation use."""
import copy
from pathlib import Path
from training_pipeline.storage import read, atomic_json
from training_pipeline.remote import operation_estimate, controller_reservation

root=Path('configs/experiments/bash-only-eval-v1')
base=read('configs/experiments/strong-model-validation-v1/student.json')
base.pop('benchmark',None)
base['execution']['operation']='baseline'
base['execution']['cohort']='selection'
base['execution']['timeout_seconds']=1800
base['evaluation']['max_tasks']=32
base['evaluation']['temperature']=0
base['tracking']['mode']='offline'
base['tracking']['notes']='Matched untrained Qwen3.5-4B on all 32 frozen selection tasks. Bash-only without command/path allowlists versus unchanged structured tools. Same judge and episode limits. No optimizer or confirmation use. Local full traces retained.'
base['tracking']['tags']=['bash-only-eval-v1','validation','no-optimizer']
summary={}
for arm in ('structured','bash'):
 c=copy.deepcopy(base)
 c['run_id']='bash-only-eval-v1-'+arm
 c['output']='artifacts/experiments/'+c['run_id']
 c['spend']['ledger']='artifacts/project-budget/bash-only-eval-v1/'+arm+'.json'
 c['spend']['cap_usd']=50
 if arm=='bash':c['environment']['solver_tools']='bash-only-v1'
 atomic_json(root/(arm+'.json'),c)
 summary[arm]={'estimate':operation_estimate(c),'controller_reservation':controller_reservation(c)}
atomic_json('reports/bash-only-eval-v1/preparation.json',summary)
print(summary)
