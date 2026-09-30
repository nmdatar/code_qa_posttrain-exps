import copy,json,shutil,sys,hashlib
from pathlib import Path
original=Path('/Users/ndatar/.codex/worktrees/parallel-experiment-campaigns/action-interview/artifacts/lr-group-v6-sweep-v1-isolated-v2-bundle')
sys.path.insert(0,str(original/'code'))
from training_pipeline.remote import verify_bundle,isolated_resources,operation_estimate,controller_reservation
from training_pipeline.storage import digest
m=verify_bundle(original);old=next(c for c in m['configs'] if c['run_id']=='lr-1e-5-group-4-v6-sweep-v1-seed42')
c=copy.deepcopy(old);run='lr-1e-5-group-4-v6-rerun-v1-seed42';ledger='artifacts/project-budget/lr-1e-5-g4-rerun-v1/'+run+'.json'
c.update(run_id=run,output='/state/artifacts/experiments/'+run)
c['spend']['ledger']='/state/'+ledger
c['tracking']['notes']='Fresh rerun of infrastructure-compromised LR=1e-5 G=4 arm, authorized by user; identical frozen code, data, seed, training and evaluation settings.'
c['tracking']['tags']+=['clean-rerun-v1']
check=copy.deepcopy(c)
for k in ['run_id','output','tracking']:check[k]=old[k]
check['spend']['ledger']=old['spend']['ledger']
assert check==old
local=copy.deepcopy(c)
local['evaluation']['cohort_manifest']=str(original/local['evaluation']['cohort_manifest'].removeprefix('/bundle/'))
est=operation_estimate(local);upper=est['upper_estimate_usd']+controller_reservation(c)
assert upper<=c['spend']['cap_usd']
bundle=Path('artifacts/lr-1e-5-g4-rerun-v1-bundle').resolve();assert not bundle.exists()
shutil.copytree(original,bundle)
m['configs']=[c];m['ledger_seeds']={ledger:None};m['parallel_training']=False
m['budget_allocation']={'project_ceiling_usd':320,'ledger_caps':{ledger:320},'reserves_usd':{}}
m.pop('bundle_id');m['bundle_id']=digest(m)
(bundle/'bundle.json').write_text(json.dumps(m,indent=2)+'\n')
verify_bundle(bundle);names=isolated_resources(m)
report={'status':'prepared','bundle':str(bundle),'bundle_id':m['bundle_id'],'run_id':run,'original_bundle_id':verify_bundle(original)['bundle_id'],'scientific_conditions_unchanged':True,'upper_estimate_usd':upper,'cap_usd':320,'expected_from_prior_healthy_arms_usd':[59.61,68.15],'controller':names[0],'volume':names[1],'training_attempts':128,'initial_final_eval_tasks_each':32,'learning_rate':1e-5,'group_size':4}
Path('reports/lr-1e-5-g4-rerun-v1/preparation.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
