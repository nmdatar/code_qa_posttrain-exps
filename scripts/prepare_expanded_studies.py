"""Prepare explicit, unsubmitted three-arm study designs; never authorize spending."""
import copy,json,math
from pathlib import Path
from training_pipeline.storage import atomic_json
from training_pipeline.remote import operation_estimate,controller_reservation

ROOT=Path('configs/experiments/expanded-studies-v1')
REPORT=Path('reports/expanded-studies')

def prepare():
    ROOT.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True)
    base=json.loads(Path('configs/experiments/reinforce-v6/run.json').read_text())
    prices=json.loads(Path('reports/expanded-studies/prices.json').read_text())
    base['spend']['prices']=prices[base['model']['base_model']]
    jp=prices[base['judge']['base_model']]
    base['judge']['prices']={k:jp[k] for k in ['model','prefill','sample','source','checked_at']}
    base['concurrency']={'rollouts':16,'judges':8}
    base['execution']['timeout_seconds']=43200
    summary={}
    for profile,tasks,batch in [('expanded',256,8),('full',858,13)]:
        out=ROOT/profile;out.mkdir(exist_ok=True);entries=[];caps={}
        for seed in [42,43,44]:
            for arm in ['direct','aligned','sft-reinforce']:
                c=copy.deepcopy(base);name=f'{profile}-{arm}-seed{seed}-v1'
                c.update(run_id=name,output='artifacts/experiments/'+name,seed=seed)
                c['stages'][0].update(batch_size=batch,max_batches=tasks//batch,max_updates=tasks//batch)
                c['evaluation']['every']=(tasks//batch)//2
                if arm=='aligned':c['training_reward']['version']='aligned-coverage-v1'
                deps=[]
                if arm=='sft-reinforce':
                    deps=['verified-complete-sft-seed'+str(seed)]
                    c['execution'].update(operation='fork',checkpoint=f'artifacts/experiments/verified-complete-sft-seed{seed}-v1/checkpoints/latest.json')
                ledger=f'artifacts/project-budget/expanded-studies-v1/{name}.json';c['spend']['ledger']=ledger
                c['tracking'].update(experiment_id=f'expanded-studies-v1-{profile}',tags=['expanded-studies-v1',profile,arm,'seed'+str(seed)],notes='Preregistered '+profile+' three-arm comparison. Same full factual grader; aligned arm changes reward penalties. Fresh base except explicitly registered complete-SFT initialization. Fixed final checkpoint, not best selection checkpoint, for locked confirmation. Three paired seeds; confidence intervals and missing-grade sensitivity required. No authorization or launch implied by this file.')
                cost=operation_estimate(c)['upper_estimate_usd']+controller_reservation(c)
                cap=math.ceil(cost*1.10/10)*10;c['spend']['cap_usd']=cap;caps[ledger]=cap
                path=out/(name+'.json');atomic_json(path,c)
                entries.append({'id':name,'arm':arm,'seed':seed,'config':str(path),'training_tasks':tasks,'rollouts':tasks*4,'batches':tasks//batch,'episodes_per_batch':batch*4,'cost_upper_usd':cost,'cap_usd':cap,'dependencies':deps,'submitted':False})
        # Shared data collection, SFT and locked confirmation are separate gates.
        extras={'teacher_collection':2200,'reward_validation':200,'complete_sft_three_seeds':750,'confirmation_12_jobs':1200,'historical_project_allocation':1000,'infrastructure_contingency':1370}
        total=sum(caps.values())+sum(extras.values());ceiling=math.ceil(total/1000)*1000
        budget={'status':'PROPOSED_REQUIRES_USER_BUDGET_APPROVAL','project_ceiling_usd':ceiling,'ledger_caps':caps,'reserves_usd':extras,'authorization':'Not authorized. Existing authorized total remains $1000 until explicit user approval. Teacher/SFT/confirmation allocations must become named ledgers before dispatch without raising this ceiling.'}
        atomic_json(out/'budget-proposal.json',budget)
        record={'status':'designed_not_launched','profile':profile,'seeds':[42,43,44],'training_tasks_per_seed':tasks,'batches_per_job':tasks//batch,'episodes_per_batch':batch*4,'total_rl_rollouts':9*tasks*4,'total_rl_upper_usd':sum(e['cost_upper_usd'] for e in entries),'proposed_total_ceiling_usd':ceiling,'expected_spend_range_usd':[3000,5000] if profile=='expanded' else [9000,14000],'expected_spend_is_not_cap':True,'entries':entries,'gates':['Budget approval','Training-only reward controls resolved and ordered as specified','SFT: >=200 strict-source/token-verified distinct lineages, >=15 families; source audit','Freeze all arms and analysis before confirmation','Do not promote or tune on confirmation']}
        atomic_json(out/'study.json',record);summary[profile]=record
    atomic_json(REPORT/'design-and-costs.json',summary)
    print(json.dumps({k:{key:v[key] for key in ['training_tasks_per_seed','batches_per_job','episodes_per_batch','total_rl_rollouts','total_rl_upper_usd','proposed_total_ceiling_usd','expected_spend_range_usd']} for k,v in summary.items()},indent=2))

if __name__=='__main__':prepare()
