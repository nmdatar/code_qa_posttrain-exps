"""Prepare the current experiment suite offline; never submit or allocate providers."""
import argparse
import copy
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
import shutil

from training_pipeline.config import inputs
from training_pipeline.storage import atomic_json, digest
from training_pipeline.collection import CollectionFactory
from training_pipeline.benchmark import task_manifest, selected_tasks
from training_pipeline.harness_variants import source_manifest
from training_pipeline.remote import operation_estimate, controller_reservation


def prepare(destination):
    root=Path(destination)
    if root.exists():raise ValueError('Use a fresh versioned output directory')
    campaign = root.name
    if not re.fullmatch(r'[A-Za-z0-9_-]+', campaign):
        raise ValueError('Use a simple versioned directory name for unique run identities')
    base=json.loads(Path('configs/experiments/grpo-long-v6/run.json').read_text())
    data=inputs(base)
    from transformers import AutoTokenizer
    from training_pipeline.rendering import ChatRenderer
    from training_pipeline.sft_collection import validate_rendering
    from training_pipeline.investigation_sft import candidate, VERSION
    tokenizer=AutoTokenizer.from_pretrained(base['model']['base_model'],local_files_only=True)
    renderer=ChatRenderer(tokenizer,base['limits']['context_tokens'])
    version=CollectionFactory(base,Path('/tmp/experiment-preparation'),None).reward_version
    tasks={r['id']:r for r in data['tasks']}
    choices={};rejected=Counter();passing=0
    for path in sorted(Path('artifacts').glob('**/trajectories/*.json')):
        try:trace=json.loads(path.read_text())
        except (ValueError,OSError):continue
        if trace.get('task_id') not in tasks or trace.get('split') != 'train':continue
        if (trace.get('verification') or {}).get('diagnostics',{}).get('strict_score') != 1:continue
        passing+=1
        try:
            example=candidate(trace,tasks[trace['task_id']],version)
            validate_rendering(renderer,[example])
            # Deterministic deduplication; prefer fewer supervised tokens.
            tokens=sum(len(g['generation']['tokens']) for g in example['proofs'])
            ident=example['lineage_id'];rank=(tokens,str(path))
            if ident not in choices or rank<choices[ident][0]:choices[ident]=(rank,example,path)
        except (ValueError,KeyError,TypeError) as exc:rejected[str(exc)]+=1
    if not choices:raise ValueError('No current strictly verified investigation demonstrations')
    root.mkdir(parents=True)
    source=root/'investigation-sft';(source/'sources').mkdir(parents=True)
    records=[]
    for _,example,path in sorted(choices.values(),key=lambda x:x[1]['id']):
        target=source/'sources'/(example['id']+'.json');shutil.copyfile(path,target)
        records.append({'source':str(target.relative_to(source)),'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
                        'example_hash':digest(example),'task_id':example['task_id'],'original_path':str(path)})
    manifest={'version':VERSION,'data_identity':data['identity'],'human_reviewed':False,
              'grading_version':version,'records':records,'scope':'Complete strictly passing policy-generated investigations; not expert demonstrations',
              'selection':'All admitted distinct training lineages; shortest supervised token sequence per lineage'}
    atomic_json(source/'manifest.json',manifest)
    supervised={'manifest':str((source/'manifest.json').resolve()),'sha256':hashlib.sha256((source/'manifest.json').read_bytes()).hexdigest()}
    audit={'strict_passing_candidates':passing,'admitted_lineages':len(records),'rejections':dict(rejected),
           'selected_target_tokens':sum(sum(len(g['generation']['tokens']) for g in e['proofs']) for _,e,_ in choices.values()),
           'families':sorted({e['family_id'] for _,e,_ in choices.values()}),'native_rendering':'passed',
           'human_reviewed':False,'new_teacher_calls':0,'scope':'Small verified-investigation SFT screen; statistical power is limited'}
    atomic_json(source/'admission.json',audit)
    atomic_json(root/'source-index-manifest.json',source_manifest(data))
    source_hash=hashlib.sha256((root/'source-index-manifest.json').read_bytes()).hexdigest()
    benchmark=task_manifest(data);atomic_json(root/'throughput-tasks.json',benchmark)
    configs={};entries=[]
    def fresh(name,operation='run',cohort=None):
        c=copy.deepcopy(base);c['run_id']=name+'-'+campaign+'-seed42';c['output']='artifacts/experiments/'+c['run_id']
        c['execution'].update(operation=operation,timeout_seconds=14400)
        if cohort:c['execution']['cohort']=cohort
        c['tracking'].update(notes='Prepared '+campaign+' experiment. Data v6, strict v7, definition context, action alias and pagination. No live launch implied.',tags=[campaign,name,'training-claims-v6'])
        c['spend']['ledger']='artifacts/project-budget/'+campaign+'/'+name+'.json'
        return c
    def add(name,c,dependencies=(),scope=None):
        cost=operation_estimate(c)['upper_estimate_usd']+controller_reservation(c)
        c['spend']['cap_usd']=math.ceil(cost*1.10/10)*10
        configs[name]=c
        atomic_json(root/(name+'.json'),c)
        entries.append({'id':name,'config':str(root/(name+'.json')),'dependencies':list(dependencies),
                        'scope':scope or 'planned matched study','estimated_upper_usd_including_controller':cost,
                        'cap_usd':c['spend']['cap_usd'],'submitted':False})
    for name,cohort in [('01-baseline','selection'),('01-baseline-repeat','selection'),('01-confirmation','confirmation')]:
        add(name,fresh(name,'baseline',cohort),scope='confirmation only after finalist/analysis lock' if cohort=='confirmation' else 'matched base control')
    for n in [1,8,16,32]:
        for rep in [1,2]:
            name=f'01-throughput-c{n:02d}-r{rep}';c=fresh(name,'benchmark');c['concurrency']['rollouts']=n
            c['benchmark']={'attempts':4,'task_manifest':str(root/'throughput-tasks.json'),'manifest_hash':benchmark['manifest_hash']}
            add(name,c,scope='untrained throughput; never overlap timing arms')
    add('02-direct-grpo',fresh('02-direct-grpo'),scope='fresh control under current code; existing historical runs retained')
    # Full small factorial eliminates a placeholder for the not-yet-selected LR.
    for lr,label in [(5e-6,'5e-6'),(1e-5,'1e-5')]:
        for group in [4,8]:
            name=f'03-lr-{label}-group-{group}';c=fresh(name)
            c['stages'][0].update(learning_rate=lr,group_size=group,batch_size=8//group)
            add(name,c,scope='16 batches x 8 episodes; equal worst-case rollout token allowance; fixed factorial arm')
    name='04-investigation-sft';c=fresh(name);c['supervised']=supervised
    updates=math.ceil(len(records)/8);c['stages']=[{'kind':'sft','learning_rate':1e-4,'batch_size':8,'max_updates':updates,'max_batches':updates}]
    add(name,c,scope='one pass over all current verified investigations; no repeat-to-fill final batch')
    name='04-sft-then-grpo';c=fresh(name,'fork');c['execution']['checkpoint']=configs['04-investigation-sft']['output']+'/checkpoints/best.json'
    add(name,c,['04-investigation-sft'],scope='selected SFT weights with fresh optimizer; matched to direct GRPO; best can be initial weights')
    name='04-tool-only-sft';c=fresh(name)
    historical=json.loads(Path('configs/experiments/sft-tool-warmup-v1/train.json').read_text())
    c['supervised']=historical['supervised'];c['stages']=historical['stages']
    add(name,c,scope='optional replication of the completed negative 30-example tool-only pilot; not full-answer SFT')
    name='04-tool-sft-then-grpo';c=fresh(name,'fork')
    c['execution']['checkpoint']=configs['04-tool-only-sft']['output']+'/checkpoints/best.json'
    add(name,c,['04-tool-only-sft'],scope='separate tool-only SFT plus RL ablation; do not presume SFT benefit')
    # Collect additional answers without optimizing or exposing private references.
    demonstration_tasks=task_manifest(data,count=len(data['tasks']))
    atomic_json(root/'demonstration-tasks.json',demonstration_tasks)
    name='04-collect-demonstrations';c=fresh(name,'benchmark')
    c['execution']['timeout_seconds']=86400
    c['benchmark']={'attempts':4,'task_manifest':str(root/'demonstration-tasks.json'),
                    'manifest_hash':demonstration_tasks['manifest_hash']}
    add(name,c,scope='four fresh base-policy attempts per training task; no optimizer; freeze only current strictly passing clean demonstrations afterward')
    for name,reward in [('05-quality-only','tier-quality-v1'),('05-efficiency','tier-efficiency-v1')]:
        c=fresh(name);c['training_reward']['version']=reward
        add(name,c,scope='verifier-tier Q/E comparison from unchanged base; strict evaluation shared; experimental uncalibrated judge')
    variants={'control':(False,'none','raw'),'symbols':(True,'none','raw'),
              'lexical':(True,'lexical-v1','raw'),'hybrid-subword':(True,'hybrid-subword-v1','raw'),
              'history':(True,'hybrid-subword-v1','evidence-ledger-v1')}
    for label,(symbols,retrieval,history) in variants.items():
        for rep in [1,2]:
            name=f'06-{label}-r{rep}';c=fresh(name,'baseline','selection')
            c['harness']={'symbols':symbols,'retrieval':retrieval,'history':history,
                          'source_manifest':str(root/'source-index-manifest.json'),'source_manifest_sha256':source_hash}
            add(name,c,scope='frozen base inference; compare control/symbols, lexical/hybrid, hybrid/history as matched pairs')
    # Separate pairwise confirmation configs; chosen only after selection decisions.
    for label in variants:
        for rep in [1,2]:
            name=f'06-{label}-confirmation-r{rep}';c=copy.deepcopy(configs[f'06-{label}-r{rep}'])
            c['run_id']=name+'-'+campaign+'-seed42';c['output']='artifacts/experiments/'+c['run_id']
            c['spend']['ledger']='artifacts/project-budget/'+campaign+'/'+name+'.json'
            c['execution']['cohort']='confirmation'
            add(name,c,scope='run only locked finalists and matched control; never use for tuning')
    results=[]
    for name,c in configs.items():
        d=inputs(copy.deepcopy(c))
        if 'benchmark' in c:selected_tasks(c,d)
        if 'supervised' in c:validate_rendering(renderer,d['sft'])
        results.append({'id':name,'validation':'passed','counts':{k:len(d[k]) for k in ['tasks','development','sft']},
                        'operation_estimate':operation_estimate(c),'controller_reservation':controller_reservation(c)})
    total=sum(c['spend']['cap_usd'] for c in configs.values())
    atomic_json(root/'budget-plan.json',{'project_ceiling_usd':total+1000,
        'ledger_caps':{**json.loads(Path('configs/experiments/grpo-long-v6/budget.json').read_text())['ledger_caps'],
                       **{c['spend']['ledger']:c['spend']['cap_usd'] for c in configs.values()}},
        'reserves_usd':{'historical_and_other_work':70},
        'authorization':'Preparation requested by user; proposed explicit per-arm limits, no new spending or launch performed.',
        'note':'Ceiling includes original $1000 allocation intact. Optional controls/finalist arms are alternatives, not an instruction to run every arm. Recorded price snapshots require recheck at dispatch.'})
    atomic_json(root/'study-plan.json',{'version':campaign,'source':'configs/experiments/grpo-long-v6/run.json',
        'experiments':entries,'new_launches':0,
        'limitations':['SFT uses a small admitted policy-generated demonstration set, not an expert teacher dataset.',
                        'Hybrid retrieval uses a frozen deterministic character embedding; neural semantic embeddings remain a distinct extension.',
                        'Symbol tools cover Python ASTs, not semantic cross-language navigation.',
                        'Live semantic calibration and experimental outcomes cannot be established by offline configuration checks.',
                        'PPO, REINFORCE, learned rewards, code-execution probes, annotations and 9B scaling remain deferred roadmap extensions.']})
    atomic_json(root/'validation.json',results)
    print(json.dumps({'configs':len(configs),'sft':audit,'planned_new_caps':total,'new_launches':0},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default='configs/experiments/current-v8')
    prepare(p.parse_args().output)
