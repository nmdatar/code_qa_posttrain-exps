"""Freeze a source-verified, lineage-deduplicated tool-only SFT pilot."""
import json, math, shutil, collections
from pathlib import Path
from transformers import AutoTokenizer
from training_pipeline.collection import load_collection
from training_pipeline.rendering import ChatRenderer
from training_pipeline.sft_collection import candidate, validate_rendering, VERSION
from training_pipeline.storage import digest, atomic_json
from training_pipeline.admission import sha
from training_pipeline.config import inputs
from training_pipeline.launch import estimate

base=json.loads(Path('configs/experiments/grpo-v6-parallel/baseline-v1.json').read_text())
data=load_collection(base['environment']); tasks={r['id']:r for r in data['tasks']}
renderer=ChatRenderer(AutoTokenizer.from_pretrained(base['model']['base_model'],local_files_only=True),base['limits']['context_tokens'])
choices={}; rejected=collections.Counter(); scanned=0
for p in sorted(Path('artifacts').glob('**/trajectories/*.json')):
    t=json.loads(p.read_text())
    if t.get('split')!='train' or t.get('task_id') not in tasks: continue
    scanned+=1
    try:
        e=candidate(t,tasks[t['task_id']]); validate_rendering(renderer,[e])
        rank=(len(e['assistant_turns']),-sum(len(x['generation']['prompt']) for x in e['proofs']))
        if e['lineage_id'] not in choices or rank>choices[e['lineage_id']][0]: choices[e['lineage_id']]=(rank,e,p)
    except (ValueError,KeyError,TypeError) as exc: rejected[str(exc)]+=1
root=Path('data/sft/tool-prefix-v1')
if root.exists(): raise ValueError('Never overwrite frozen SFT release')
(root/'sources').mkdir(parents=True)
records=[]; names=collections.Counter(); total_tokens=0
for _,e,p in sorted(choices.values(),key=lambda x:x[1]['id']):
    dest=root/'sources'/(e['id']+'.json');shutil.copyfile(p,dest)
    records.append({'source':str(dest.relative_to(root)),'sha256':sha(dest.read_bytes()),'example_hash':digest(e),'task_id':e['task_id'],'original_path':str(p),'selected_turns':e['assistant_turns']})
    for i in e['assistant_turns']: names[json.loads(e['messages'][i]['content'])['tool']]+=1
    total_tokens+=sum(len(r.input_tokens) for r in renderer.supervised(e))
manifest={'version':VERSION,'data_identity':data['identity'],'human_reviewed':False,'scope':'successful tool actions only; answers not supervised','records':records,'seed':42,'source_policy':'archived Qwen/Qwen3.5-4B base and RL adapters; no new teacher calls','deduplication':'one trajectory prefix per training task lineage'}
atomic_json(root/'manifest.json',manifest)
base['supervised']={'manifest':str((root/'manifest.json').resolve()),'sha256':sha((root/'manifest.json').read_bytes())}
base['run_id']='sft-tool-prefix-v1-seed42';base['output']='artifacts/experiments/'+base['run_id']
base['execution'].update(operation='run',timeout_seconds=1800);base['execution'].pop('cohort',None)
updates=math.ceil(len(records)/8)
base['stages']=[{'kind':'sft','batch_size':8,'learning_rate':1e-4,'max_updates':updates,'max_batches':updates}]
base['evaluation']['every']=5;base.pop('stopping',None)
base['spend']['ledger']='artifacts/project-budget/02-direct-grpo.json';base['spend']['cap_usd']=550
base['best_checkpoint']={'retention_seconds':1209600}
base['tracking'].update(notes='Preregistered tool-only SFT: one verified prefix per training lineage, one pass. No final answers supervised. v6 matched evaluation.',tags=['sft','tool-prefix-v1','seed42'])
path=Path('configs/experiments/sft-tool-warmup-v1/train.json');atomic_json(path,base)
loaded=inputs(base); assert len(loaded['sft'])==len(records)
summary={'scanned_training_trajectories':scanned,'admitted_examples':len(records),'families':sorted(set(e['family_id'] for _,e,_ in choices.values())),'selected_actions':dict(names),'rejections':dict(rejected),'updates':updates,'training_input_tokens_one_pass':total_tokens,'manifest_sha256':base['supervised']['sha256'],'source_collection_identity':data['identity'],'confirmation_touched':False,'plan':estimate(base,base['spend']['prices'])}
atomic_json('reports/sft-tool-warmup-v1/admission.json',summary);print(json.dumps(summary,indent=2))
