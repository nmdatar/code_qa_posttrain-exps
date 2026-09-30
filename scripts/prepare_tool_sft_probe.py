"""Freeze the preregistered paired sampling probe without reading SFT answers."""
import copy,json,hashlib
from pathlib import Path
from transformers import AutoTokenizer
from training_pipeline.rendering import ChatRenderer
from training_pipeline.config import inputs
from training_pipeline.remote import prepare
from training_pipeline.storage import atomic_json,digest
root=Path('artifacts/sft-tool-prefix-v1-results/artifacts/experiments/sft-tool-prefix-v1-seed42')
checkpoints=[json.loads(p.read_text()) for p in (root/'checkpoints').glob('ckpt-*.json')]
by_step={c['state']['optimizer_step']:c for c in checkpoints};assert 0 in by_step and 4 in by_step
# No post-SFT answers are read here. Step metadata only selects the control report.
initial=None
for p in (root/'evaluations').glob('*.json'):
 r=json.loads(p.read_text())
 if r['optimizer_step']==0:initial=r
assert initial is not None
config=json.loads(Path('configs/experiments/grpo-v6-parallel/baseline-v1.json').read_text())
tasks={r['id']:r for r in inputs(config)['development']}
renderer=ChatRenderer(AutoTokenizer.from_pretrained(config['model']['base_model'],local_files_only=True),8192)
contexts=[]
for result in initial['results']:
 trace=json.loads((root/'trajectories'/(result['episode_id']+'.json')).read_text());messages=trace['events'][0]['messages']
 indices=[i for i,m in enumerate(messages) if m['role']=='assistant']
 for k in sorted(set([0,len(indices)-1])):
  i=indices[k];prefix=messages[:i];assert renderer.prompt(prefix)==trace['generations'][k]['prompt'];assert len(renderer.prompt(prefix))+512<=8192
  observed={};n=0
  for event in trace['events']:
   if event['kind']=='generation':
    if n>=k:break
    try:action=json.loads(event['text'])
    except ValueError:action={}
    if not isinstance(action,dict):action={}
    n+=1
   if event['kind']=='observation' and action.get('tool',action.get('action'))=='read_file' and event['value'].get('exit_code')==0:
    path=action['arguments']['path']
    # Hash must have been observed in the successful saved read result.
    observed[path]=tasks[result['task_id']]['image_result']['snapshot_files'][path]
  contexts.append({'id':result['task_id']+'-'+str(k),'task_id':result['task_id'],'episode_id':trace['episode_id'],'messages':prefix,'observed_files':observed})
fixture={'version':'tool-sft-sampled-interface-v1','contexts':contexts,'samples_per_context':2,'temperature':1,'policies':{name:by_step[step]['artifacts'] for name,step in [('before',0),('after',4)]},'spec_sha256':hashlib.sha256(Path('reports/sft-tool-warmup-v1/probe-spec.json').read_bytes()).hexdigest()}
assert len(contexts)<=64
config['run_id']='sft-tool-prefix-v1-probe';config['output']='artifacts/experiments/'+config['run_id'];config['execution']['timeout_seconds']=900
config['tracking']['mode']='disabled';config['tracking']['notes']='Sampling-only paired interface probe. No optimization, tool execution or judge calls.'
p=Path('configs/experiments/sft-tool-warmup-v1/probe.json');atomic_json(p,config)
bundle=Path('artifacts/sft-tool-prefix-v1-probe-bundle');plan=prepare([p],bundle)
atomic_json(bundle/'probe-fixture.json',fixture)
m=json.loads((bundle/'bundle.json').read_text());m.pop('bundle_id');m['tool_sft_probe']={'fixture':'probe-fixture.json','reservation_cap_usd':2}
m['files']['probe-fixture.json']=hashlib.sha256((bundle/'probe-fixture.json').read_bytes()).hexdigest();m['bundle_id']=digest(m);atomic_json(bundle/'bundle.json',m)
plan.update(bundle_id=m['bundle_id'],contexts=len(contexts),generation_calls=len(contexts)*4,model_reservation_cap_usd=2)
atomic_json('reports/sft-tool-warmup-v1/probe-launch.json',plan);print(json.dumps(plan,indent=2))
