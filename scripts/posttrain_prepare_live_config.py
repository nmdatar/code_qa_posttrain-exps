"""Freeze the small paid acceptance configuration from reviewed release bindings."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from posttrain.storage import read,atomic
from posttrain.releases import publish
rows=[json.loads(l) for l in Path('reports/posttrain/fresh-task-candidates.jsonl').read_text().splitlines()]
selected=[r for r in rows if r['task']['repository']['family_id'] in {'go-task/task','open-telemetry/opentelemetry-python'}]
path=Path('reports/posttrain/smoke-training-candidates.jsonl')
path.write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in selected))
train='data/releases/repo-qa-training-smoke-v1';dev='data/releases/repo-qa-development-posttrain-v1'
if not Path(train).exists():print(publish(path,'reports/posttrain/fresh-task-independent-review.json',['data/prepared/posttrain-fresh-v1'],train,20,'train'))
bindings={**read(Path(train)/'private/runtime-bindings.json'),**read(Path(dev)/'private/runtime-bindings.json')}
training=[json.loads(l) for l in (Path(train)/'public/tasks.jsonl').read_text().splitlines()]
evaluation=[json.loads(l) for l in (Path(dev)/'public/tasks.jsonl').read_text().splitlines()]
verifier=read('examples/experiment-config.json')
verifier.update(training_judge_family='nemotron3',evaluation_judge_family='qwen3',judge_version='tinker-nemotron3-qwen3-2026-09-29',
 judge_command=['tinker','nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16','nemotron3_disable_thinking','Qwen/Qwen3-8B','qwen3_disable_thinking'],
 environment_id='modal-immutable-source-tools-v1',max_evidence_bytes=200000,judge_timeout_seconds=120)
c={'run_id':'live-repository-grpo-001','backend':'tinker','model':'openai/gpt-oss-20b','renderer_name':'gpt_oss_no_sysprompt',
 'pricing_snapshot':read('reports/posttrain/services/tinker-candidates.json')[0],
 'stages':[{'algorithm':'grpo','max_updates':1,'learning_rate':1e-5}],
 'training_release':train,'evaluation_release':dev,
 'train_tasks':[{'id':t['id']} for t in training], 'eval_tasks':[{'id':t['id']} for t in evaluation[:2]],
 'environment_bundles':{k:v['bundle'] for k,v in bindings.items()},'repository_roots':{k:v['source_root'] for k,v in bindings.items()},
 'paid_call_bounds':{'tinker':{'connect':.01,'save':.01,'sample':.005,'update':.05},'judge':.01,'modal':.003},
 'verifier_config':verifier,'max_attempted_batches':3,'max_output_tokens':2048,'max_tool_calls':8,'max_turns':12,
 'max_episode_seconds':300,'max_context_tokens':16384,'temperature':0.8,'tracking_mode':'online','diagnostic':True}
atomic('examples/posttrain/repository-diagnostic.json',c)
print('Frozen live acceptance:1 update,4rollouts/group,2-task diagnostic evaluation each step; full20-task evaluation config follows separately.')
