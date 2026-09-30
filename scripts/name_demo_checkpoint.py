"""Create one readable sampler copy without modifying the source trainer."""
import os,json,copy
from pathlib import Path
import certifi
os.environ.setdefault('SSL_CERT_FILE',certifi.where())
import tinker
from training_pipeline.storage import read,atomic_json,digest
root=Path('reports/bash-correctness-grpo-15-v1')
receipt=root/'named-demo-checkpoint.json'
if receipt.exists():
 print(json.dumps(read(receipt),indent=2));raise SystemExit
intent=root/'named-demo-checkpoint.pending.json'
if intent.exists():raise RuntimeError('Prior naming attempt exists; inspect it before retrying')
m=read(root/'best-eval-checkpoint.json')['best_manifest']
name='bash-correctness-grpo-15-v1-qwen35-4b-b8-g8-step003-eval07113'
metadata={'source_run':m['config']['run_id'],'source_checkpoint':m['id'],'optimizer_step':'3','eval_mean_reward':'0.7113095238095237','eval_resolved':'28/32','purpose':'demo-copy-no-training','wandb_run':'https://wandb.ai/nmdatar-harvard-university/repository-qa-training/runs/bash-correctness-grpo-15-v1'}
service=tinker.ServiceClient(user_metadata=metadata)
atomic_json(intent,{'name':name,'source':m['artifacts']['training'],'status':'loading'})
try:
 print('Loading existing step-3 weights into a separate temporary client...',flush=True)
 client=service.create_training_client_from_state(m['artifacts']['training'],user_metadata=metadata)
 atomic_json(intent,{'name':name,'source':m['artifacts']['training'],'status':'saving'})
 saved=client.save_weights_for_sampler(name,ttl_seconds=172800,user_metadata=metadata).result(timeout=600)
 result={'name':name,'sampler':saved.path,'source_sampler':m['artifacts']['sampler'],'metadata':metadata,'ttl_seconds':172800}
 atomic_json(receipt,result)
 # Register the inference copy with the existing demo model catalog.
 demo=copy.deepcopy(m)
 demo.update(id=name,parent=m['id'])
 demo['artifacts']={'sampler':saved.path,'ttl_seconds':172800}
 demo['manifest_hash']=digest({k:v for k,v in demo.items() if k!='manifest_hash'})
 atomic_json('artifacts/demo-named-checkpoints/checkpoints/'+name+'.json',demo)
 info=service.create_rest_client().get_weights_info_by_tinker_path(saved.path).result(timeout=60)
 print(json.dumps(result,indent=2),flush=True)
 intent.unlink()
finally:
 service.close('success' if receipt.exists() else 'errored',detail='Named inference copy only; no optimizer updates').result(timeout=60)
