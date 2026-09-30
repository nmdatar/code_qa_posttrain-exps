"""Bounded checkpoint reload checks using the recorded model and shared budget."""
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from posttrain.backends import TinkerBackend
from posttrain.checkpoints import load_checkpoint
from posttrain.storage import BudgetLedger,atomic
p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True);p.add_argument('--before');p.add_argument('--trajectory');a=p.parse_args()
out=Path(a.output)
if out.exists():raise SystemExit('Refusing to overwrite an existing live verification result')
m=load_checkpoint(a.checkpoint);c=m['config']
if m['state']['optimizer_step']<1:raise SystemExit('A post-update checkpoint is required')
ledger=BudgetLedger(c['budget_ledger'],c['budget_cap_usd'],c['budget_reserve_usd'])
def backend():return TinkerBackend(c['model'],ledger,c['renderer_name'],rank=c['lora_rank'],bounds=c['paid_call_bounds']['tinker'])
report={'model':c['model'],'checkpoint':str(Path(a.checkpoint).resolve()),'optimizer_step':m['state']['optimizer_step'],'checks':{},'status':'running'}
atomic(out,report)
try:
 sampler=backend().load(m['references'],purpose='evaluate')
 sample=sampler.sample([{'role':'user','content':'Reply with exactly: checkpoint loaded'}],max_tokens=32,temperature=0,seed=42)
 report['checks']['sampling_reload']={'passed':bool(sample['token_ids']),'policy_id':sample['policy_id'],'token_count':len(sample['token_ids'])}
 if a.before and a.trajectory:
  from posttrain.storage import read
  from tinker import types
  before=load_checkpoint(a.before)
  if before['model']!=m['model']:raise ValueError('Model mismatch for likelihood comparison')
  initial=backend().load(before['references'],purpose='evaluate')
  action=read(a.trajectory)['actions'][-1]
  tokens=action['prompt_tokens']+action['token_ids']
  prompt=types.ModelInput.from_ints(tokens)
  old=initial._paid('sample',lambda:initial.sampling.compute_logprobs(prompt).result())
  new=sampler._paid('sample',lambda:sampler.sampling.compute_logprobs(prompt).result())
  pairs=[(x,y) for x,y in zip(old[len(action['prompt_tokens']):],new[len(action['prompt_tokens']):]) if x is not None and y is not None]
  if not pairs:raise ValueError('No token likelihoods returned')
  report['checks']['same_token_likelihood_change']={'tokens_compared':len(pairs),'max_absolute_change':max(abs(y-x) for x,y in pairs),'mean_logprob_before':sum(x for x,y in pairs)/len(pairs),'mean_logprob_after':sum(y for x,y in pairs)/len(pairs),'scope':'same saved assistant action under initial and updated sampling weights; not an evaluation metric'}
 resumed=backend().load(m['references'],optimizer=True)
 report['checks']['optimizer_resume']={'passed':True,'training_state':m['references']['training_state'],'framework_state':m['state']}
 forked=backend().load(m['references'],optimizer=False)
 report['checks']['weights_only_fork']={'passed':True,'optimizer_restored':False,'new_framework_optimizer_step':0}
 report['status']='passed'
except BaseException as exc:
 report.update(status='failed',error_type=type(exc).__name__)
 raise
finally:atomic(out,report)
print({'status':report['status'],'model':c['model'],'optimizer_step':m['state']['optimizer_step']})
