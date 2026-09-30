import copy,json,unittest
from unittest.mock import Mock
from training_pipeline.strategies import grpo_batch,reinforce_batch
from tests.test_training_pipeline import trajectory
from tests.test_parallel_training_tools import ParallelCollectionTests
from training_pipeline.contracts import PolicyFormatError
from training_pipeline.config import validate_config
from training_pipeline.storage import read
class StabilityMathTests(unittest.TestCase):
 def test_mask_keeps_baseline_and_does_not_amplify_survivor(self):
  a,b=trajectory(0,0),trajectory(1,1); b.generations[0].stop_reason='length'
  base,_=grpo_batch([[a,b]]); rows,stats=grpo_batch([[a,b]],{'mask_overlong':True})
  self.assertEqual(rows,base[:1]); self.assertEqual(stats['masked_overlong_trajectories'],1)
  self.assertEqual(stats['rewards'],[0,1]);self.assertEqual(stats['contributing_trajectories'],1)
 def test_reinforce_mask_retains_baseline_state(self):
  a,b=trajectory(0,0),trajectory(1,1);b.events=[{'kind':'termination_cause','reason':'generations'}]
  rows,stats,state=reinforce_batch([[a,b]],{'reward_sum':1,'count':2},{'mask_overlong':True})
  self.assertEqual(len(rows),1); self.assertEqual(state,{'reward_sum':2,'count':4})
 def test_width_scaling_and_serial_noop(self):
  a,b=trajectory(0,0),trajectory(1,1);a.usage={'tool_calls':4,'tool_turns':1};b.usage={'tool_calls':3,'tool_turns':3}
  base,_=grpo_batch([[a,b]]);rows,stats=grpo_batch([[a,b]],{'scale_tool_width':True})
  self.assertEqual(rows[0].weights,[w/4 for w in base[0].weights]);self.assertEqual(rows[1],base[1])
  self.assertEqual(stats['width_scaled_trajectories'],1)
 def test_all_masked_is_a_skipped_update_not_divide_by_zero(self):
  a,b=trajectory(0,0),trajectory(1,1)
  for t in (a,b): t.generations[0].stop_reason='length'
  rows,stats=grpo_batch([[a,b]],{'mask_overlong':True})
  self.assertEqual(rows,[]);self.assertEqual(stats['masked_overlong_trajectories'],2)

 def test_latency_is_not_overlong(self):
  a,b=trajectory(0,0),trajectory(1,1);b.termination='budget_exhausted';b.events=[{'kind':'termination_cause','reason':'latency'}]
  rows,stats=grpo_batch([[a,b]],{'mask_overlong':True});self.assertEqual(len(rows),2)
 def test_flags_validate(self):
  c=read('examples/training-repository-qwen-v3.json');c['stages'][0]['stability']={'mask_overlong':True};validate_config(c)
  c['stages'][0]['stability']['mask_overlong']=1
  with self.assertRaises(ValueError):validate_config(c)
class StabilityFormatTests(ParallelCollectionTests):
 def test_schema_failures_are_distinct_from_permission(self):
  self.ep.invalid_action_policy='zero-v1'
  for action in [[],{'tool':'read_file','arguments':{}},{'answer':'x','citations':'bad'},{'tool_calls':[{'answer':'x'}]}]:
   with self.assertRaises(PolicyFormatError):self.ep.parse_action(json.dumps(action))
  action=self.ep.parse_action(json.dumps({'tool':'read_file','arguments':{'path':'unknown.py'}}))
  with self.assertRaises(ValueError) as err:self.ep.step(action)
  self.assertNotIsInstance(err.exception,PolicyFormatError)
 def test_zero_format_never_calls_judge(self):
  from agent_harness.training_runner import run_episode
  from training_pipeline.contracts import Generation,VerificationResult
  from pathlib import Path
  from types import SimpleNamespace
  self.ep.invalid_action_policy='zero-v1';self.factory.identity='env';self.factory.create=Mock(return_value=self.ep)
  self.ep.verify_invalid_format=Mock(return_value=VerificationResult('resolved',0,'v'))
  self.ep.verify=Mock();self.ep.close=Mock()
  backend=SimpleNamespace(policy_id='p',identity={},sample=Mock(return_value=Generation([1],[2],[-.1],'not json','stop','p')))
  limits={'max_tool_calls':5,'max_generations':6,'max_output_tokens':3000,'max_tokens_per_call':512,'latency_seconds':600,'max_tool_output_bytes':3500}
  t=run_episode(backend,self.factory,{'id':'task','split':'train'},limits,run_id='r',stage=0,group_id='g',episode_id='e',experiment_hash='h',temperature=1,tracker=Mock(root=Path(self.tmp.name)))
  self.assertEqual(t.termination,'invalid_format');self.assertEqual(t.verification.reward,0);self.assertEqual(len(t.generations),1)
  self.ep.verify.assert_not_called();self.sandbox.execute_many.assert_not_called()
if __name__=='__main__':unittest.main()
