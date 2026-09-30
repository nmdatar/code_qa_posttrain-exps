import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from posttrain.runner import run,Coordinator,freeze_data
from posttrain.config import load_config
from posttrain.storage import BudgetLedger
from posttrain.backends import FakeBackend
from posttrain.environments import FakeEnvironment
from posttrain.rollout import run_episode
from test_posttrain_integration import config,fixture_task

class RuntimeAuditTests(unittest.TestCase):
    def test_checkpoint_identity_and_shared_budget_cannot_drift(self):
        with tempfile.TemporaryDirectory() as root:
            c=config(root);c['stages'][0]['max_updates']=1
            first=run(c)
            for key,value in [('model','other'),('lora_rank',32),('renderer_name','other'),('budget_ledger',str(Path(root)/'fresh-budget.json'))]:
                other=dict(c,run_id='fork-'+key);other[key]=value
                with self.subTest(key=key),patch('posttrain.runner.make_backend') as make:
                    make.return_value=FakeBackend()
                    with self.assertRaisesRegex(ValueError,'identity mismatch|shared spending ledger'):
                        run(other,checkpoint=first['checkpoint'])
                    self.assertEqual(make.return_value.step,0)
    def test_wrong_snapshot_environment_never_reaches_solver(self):
        with tempfile.TemporaryDirectory() as root:
            c=load_config(config(root));c['backend']='tinker';t=c['train_tasks'][0]
            c['environment_bundles']={t['id']:'fake-bundle'};c['paid_call_bounds']={'modal':.01}
            wrong=dict(t,environment_id='other')
            co=Coordinator(c,root,FakeBackend(),BudgetLedger(Path(root)/'ledger.json'))
            with patch('posttrain.runner.ModalEnvironment',return_value=SimpleNamespace(record={'task':wrong})),patch('posttrain.runner.run_episode') as episode:
                with self.assertRaisesRegex(ValueError,'admitted task binding'):co.episode(t,'episode',0)
                episode.assert_not_called()
    def test_gptoss_json_channel_is_conditioning_not_sampled_target(self):
        from posttrain.backends import TinkerBackend,training_rows
        backend=TinkerBackend('openai/gpt-oss-20b',None,'gpt_oss_no_sysprompt')
        seen={}
        def build(messages,**kwargs):
            seen.update(kwargs);return SimpleNamespace(to_ints=lambda:[10,11,12])
        backend.renderer=SimpleNamespace(build_generation_prompt=build)
        prompt=backend.render([{'role':'user','content':'Question'}])
        self.assertEqual(seen['prefill'],'<|channel|>final<|message|>')
        row=training_rows([{'task_id':'t','policy_id':'p','split':'train','actions':[{'prompt_tokens':prompt,'token_ids':[13,14],'logprobs':[-1.,-1.],'policy_id':'p'}]}],[1.])[0]
        self.assertEqual(row['mask'],[0,0,1,1])

    def test_task_latency_limit_applies(self):
        task=fixture_task('t','train');task['budgets']['latency_seconds']=1
        with patch('posttrain.rollout.time.monotonic',side_effect=[0,2,3]):
            result=run_episode(FakeBackend(),FakeEnvironment(),task,{'max_episode_seconds':300})
        self.assertEqual(result['termination'],'time_budget')
        self.assertEqual(result['actions'],[])
    def test_live_sft_requires_release_bound_examples(self):
        # Empty manifests isolate admission enforcement before any service call.
        with tempfile.TemporaryDirectory() as root:
            c=load_config(config(root));c.update(backend='tinker',training_release=root,evaluation_release=root,train_tasks=[],eval_tasks=[])
            c['stages']=[{'algorithm':'sft','max_updates':1,'learning_rate':.01}]
            manifest={'artifacts':{'public/tasks.jsonl':'x','private/grading.jsonl':'x','private/rubric_reviews.jsonl':'x'}}
            with patch('posttrain.data.validate_artifacts'),patch('posttrain.runner.read',return_value=manifest),patch('posttrain.data.read_jsonl',return_value=[]):
                with self.assertRaisesRegex(ValueError,'manifest-bound examples'):freeze_data(c)
