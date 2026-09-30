from tests.test_claim_grading import request_fixture, judgments
import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as N
from unittest.mock import Mock
from training_pipeline.judge import TinkerJudge
from training_pipeline.collection import CollectionFactory, VERSION
from training_pipeline.config import validate_config
from training_pipeline.storage import read, semantic_hash
from training_pipeline.contracts import ConfigurationError, Generation
from training_pipeline.budget import SpendLedger, BudgetLimit
from training_pipeline.launch import estimate


class JudgeTests(unittest.TestCase):
    def setUp(self):
        self.config = read('examples/training-repository-qwen-nemotron.json')
        self.j = self.config['judge']
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.tokenizer = N(chat_template='template', decode=lambda *a, **k:'raw')
        self.renderer = Mock(tokenizer=self.tokenizer)
        self.renderer.build_generation_prompt.return_value.to_ints.return_value = [1, 2, 3]
        self.renderer.parse_response.return_value = ({'content':'{"status":"resolved","score":1,"reason":"yes"}'}, True)
        self.renderer.get_stop_sequences.return_value = [4]
        self.sampler = Mock()
        self.sampler.get_tokenizer.return_value = self.tokenizer
        self.sampler.sample.return_value.result.return_value = N(sequences=[N(tokens=[5],logprobs=[-.1],stop_reason='stop')])
        self.service = Mock()
        self.service.get_server_capabilities.return_value = N(supported_models=[N(
            model_name=self.j['base_model'], sampleable=True, trainable=False,max_context_length=32768)])
        self.service.create_sampling_client.return_value = self.sampler
        self.sdk = N(SamplingParams=lambda **kw:kw)

    def judge(self, ledger=None):
        judge = TinkerJudge(self.j,ledger,self.service,self.sdk,lambda *a,**kw:self.renderer)
        self.addCleanup(judge.close)
        return judge

    def test_sampling_only_and_exact_renderer_and_pricing(self):
        ledger = SpendLedger(self.root/'spend.json',5,self.config['spend']['prices'],3600)
        judge = self.judge(ledger)
        self.service.create_sampling_client.assert_called_once()
        self.assertEqual(self.service.create_sampling_client.call_args.kwargs['base_model'],self.j['base_model'])
        result = judge.sample([{'role':'user','content':'grade'}],1024,0)
        self.service.create_lora_training_client.assert_not_called()
        self.assertEqual(result.prompt,[1,2,3])
        self.assertEqual(result.policy_id,judge.policy_id)
        reservation=read(ledger.path)['reservations'][0]
        self.assertAlmostEqual(read(ledger.path)['reserved_usd'],(3*.195+1024*.495)/1e6)
        self.assertEqual(reservation['model'],self.j['base_model'])

    def test_budget_and_context_block_before_generation(self):
        ledger = SpendLedger(self.root/'spend.json',.00001,self.config['spend']['prices'],3600)
        judge=self.judge(ledger)
        with self.assertRaises(BudgetLimit):judge.sample([],1024,0)
        self.sampler.sample.assert_not_called()
        self.renderer.build_generation_prompt.return_value.to_ints.return_value=[1]*16384
        with self.assertRaisesRegex(ValueError,'overflow'):judge.sample([],1024,0)
        self.sampler.sample.assert_not_called()

    def test_capability_failures_close_service(self):
        self.service.get_server_capabilities.return_value.supported_models=[]
        with self.assertRaises(ConfigurationError):self.judge()
        self.service.close.assert_called_once_with('errored')

    def test_config_identity_estimates_and_legacy(self):
        validate_config(self.config)
        old=copy.deepcopy(self.config);old.pop('judge')
        validate_config(old)
        self.assertNotEqual(semantic_hash(old),semantic_hash(self.config))
        newfactory=CollectionFactory(self.config,self.root,None)
        oldfactory=CollectionFactory(old,self.root,None)
        self.assertTrue(oldfactory.reward_version.startswith('source-claims-v2'))
        self.assertNotEqual(newfactory.reward_version,VERSION)
        prices=self.config['spend']['prices'];plan=estimate(self.config,prices)
        self.assertAlmostEqual(plan['components_usd']['reference_grading'],plan['episodes_including_retries']*(16384*.195+1024*.495)/1e6)
        self.config['judge']['prices']['model']='wrong'
        with self.assertRaises(ConfigurationError):validate_config(self.config)

    def test_truncated_grade_keeps_raw_and_frozen_identity(self):
        judge=Mock(identity={'base_model':self.j['base_model']})
        judge.sample.return_value=Generation([1],[2],[-.1],'{"status":"resolved","score":1,"reason":"yes"}','length','frozen')
        factory=CollectionFactory(self.config,self.root,None,judge=judge)
        with self.assertRaisesRegex(ValueError,'Truncated'):factory.grade(request_fixture())
        raw=read(self.root/'private/test.judge-raw.json')
        self.assertEqual(raw['generation']['stop_reason'],'length')
        self.assertEqual(raw['judge_identity']['base_model'],self.j['base_model'])
        self.assertFalse((self.root/'private/test.judge.json').exists())
        judge.sample.assert_called_once()
        self.assertEqual(judge.sample.call_args.args[1:],(1024,0))


if __name__ == '__main__':unittest.main()
