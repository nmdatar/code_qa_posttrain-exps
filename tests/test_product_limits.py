"""Generation budgets must expand without breaking smaller checkpoint contexts."""
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock
from product_api.model import ProductModel


class ProductLimitTests(unittest.TestCase):
    def test_generation_expands_and_still_honors_context_and_remaining_budget(self):
        model = ProductModel({'base_model':'test/model'})
        model.sdk = SimpleNamespace(ModelInput=SimpleNamespace(from_ints=lambda x:x), SamplingParams=lambda **kw:kw)
        tokenizer = SimpleNamespace(eos_token_id=0, decode=lambda *args, **kw:'{"answer":"Done","citations":[]}')
        model.renderer = SimpleNamespace(tokenizer=tokenizer, context_limit=32768, prompt=lambda _: [1,2,3])
        model.sampling = MagicMock()
        model.sampling.sample.return_value.result.return_value = SimpleNamespace(sequences=[SimpleNamespace(tokens=[4],stop_reason='stop')])
        messages = [{'role':'system','content':'System'}, {'role':'user','content':'Question'}]
        model.generate(messages, [], 12_000, timeout_seconds=20)
        self.assertEqual(model.sampling.sample.call_args.kwargs['sampling_params']['max_tokens'], 8192)
        self.assertEqual(model.sampling.sample.return_value.result.call_args.kwargs['timeout'],20)
        model.renderer.context_limit = 8192
        model.generate(messages, [], 12_000)
        self.assertEqual(model.sampling.sample.call_args.kwargs['sampling_params']['max_tokens'],2048)
        model.generate(messages, [], 100)
        self.assertEqual(model.sampling.sample.call_args.kwargs['sampling_params']['max_tokens'],100)
