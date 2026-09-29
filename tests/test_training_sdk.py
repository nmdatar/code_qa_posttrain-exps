"""Optional real SDK schema checks with fixture clients: no network or API keys."""
import importlib.util
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from training_eval.backends import TinkerBackend
from training_eval.contracts import TrainingRow, UpdateBatch


@unittest.skipUnless(importlib.util.find_spec('tinker'), 'optional Tinker SDK not installed')
class RealSDKSchemaTests(unittest.TestCase):
    def test_real_sdk_training_and_sampling_types(self):
        import tinker
        def future(value):
            return SimpleNamespace(result=lambda: value)
        client = Mock()
        client.get_tokenizer.return_value = SimpleNamespace(decode=lambda ids: str(ids))
        client.forward_backward.return_value = future(SimpleNamespace(metrics={'loss:sum': 0.5}))
        client.optim_step.return_value = future(None)
        service = Mock()
        service.create_lora_training_client.return_value = client
        backend = TinkerBackend(service, tinker, 'mock-model', renderer='token-native-v1', context_limit=64)
        for loss in ['cross_entropy', 'importance_sampling']:
            row = TrainingRow((1, 2), (2, 3), (0., .5), (0., -.2), (0., .5))
            backend.update(UpdateBatch(loss, (row,)), .001)
            datum = client.forward_backward.call_args.args[0][0]
            self.assertIsInstance(datum, tinker.Datum)
            self.assertEqual(datum.model_input.to_ints(), [1, 2])
            self.assertEqual(datum.loss_fn_inputs['target_tokens'].data, [2, 3])
        sampler = Mock()
        sampler.sample.return_value = future(SimpleNamespace(sequences=[
            SimpleNamespace(tokens=[3], logprobs=[-.2], stop_reason='stop')]))
        client.save_weights_and_get_sampling_client.return_value = sampler
        backend.snapshot('test').sample([1, 2], max_tokens=1, temperature=1, seed=3)
        self.assertIsInstance(sampler.sample.call_args.kwargs['sampling_params'], tinker.SamplingParams)


if __name__ == '__main__':
    unittest.main()
