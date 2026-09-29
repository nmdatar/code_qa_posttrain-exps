import tempfile
import unittest
from dataclasses import asdict
from types import SimpleNamespace as NS
from unittest.mock import Mock

from training_eval.backends import FakeBackend, FakeBackendFactory, TinkerBackend, TinkerBackendFactory
from training_eval.contracts import (CapabilityError, InfrastructureError, ModelRef, Sample,
                                    ScoredTrajectory, Trajectory, TrainingRow,
                                    UncertainUpdateError, UpdateBatch, Verification)
from training_eval.model import ModelFactory
from training_eval.strategies import GRPOStrategy


def future(value):
    return NS(result=Mock(return_value=value))


def sdk_stub():
    return NS(ModelInput=NS(from_ints=lambda x: tuple(x)), TensorData=lambda **kw: NS(**kw),
              Datum=lambda **kw: NS(**kw), AdamParams=lambda **kw: NS(**kw),
              SamplingParams=lambda **kw: NS(**kw))


class BackendTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)

    def test_fake_training_snapshots_and_resume(self):
        backend = FakeBackend('mock', self.directory.name)
        before = backend.snapshot('before')
        batch = UpdateBatch('cross_entropy', (TrainingRow((1,), (49,), (1.0,)),))
        backend.update(batch, 0.5)
        self.assertGreater(backend.weight, before.weight)
        artifacts = backend.save('step')
        backend.verify_artifacts(artifacts)
        fork = FakeBackend('mock', self.directory.name)
        fork.load_training(artifacts.training, optimizer=False)
        self.assertEqual(fork.weight, backend.weight)
        self.assertEqual(fork.optimizer_steps, 0)
        fork.load_training(artifacts.training, optimizer=True)
        self.assertEqual(fork.optimizer_steps, backend.optimizer_steps)
        self.assertEqual(fork.momentum, backend.momentum)
        policy = fork.load_sampling(artifacts.sampling)
        samples = [policy.sample([1], max_tokens=1, temperature=1, seed=i) for i in range(20)]
        self.assertEqual({s.text for s in samples}, {'0', '1'})
        self.assertEqual(samples[0], policy.sample([1], max_tokens=1, temperature=1, seed=0))

    def tinker(self):
        client = Mock()
        client.get_tokenizer.return_value = NS(decode=lambda ids: str(ids))
        client.forward_backward.return_value = future(NS(metrics={'loss:sum': 1.25}))
        client.optim_step.return_value = future(None)
        service = Mock()
        service.create_lora_training_client.return_value = client
        return TinkerBackend(service, sdk_stub(), 'base', renderer='tokens-v1', context_limit=10), client, service

    def test_tinker_exact_loss_inputs_and_reduction(self):
        backend, client, _ = self.tinker()
        row = TrainingRow((1, 2), (2, 3), (0.0, 0.25), (0.0, -2.0), (0.0, -0.75))
        self.assertEqual(backend.update(UpdateBatch('importance_sampling', (row,)), .001), {'loss': 1.25})
        data, loss = client.forward_backward.call_args.args
        self.assertEqual(loss, 'importance_sampling')
        self.assertEqual(data[0].model_input, (1, 2))
        self.assertEqual(data[0].loss_fn_inputs['advantages'].data, [0.0, -0.75])
        self.assertEqual(data[0].loss_fn_inputs['logprobs'].data, [0.0, -2.0])
        self.assertEqual(data[0].loss_fn_inputs['target_tokens'].dtype, 'int64')
        self.assertEqual(data[0].loss_fn_inputs['target_tokens'].shape, [2])
        client.forward_backward.return_value.result.assert_called_once()
        client.optim_step.return_value.result.assert_called_once()
        backend.update(UpdateBatch('cross_entropy', (row,)), .001)
        self.assertEqual(client.forward_backward.call_args.args[0][0].loss_fn_inputs['weights'].data, [0., .25])

    def test_uncertain_update_poisoned(self):
        backend, client, _ = self.tinker()
        client.optim_step.side_effect = TimeoutError('unknown')
        batch = UpdateBatch('cross_entropy', (TrainingRow((1,), (2,), (1.,)),))
        with self.assertRaises(UncertainUpdateError):
            backend.update(batch, .1)
        with self.assertRaises(UncertainUpdateError):
            backend.snapshot('bad')
        with self.assertRaises(UncertainUpdateError):
            backend.update(batch, .1)
        self.assertEqual(client.forward_backward.call_count, 1)
        with self.assertRaises(UncertainUpdateError):
            backend.load_training('tinker://run/weights/a', optimizer=True)

    def test_grpo_strategy_to_backend_normalizes_exactly_once(self):
        backend, client, _ = self.tinker()
        group = tuple(ScoredTrajectory(
            Trajectory('task', str(i), 'policy', 'env',
                       (Sample((1, 2), (48, 49), (-.69, -.69)),), ''),
            Verification('resolved', float(i))) for i in range(2))
        batch = GRPOStrategy().build_batch([group], context_limit=10)
        backend.update(batch, .1)
        rows = client.forward_backward.call_args.args[0]
        self.assertEqual(rows[0].loss_fn_inputs['advantages'].data, [0., -.25, -.25])
        self.assertEqual(rows[1].loss_fn_inputs['advantages'].data, [0., .25, .25])

    def test_load_and_save_api(self):
        backend, client, service = self.tinker()
        client.save_state.return_value = future(NS(path='tinker://run/weights/a'))
        client.save_weights_for_sampler.return_value = future(NS(path='tinker://run/sampler_weights/a'))
        artifacts = backend.save('a')
        backend.verify_artifacts(artifacts)
        backend.load_training(artifacts.training, optimizer=True)
        client.load_state_with_optimizer.assert_called_once_with(artifacts.training)
        client.load_state_with_optimizer.return_value.result.assert_called_once()
        backend.load_training(artifacts.training, optimizer=False)
        client.load_state.assert_called_once_with(artifacts.training)
        backend.load_sampling(artifacts.sampling)
        service.create_sampling_client.assert_called_once_with(model_path=artifacts.sampling)

    def test_sampler_preserves_tokens_probabilities_and_checks_context(self):
        backend, client, _ = self.tinker()
        sampler = Mock()
        sampler.sample.return_value = future(NS(sequences=[NS(tokens=[41, 42], logprobs=[-.2, -.4], stop_reason='stop')]))
        client.save_weights_and_get_sampling_client.return_value = sampler
        policy = backend.snapshot('one')
        sample = policy.sample([7, 8], max_tokens=2, temperature=1., seed=123)
        self.assertEqual(sample.prompt_tokens, (7, 8))
        self.assertEqual(sample.tokens, (41, 42))
        self.assertEqual(sample.logprobs, (-.2, -.4))
        self.assertEqual(sampler.sample.call_args.kwargs['sampling_params'].seed, 123)
        with self.assertRaises(ValueError):
            policy.sample([1] * 9, max_tokens=2, temperature=1., seed=0)
        sampler.sample.return_value = future(NS(sequences=[NS(tokens=[41], logprobs=None, stop_reason='stop')]))
        with self.assertRaises(InfrastructureError):
            policy.sample([7], max_tokens=2, temperature=1., seed=0)

    def test_model_factory_identity_and_purpose(self):
        factory = FakeBackendFactory(self.directory.name)
        backend = factory('mock', training=True)
        artifacts = backend.save('x')
        manifest = {'identity': asdict(backend.identity), 'artifacts': asdict(artifacts)}
        store = NS(read=lambda ref: manifest)
        models = ModelFactory(factory, store)
        for purpose in ('evaluate', 'fork', 'resume'):
            self.assertIs(models.resolve(ModelRef(checkpoint='manifest'), purpose).manifest, manifest)
        with self.assertRaises(ValueError):
            models.resolve(ModelRef(base_model='mock'), 'resume')
        with self.assertRaises(ValueError):
            models.resolve(ModelRef(checkpoint='manifest'), 'train')
        with self.assertRaises(CapabilityError):
            models.resolve(ModelRef(base_model='mock'), 'train', {'critic'})
        manifest['identity']['renderer'] = 'wrong'
        with self.assertRaises(CapabilityError):
            models.resolve(ModelRef(checkpoint='manifest'), 'resume')

    def test_context_and_catalog_fail_before_training(self):
        backend, client, service = self.tinker()
        with self.assertRaises(ValueError):
            backend.update(UpdateBatch('cross_entropy', (TrainingRow((1,), (2, 3), (1.,)),)), .1)
        client.forward_backward.assert_not_called()
        service.get_server_capabilities.return_value = NS(supported_models=[NS(model_name='other')])
        factory = TinkerBackendFactory(renderer='tokens', context_limit=10, service=service, sdk=sdk_stub())
        with self.assertRaises(CapabilityError):
            factory('base', training=True)

    def test_stop_sequences_are_forwarded_and_part_of_identity(self):
        _, client, service = self.tinker()
        service.get_server_capabilities.return_value = NS(supported_models=[NS(model_name='base')])
        factory = TinkerBackendFactory(renderer='tool-v1', context_limit=10, service=service,
                                       sdk=sdk_stub(), stop=[99])
        backend = factory('base', training=True)
        self.assertEqual(asdict(backend.identity)['adaptation']['stop'], [99])
        sampler = client.save_weights_and_get_sampling_client.return_value
        sampler.sample.return_value = future(NS(sequences=[NS(tokens=[99], logprobs=[-.2], stop_reason='stop')]))
        backend.snapshot('tool').sample([1], max_tokens=1, temperature=1., seed=1)
        self.assertEqual(sampler.sample.call_args.kwargs['sampling_params'].stop, (99,))
        for bad in ([1, 'x'], [-1], [''], ''):
            with self.assertRaises(ValueError):
                TinkerBackendFactory(renderer='x', context_limit=10, stop=bad)

    def test_invalid_direct_rows_fail_without_remote_update(self):
        backend, client, _ = self.tinker()
        for row in (TrainingRow((-1,), (2,), (1.,)), TrainingRow((1,), (True,), (1.,)),
                    TrainingRow((1,), (2,), (1.,), (.1,), (1.,))):
            loss = 'importance_sampling' if row.old_logprobs else 'cross_entropy'
            with self.assertRaises(ValueError):
                backend.update(UpdateBatch(loss, (row,)), .1)
        client.forward_backward.assert_not_called()


if __name__ == '__main__':
    unittest.main()
