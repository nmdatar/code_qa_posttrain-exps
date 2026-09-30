import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from training_pipeline.continuation import prepare_continuation, execute_continuation, remaining_estimate
from training_pipeline.contracts import ConfigurationError
from training_pipeline.storage import atomic_json, digest, semantic_hash, read


class ContinuationTests(unittest.TestCase):
    def fixture(self, root):
        config = {'run_id': 'failed', 'output': str(root/'failed'), 'tracking': {'mode': 'disabled'},
                  'seed': 42, 'spend': {'ledger': '/state/same-ledger', 'cap_usd': 1800}}
        value = {'id': 'known-checkpoint', 'config': config, 'config_hash': semantic_hash(config),
                 'state': {'data_identity': 'data', 'optimizer_step': 2, 'attempted_batches': 2,
                           'cursor': 16, 'rng': ['exact-state'], 'reinforce_baselines': {'0': {'count': 64, 'reward_sum': 9.45}}},
                 'artifacts': {'training': 'immutable-training', 'sampler': 'immutable-sampler'}}
        path = root/'checkpoint.json'
        atomic_json(path, {**value, 'manifest_hash': digest(value)})
        return path

    def test_preflight_keeps_checkpoint_and_state_unchanged(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); checkpoint = self.fixture(root); before = checkpoint.read_bytes()
            with patch('training_pipeline.continuation.inputs', return_value={'identity': 'data'}):
                config, data, receipt = prepare_continuation(checkpoint, root/'continued', 'continued')
            self.assertEqual(receipt['optimizer_step'], 2)
            self.assertEqual(receipt['cursor'], 16)
            self.assertEqual(receipt['reinforce_baselines']['0'], {'count': 64, 'reward_sum': 9.45})
            self.assertEqual(config['spend']['ledger'], '/state/same-ledger')
            self.assertEqual(checkpoint.read_bytes(), before)
            self.assertFalse((root/'continued').exists())

    def test_rejects_output_reuse_or_changed_data(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); checkpoint = self.fixture(root)
            with self.assertRaises(ConfigurationError):
                prepare_continuation(checkpoint, root/'failed', 'continued')
            with patch('training_pipeline.continuation.inputs', return_value={'identity': 'other'}):
                with self.assertRaisesRegex(ConfigurationError, 'identity'):
                    prepare_continuation(checkpoint, root/'continued', 'continued')

    def test_execution_delegates_resume_and_persists_provenance(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); checkpoint = self.fixture(root)
            with patch('training_pipeline.continuation.inputs', return_value={'identity': 'data'}), \
                 patch('training_pipeline.orchestrator.Pipeline') as pipeline:
                execute_continuation(checkpoint, root/'continued', 'continued')
            pipeline.return_value.run.assert_called_once_with(checkpoint=checkpoint, purpose='resume')
            self.assertEqual(read(root/'continued'/'continuation.json')['checkpoint_id'], 'known-checkpoint')
            self.assertEqual(read(root/'continued'/'config.json')['spend']['cap_usd'], 1800)

    def test_remaining_estimate_preserves_execution_config_and_state(self):
        config = {'stages': [{'max_updates': 32, 'max_batches': 32}], 'evaluation': {'every': 16}}
        state = {'stage': 0, 'stage_updates': 26, 'stage_batches': 26, 'optimizer_step': 26}
        before = copy.deepcopy((config, state))
        with patch('training_pipeline.launch.estimate', return_value={'upper_estimate_usd': 1}) as estimate:
            result = remaining_estimate(config, state, {})
        self.assertEqual(estimate.call_args.args[0]['stages'], [{'max_updates': 6, 'max_batches': 6}])
        self.assertEqual((config, state), before)
        self.assertEqual(result['resume_optimizer_step'], 26)

    def test_finished_stage_cannot_be_rebudgeted_as_continuation(self):
        config = {'stages': [{'max_updates': 32, 'max_batches': 32}]}
        state = {'stage': 0, 'stage_updates': 32, 'stage_batches': 32, 'optimizer_step': 32}
        with self.assertRaises(ConfigurationError):
            remaining_estimate(config, state, {})

    def test_injected_budget_checked_backend_passed_to_native_resume(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); checkpoint = self.fixture(root)
            backend = object()
            with patch('training_pipeline.continuation.inputs', return_value={'identity': 'data'}), \
                 patch('training_pipeline.orchestrator.Pipeline') as pipeline:
                execute_continuation(checkpoint, root/'continued', 'continued', backend=backend)
            self.assertIs(pipeline.call_args.kwargs['backend'], backend)
