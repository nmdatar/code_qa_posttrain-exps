"""Offline regression coverage for portable, separately budgeted evaluation."""
import copy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import unittest
import tempfile

from training_pipeline.contracts import ConfigurationError
from training_pipeline.orchestrator import evaluate_checkpoint
from training_pipeline.remote import resolve_checkpoint_pointer
from training_pipeline.storage import atomic_json, digest


def fixture_config():
    return {'model': {'base_model': 'same', 'rank': 8}, 'seed': 42,
            'limits': {'max_tool_calls': 5},
            'environment': {'kind': 'collection', 'release': '/old/release',
                            'manifest_sha256': 'release-hash', 'grading_version': 'v7'},
            'judge': {'base_model': 'judge'}, 'training_reward': {'version': 'coverage'},
            'evaluation': {'cohort_manifest': '/old/cohort.json', 'cohort_sha256': 'cohort-hash',
                           'max_tasks': 32, 'temperature': 0, 'every': 16},
            'output': '/old/run', 'run_id': 'old', 'spend': {'ledger': '/old/ledger'}}


def manifest(config):
    value = {'config': config, 'state': {'data_identity': 'same-data'},
             'identity': {key: key for key in ('base_model', 'rank', 'template_hash', 'tokenizer_class', 'renderer')},
             'artifacts': {'sampler': 'original-weights'}, 'id': 'checkpoint'}
    return {**value, 'manifest_hash': digest(value)}


def test_rebinds_paths_and_ledger_without_loading_old_inputs(tmp_path):
    old = fixture_config()
    checkpoint = tmp_path / 'checkpoint.json'
    atomic_json(checkpoint, manifest(old))
    updated = copy.deepcopy(old)
    updated['environment']['release'] = '/new/release'
    updated['evaluation']['cohort_manifest'] = '/new/cohort.json'
    updated.update(output='/new/eval', run_id='new-eval', spend={'ledger': '/new/ledger'})
    observed = {}
    def build(config, **kwargs):
        observed.update(copy.deepcopy(config))
        backend = Mock(identity=manifest(old)['identity'])
        return SimpleNamespace(config=config, data={'identity': 'same-data'},
                               backend=backend, tracker=Mock(), factory=Mock(), setup=Mock(),
                               evaluate=Mock(return_value={'complete': True}))
    with patch('training_pipeline.orchestrator.Pipeline', side_effect=build) as pipeline, \
         patch('training_pipeline.cohorts.select_tasks'):
        assert evaluate_checkpoint(checkpoint, config=updated, cohort='confirmation') == {'complete': True}
    assert observed['spend']['ledger'] == '/new/ledger'
    assert observed['environment']['release'] == '/new/release'
    assert observed['evaluation']['cohort_manifest'] == '/new/cohort.json'
    assert old['spend']['ledger'] == '/old/ledger'


SCIENTIFIC_CHANGES = [
    ('model', 'base_model', 'other'), ('model', 'rank', 16),
    ('limits', 'max_tool_calls', 99), ('environment', 'grading_version', 'other'),
    ('environment', 'manifest_sha256', 'other'), ('judge', 'base_model', 'other'),
    ('training_reward', 'version', 'other'), ('evaluation', 'cohort_sha256', 'other'),
    ('evaluation', 'temperature', 1), ('evaluation', 'max_tasks', 2),
    (None, 'seed', 43),
]
def test_scientific_changes_fail_before_pipeline_or_paid_setup(tmp_path, field, key, value):
    config = fixture_config()
    checkpoint = tmp_path / 'checkpoint.json'
    atomic_json(checkpoint, manifest(config))
    updated = copy.deepcopy(config)
    (updated[field] if field else updated)[key] = value
    with patch('training_pipeline.orchestrator.Pipeline') as pipeline:
        with unittest.TestCase().assertRaisesRegex(ConfigurationError, 'scientific'):
            evaluate_checkpoint(checkpoint, config=updated)
        pipeline.assert_not_called()


def test_data_identity_checked_before_backend_setup(tmp_path):
    config = fixture_config()
    checkpoint = tmp_path / 'checkpoint.json'
    atomic_json(checkpoint, manifest(config))
    with patch('training_pipeline.orchestrator.Pipeline') as constructor:
        constructor.return_value.data = {'identity': 'wrong'}
        with unittest.TestCase().assertRaisesRegex(ConfigurationError, 'dataset identity'):
            evaluate_checkpoint(checkpoint, config=config)
        constructor.return_value.setup.assert_not_called()


def test_latest_and_best_pointer_resolve_inside_volume(tmp_path, key):
    target = tmp_path / 'checkpoint.json'
    atomic_json(target, manifest(fixture_config()))
    atomic_json(tmp_path / 'pointer.json', {key: str(target)})
    assert resolve_checkpoint_pointer(tmp_path, 'pointer.json').resolve() == target.resolve()
    atomic_json(tmp_path / 'pointer.json', {key: 'checkpoint.json'})
    assert resolve_checkpoint_pointer(tmp_path, 'pointer.json').resolve() == target.resolve()


def test_pointer_cannot_escape_volume(tmp_path, target):
    atomic_json(tmp_path / 'pointer.json', {'path': target})
    with unittest.TestCase().assertRaises(ConfigurationError):
        resolve_checkpoint_pointer(tmp_path, 'pointer.json')


def test_no_config_preserves_original_ledger_and_existing_output(tmp_path):
    config = fixture_config()
    checkpoint = tmp_path / 'checkpoint.json'
    atomic_json(checkpoint, manifest(config))
    pipeline = Mock()
    pipeline.config = config
    pipeline.data = {'identity': 'same-data'}
    pipeline.backend.identity = manifest(config)['identity']
    with patch('training_pipeline.orchestrator.Pipeline', return_value=pipeline) as constructor, \
         patch('training_pipeline.cohorts.select_tasks'):
        evaluate_checkpoint(checkpoint)
    assert constructor.call_args.args[0]['spend']['ledger'] == '/old/ledger'
    pipeline.setup.assert_called_once_with(create_run=False, job_type='evaluation')
    pipeline.backend.load.assert_called_once_with({'sampler': 'original-weights'}, 'evaluate')


class ExpandedEvaluationTests(unittest.TestCase):
    def test_paths_and_ledger(self):
        with tempfile.TemporaryDirectory() as root:
            test_rebinds_paths_and_ledger_without_loading_old_inputs(Path(root))

    def test_scientific_changes(self):
        for field, key, value in SCIENTIFIC_CHANGES:
            with self.subTest(field=field, key=key), tempfile.TemporaryDirectory() as root:
                test_scientific_changes_fail_before_pipeline_or_paid_setup(Path(root), field, key, value)

    def test_input_identity(self):
        with tempfile.TemporaryDirectory() as root:
            test_data_identity_checked_before_backend_setup(Path(root))

    def test_pointers(self):
        for key in ('path', 'manifest'):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as root:
                test_latest_and_best_pointer_resolve_inside_volume(Path(root), key)

    def test_pointer_escape(self):
        for target in ('/outside/checkpoint.json', '../checkpoint.json'):
            with self.subTest(target=target), tempfile.TemporaryDirectory() as root:
                test_pointer_cannot_escape_volume(Path(root), target)

    def test_legacy_evaluation(self):
        with tempfile.TemporaryDirectory() as root:
            test_no_config_preserves_original_ledger_and_existing_output(Path(root))
