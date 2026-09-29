import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from training_pipeline.cohorts import build_cohorts, load_cohorts, select_tasks
from training_pipeline.contracts import ConfigurationError, Trajectory, VerificationResult
from training_pipeline.orchestrator import Pipeline, evaluate_base
from training_pipeline.smoke import smoke_config
from training_pipeline.storage import atomic_json, load_checkpoint
from training_pipeline.toy import toy_inputs
from tests.test_training_pipeline import FakeBackend, PRICES


class ReadinessTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.config = smoke_config(self.root, PRICES, self.root/'spend.json')

    def cohorts(self):
        data = toy_inputs()
        data['development'] = [dict(id=f'd{i}', split='development', family_id=f'f{i%3}') for i in range(117)]
        return data, build_cohorts(data)

    def test_cohorts_are_stratified_disjoint_and_order_independent(self):
        data, manifest = self.cohorts()
        self.assertEqual(len(manifest['selection']), 32)
        self.assertEqual(len(manifest['confirmation']), 85)
        self.assertFalse(set(manifest['selection']) & set(manifest['confirmation']))
        data['development'].reverse()
        self.assertEqual(build_cohorts(data), manifest)
        self.assertEqual(sorted(v['selection'] for v in manifest['families'].values()), [10, 11, 11])
        data['development'][0]['family_id'] = data['tasks'][0]['family_id']
        with self.assertRaises(ConfigurationError):
            build_cohorts(data)

    def test_cohort_routing_integrity_and_no_truncation(self):
        data, manifest = self.cohorts()
        path = self.root/'cohorts.json'
        atomic_json(path, manifest)
        evaluation = dict(max_tasks=32, cohort_manifest=str(path), cohort_sha256=manifest['manifest_hash'])
        selected, identity = select_tasks(data, evaluation, 'confirmation')
        self.assertEqual(len(selected), 85)
        self.assertEqual(identity['name'], 'confirmation')
        manifest['selection'][0] = 'changed'
        atomic_json(path, manifest)
        with self.assertRaises(ConfigurationError):
            load_cohorts(path, evaluation['cohort_sha256'], data)

    def test_epoch_boundaries_are_unique_and_resume_exactly(self):
        pipeline = Pipeline(self.config, backend=FakeBackend())
        rows = pipeline.data['tasks']
        pipeline._batch(rows, 6)
        state, rng = copy.deepcopy(pipeline.state), pipeline.rng.getstate()
        actual = pipeline._batch(rows, 6)
        self.assertEqual(len({r['id'] for r in actual}), 6)
        clone = Pipeline(self.config, backend=FakeBackend())
        clone.state, clone.rng = state, __import__('random').Random()
        clone.rng.setstate(rng)
        self.assertEqual(clone._batch(rows, 6), actual)
        all_ids = [r['id'] for r in rows]
        self.assertEqual(sorted(rows[i]['id'] for i in pipeline.state['order']), sorted(all_ids))

    def test_sft_final_batch_does_not_repeat(self):
        pipeline = Pipeline(self.config, backend=FakeBackend())
        rows = pipeline.data['sft']
        first = pipeline._batch(rows, 6, partial=True)
        final = pipeline._batch(rows, 6, partial=True)
        self.assertEqual(len(final), 2)
        self.assertEqual(len({r['id'] for r in first+final}), 8)

    def test_base_evaluation_never_allocates_trainer_or_saves(self):
        backend = FakeBackend()
        report = evaluate_base(self.config, backend=backend)
        self.assertEqual(backend.trainer_count, 0)
        self.assertFalse(any(c[0] == 'save' for c in backend.calls))
        self.assertEqual(report['checkpoint_id'], 'unchanged-base')
        self.assertEqual(report['scoring_coverage'], 1)
        self.assertEqual(report['completion_rate'], 1)

    def test_unresolved_affects_demonstrated_quality_not_training_reward(self):
        pipeline = Pipeline(self.config, backend=FakeBackend())
        pipeline.setup(True)
        outcomes = iter([VerificationResult('resolved', .8, 'test'), VerificationResult('unresolved', None, 'test')])
        def rollout(task, *_):
            return Trajectory('r', 0, task['id'], 'h', 'g', task['id'], 'p', 'e', 'x', 'development',
                              termination='completed', verification=next(outcomes))
        pipeline.rollout = rollout
        report = pipeline.evaluate(base=True)
        self.assertEqual(report['demonstrated_quality'], .4)
        self.assertEqual(report['mean_reward'], .8)
        self.assertEqual(report['scoring_coverage'], .5)
        self.assertIsNone(report['results'][1]['reward'])
        self.assertFalse(report['coverage_eligible'])

    def test_no_signal_stops_and_cannot_blindly_resume(self):
        self.config['stages'] = [dict(kind='grpo', max_updates=30, max_batches=60,
            batch_size=2, group_size=4, temperature=1, learning_rate=1e-5)]
        self.config['stopping'] = {'initial_zero_batches': 5}
        backend = FakeBackend()
        with patch('training_pipeline.orchestrator.grpo_batch', return_value=([], {})):
            checkpoint = Pipeline(self.config, backend=backend).run()
        state = load_checkpoint(checkpoint)['state']
        self.assertEqual(state['stage_batches'], 5)
        self.assertEqual(state['optimizer_step'], 0)
        self.assertEqual(state['stop_reason'], 'initial_zero_batches')
        with self.assertRaisesRegex(ConfigurationError, 'safety rule'):
            Pipeline(self.config, backend=backend).run(checkpoint, 'resume')

    def test_regression_requires_consecutive_checks(self):
        self.config['stopping'] = {'regression_delta': .1, 'regression_checks': 2}
        pipeline = Pipeline(self.config, backend=FakeBackend())
        def report(score):
            return {'demonstrated_quality': score, 'coverage_eligible': True}
        self.assertFalse(pipeline._regression_stop(report(.8)))
        self.assertFalse(pipeline._regression_stop(report(.7)))
        self.assertFalse(pipeline._regression_stop(report(.8)))
        self.assertFalse(pipeline._regression_stop(report(.65)))
        self.assertTrue(pipeline._regression_stop(report(.7)))

    def test_regression_stop_is_committed_after_two_updates(self):
        self.config['stages'] = [dict(kind='grpo', max_updates=30, max_batches=60,
            batch_size=2, group_size=4, temperature=1, learning_rate=1e-5)]
        self.config['stopping'] = {'regression_delta': .1, 'regression_checks': 2}
        backend = FakeBackend()
        reports = [dict(demonstrated_quality=q, coverage_eligible=True) for q in (.8, .65, .6)]
        with patch.object(Pipeline, 'evaluate', side_effect=reports):
            checkpoint = Pipeline(self.config, backend=backend).run()
        state = load_checkpoint(checkpoint)['state']
        self.assertEqual(state['optimizer_step'], 2)
        self.assertEqual(state['baseline_quality'], .8)
        self.assertEqual(state['regression_count'], 2)
        self.assertEqual(state['stop_reason'], 'selection_regression')

    def test_invalid_cohort_fails_before_provider_setup(self):
        with patch.object(Pipeline, 'setup') as setup:
            with self.assertRaises(ConfigurationError):
                evaluate_base(self.config, cohort='confirmation')
        setup.assert_not_called()

    def test_comparison_pairs_tasks_and_preserves_unresolved_sensitivity(self):
        from training_pipeline.comparison import compare
        rows = [dict(task_id=str(i), family_id='f'+str(i//2), reward=.5,
                     status='resolved', termination='completed') for i in range(6)]
        base = dict(expected=6, results=rows, data_identity='data', environment='env', reward_version='v', cohort=None)
        better = copy.deepcopy(base)
        for row in better['results']:
            row['reward'] = .75
        result = compare(base, [better], repetitions=100)
        self.assertEqual(result['quality_difference_95ci'], [.25, .25])
        self.assertTrue(result['passes_quality_gate'])
        self.assertEqual(result, compare(base, [better], repetitions=100))
        better['results'][0].update(status='unresolved', reward=None)
        result = compare(base, [better], repetitions=100)
        self.assertFalse(result['passes_quality_gate'])
        self.assertLess(*result['unresolved_sensitivity_bounds'])
        better['results'][0]['task_id'] = 'wrong'
        with self.assertRaises(ConfigurationError):
            compare(base, [better])
