import copy
import tempfile
import unittest

from qa_eval.demo import fixture
from training_pipeline.reward_alignment import feedback, VERSION
from training_pipeline.strict_grading import assess, RELIABLE_VERSION
from tests.test_strict_collection_grading import StrictCollectionGradingTests


class AlignedRewardTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.task, self.answer, _, _, self.semantic = fixture(self.tmp.name)
        self.rubric = {'claims': self.task['claims']}

    def score(self, semantic=None, strict=0., errors=()):
        return feedback(semantic or self.semantic, self.rubric, strict, errors)

    def test_correct_partial_wrong_ranking(self):
        self.assertEqual(self.score()['reward'], 1.)
        self.semantic['required_claims'][0]['coverage'] = 'partial'
        self.assertEqual(self.score()['reward'], .5)
        self.semantic['required_claims'][0].update(coverage='absent', verdict='contradicted')
        self.assertEqual(self.score()['reward'], 0.)

    def test_source_backed_falsehood_reduces_full_coverage(self):
        self.semantic['additional_claims'][0].update(verdict='contradicted', material_error=True)
        self.assertEqual(self.score()['reward'], .25)
        self.assertEqual(self.score()['components']['required_coverage'], 1.)

    def test_missing_source_cannot_become_falsehood_penalty(self):
        claim = self.semantic['additional_claims'][0]
        claim.update(verdict='contradicted', material_error=True, evidence_keys=[])
        self.assertFalse(self.score()['eligible'])
        claim['evidence_keys'] = ['invented-source']
        self.assertIsNone(self.score()['reward'])

    def test_unknown_is_not_contradiction(self):
        self.semantic['additional_claims'][0].update(verdict='insufficient', evidence_keys=[])
        result = self.score()
        self.assertEqual(result['components']['contradicted'], 0)
        self.assertAlmostEqual(result['reward'], .85)

    def test_filler_cannot_dilute_penalty(self):
        self.semantic['additional_claims'][0].update(verdict='contradicted', material_error=True)
        before = self.score()['reward']
        for n in range(20):
            ident = f'filler-{n}'
            self.semantic['additional_claims'].append({**self.semantic['required_claims'][0], 'id': ident})
            self.semantic['extracted_claims'].append({'id': ident})
            self.semantic['citation_links'].append({'claim_id': ident, 'citation_id': 'src', 'supported': True})
        self.assertEqual(self.score()['reward'], before)

    def test_citations_penalized_without_erasing_partial_credit(self):
        self.semantic['required_claims'][0]['coverage'] = 'partial'
        self.semantic['citation_links'][0]['supported'] = False
        self.semantic['uncited_claim_ids'] = ['a1']
        result = self.score()
        self.assertAlmostEqual(result['reward'], .4)
        self.assertEqual(result['components']['uncited'], 1)
        self.assertAlmostEqual(self.score(errors=['invalid_citation'])['reward'], .4)

    def test_deterministic_invalid_citation_recorded(self):
        self.assertAlmostEqual(self.score(errors=['invalid_citation'])['reward'], .95)

    def test_independent_factual_coverage_is_exact_control_base(self):
        self.semantic['additional_claims'][0].update(verdict='insufficient')
        result = feedback(self.semantic, self.rubric, 0., factual_coverage=.5)
        self.assertAlmostEqual(result['reward'], .35)
        self.assertEqual(result['components']['required_coverage'], .5)
        self.assertEqual(result['definition']['coverage_version'], 'independent-factual-coverage-v3')

    def test_missing_independent_coverage_excluded_invalid_rejected(self):
        self.assertFalse(feedback(self.semantic, self.rubric, 0., factual_coverage=None)['eligible'])
        for invalid in (True, float('nan'), -1., 1.01, '1'):
            with self.assertRaises(ValueError):
                feedback(self.semantic, self.rubric, 0., factual_coverage=invalid)

    def test_independent_coverage_does_not_bypass_strict_eligibility(self):
        self.semantic['additional_claims'][0].update(verdict='contradicted', evidence_keys=[])
        self.assertFalse(feedback(self.semantic, self.rubric, 0., factual_coverage=1.)['eligible'])

    def test_material_fallback_and_false_execution(self):
        self.semantic['critical_error'] = True
        self.assertIsNone(self.score()['reward'])
        self.semantic['additional_claims'][0]['material_error'] = True
        self.assertEqual(self.score()['reward'], .25)
        self.semantic['false_execution_claim'] = True
        self.assertEqual(self.score()['reward'], 0.)

    def test_required_contradiction_without_material_flag_still_penalized(self):
        second = {**self.rubric['claims'][0], 'id': 'c2'}
        self.rubric['claims'].append(second)
        self.semantic['required_claims'].append({**self.semantic['required_claims'][0], 'id': 'c2'})
        self.semantic['required_claims'][0].update(verdict='contradicted', material_error=False)
        result = self.score()
        self.assertEqual(result['components']['required_coverage'], .5)
        self.assertEqual(result['components']['contradicted'], 1)
        self.assertEqual(result['reward'], 0.)

    def test_unresolved_is_excluded_and_no_input_mutation(self):
        before = copy.deepcopy(self.semantic)
        self.assertEqual(self.score()['version'], VERSION)
        self.assertEqual(self.semantic, before)
        self.assertIsNone(self.score(strict=None)['reward'])
        for key, value in [('needs_review', True), ('assessment_complete', False),
                           ('disagreements', ['ambiguous'])]:
            semantic = copy.deepcopy(before)
            semantic[key] = value
            self.assertFalse(self.score(semantic)['eligible'])

    def test_duplicate_and_invalid_fixed_weights_rejected(self):
        self.rubric['claims'][0]['weight'] = float('nan')
        with self.assertRaises(ValueError):
            self.score()
        self.rubric['claims'][0]['weight'] = 1
        self.semantic['required_claims'].append(copy.deepcopy(self.semantic['required_claims'][0]))
        with self.assertRaises(ValueError):
            self.score()


class AlignedFullGraderTests(StrictCollectionGradingTests):
    def test_training_source_fixture_full_grader_reward_keeps_strict_zero(self):
        self.assessment['additional_claims'][0].update(verdict='contradicted', material_error=True)
        result = assess(self.request, self.call, RELIABLE_VERSION)
        self.assertEqual(result['score'], 0.)
        reward = feedback(result['semantic'], self.request['rubric'], result['score'])
        self.assertEqual(reward['reward'], .25)
        self.assertEqual(self.calls, ['extract', 'assess'])


class AlignedCollectionTests(unittest.TestCase):
    from tests.test_collection_training import CollectionTests
    setUp = CollectionTests.setUp
    episode = CollectionTests.episode

    def prepare(self):
        from pathlib import Path
        self.config['environment']['grading_version'] = 'all-claims-v7'
        self.config['training_reward'] = {'version': VERSION}
        self.row['public']['budgets']['max_submission_bytes'] = 64000
        task, _, _, _, semantic = fixture(Path(self.tmp.name) / 'source-fixture')
        self.row['rubric']['claims'] = task['claims']
        return {'status': 'resolved', 'strict_status': 'resolved', 'score': 0.,
                'strict_score': 0., 'reason': 'strict defect', 'semantic': semantic,
                'claims': semantic['required_claims'], 'claim_count': 1, 'rubric_hash': 'hash',
                'coverage_assessment': {'status': 'resolved', 'score': .5}}

    def verify(self, grade, name='alignment', invalid=False):
        from unittest.mock import Mock, patch
        from training_pipeline.contracts import Trajectory
        from training_pipeline.admission import sha
        self.factory.grade = Mock(return_value=grade)
        answer = {'schema_version': '1.0', 'task_id': 'task', 'text': 'Candidate fact', 'diagram': None,
                  'citations': [{'id': 'src', 'path': 'missing.py' if invalid else 'a.py',
                                 'start_line': 1, 'end_line': 1, 'file_sha256': sha(b'pass\n')}]}
        trajectory = Trajectory('r', 0, 'task', 'h', 'g', name, 'p', 'e', 'x', self.row['split'],
                                submission=answer, termination='completed')
        with patch('training_pipeline.collection.blob', return_value=b'pass\n'):
            return self.episode(name).verify(trajectory)

    def test_training_exact_coverage_with_penalties_dev_remains_strict(self):
        grade = self.prepare()
        grade['semantic']['additional_claims'][0]['verdict'] = 'insufficient'
        result = self.verify(grade)
        self.assertAlmostEqual(result.reward, .35)
        self.assertEqual(result.diagnostics['strict_score'], 0.)
        self.row['split'] = 'development'
        result = self.verify(grade, 'dev')
        self.assertEqual(result.reward, 0.)
        self.assertEqual(result.diagnostics['reward_applied'], 'strict')

    def test_missing_or_unresolved_audits_are_excluded(self):
        original = self.prepare()
        for i, changes in enumerate(({'semantic': None}, {'strict_status': 'unresolved'},
                                     {'coverage_assessment': {'status': 'unresolved', 'score': None}},
                                     {'coverage_assessment': {'status': 'resolved', 'score': None}})):
            result = self.verify({**copy.deepcopy(original), **changes}, f'excluded-{i}')
            self.assertEqual(result.status, 'unresolved')
            self.assertIsNone(result.reward)

    def test_invalid_citations_removed_from_evidence_and_penalized(self):
        grade = self.prepare()
        result = self.verify(grade, invalid=True)
        self.assertAlmostEqual(result.reward, .45)
        self.assertEqual(result.diagnostics['strict_score'], 0.)
        self.assertEqual(self.factory.grade.call_args.args[0]['answer']['citations'], [])
        grade['strict_status'] = 'unresolved'
        self.assertIsNone(self.verify(grade, 'invalid-unresolved', invalid=True).reward)

    def test_same_independent_pass_runs_for_alignment_training_only(self):
        import json
        from pathlib import Path
        from types import SimpleNamespace
        from unittest.mock import patch
        from training_pipeline.collection import CollectionFactory
        config = json.loads(Path('tests/fixtures/experiment_contracts/reward-diagnostic.json').read_text())
        config['environment']['grading_version'] = 'all-claims-v7'
        config['training_reward'] = {'version': VERSION}
        factory = CollectionFactory(config, self.root, None, judge=SimpleNamespace())
        request = {'episode_id': 'grade', 'source_row': {'split': 'train'},
                   'rubric': {'claims': [], 'rubric_hash': 'hash'}}
        strict = {'status': 'resolved', 'score': 0., 'strict_score': 0., 'reason': 'defect'}
        with patch('training_pipeline.strict_grading.assess', return_value=strict), \
             patch('training_pipeline.coverage_judge.assess_coverage', return_value={
                 'status': 'resolved', 'score': .5, 'reason': 'coverage'}) as coverage:
            result = factory.grade(request)
            self.assertEqual(result['coverage_assessment']['score'], .5)
            self.assertEqual(result['score'], 0.)
            request['source_row']['split'] = 'development'
            self.assertNotIn('coverage_assessment', factory.grade(request))
            self.assertEqual(coverage.call_count, 1)

    def test_config_requires_v7_and_identity_binds_coefficients(self):
        import json
        from pathlib import Path
        from unittest.mock import patch
        from training_pipeline.config import validate_config
        from training_pipeline.collection import CollectionFactory
        from training_pipeline.contracts import ConfigurationError
        config = json.loads(Path('tests/fixtures/experiment_contracts/reward-diagnostic.json').read_text())
        config['training_reward'] = {'version': VERSION}
        config['environment']['grading_version'] = 'all-claims-v7'
        validate_config(config)
        before = CollectionFactory(config, self.root, None).reward_version
        with patch('training_pipeline.reward_alignment.WEIGHTS', {'contradicted': .7}):
            self.assertNotEqual(before, CollectionFactory(config, self.root, None).reward_version)
        config['training_reward'] = {'version': 'positive-coverage-v4'}
        control = CollectionFactory(config, self.root, None).reward_version
        with patch('training_pipeline.reward_alignment.WEIGHTS', {'contradicted': .7}):
            self.assertEqual(control, CollectionFactory(config, self.root, None).reward_version)
        config['training_reward'] = {'version': VERSION}
        config['environment']['grading_version'] = 'all-claims-v6'
        with self.assertRaises(ConfigurationError):
            validate_config(config)
