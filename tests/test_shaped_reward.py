import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from qa_eval.demo import fixture
from training_pipeline.shaped_reward import shape, group_signal
from training_pipeline.evidence_compaction import compact
from training_pipeline.contracts import VerificationResult, Generation
from training_pipeline.strict_grading import assess, RELIABLE_VERSION
from tests import test_strict_collection_grading as strict_tests


class ShapedRewardTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.task, self.answer, _, _, self.semantic = fixture(self.tmp.name)
        self.rubric = {'claims': self.task['claims']}

    def score(self, s=None, strict=0):
        return shape(s or self.semantic, self.rubric, strict, version='supported-coverage-v3')['reward']

    def test_correct_partial_incorrect_ranking(self):
        correct = self.score(strict=1)
        partial = copy.deepcopy(self.semantic)
        partial['required_claims'][0]['coverage'] = 'partial'
        wrong = copy.deepcopy(self.semantic)
        wrong['required_claims'][0].update(verdict='contradicted', material_error=True)
        wrong['additional_claims'][0].update(verdict='contradicted', material_error=True)
        self.assertGreater(correct, self.score(partial))
        self.assertGreater(self.score(partial), self.score(wrong))
        self.assertEqual(self.score(partial), .5)

    def test_adding_supported_filler_cannot_dilute_penalty(self):
        s = copy.deepcopy(self.semantic)
        s['additional_claims'].append({**s['additional_claims'][0], 'id':'bad', 'verdict':'insufficient'})
        before = self.score(s)
        for i in range(20):
            ident = 'filler'+str(i)
            s['additional_claims'].append({**s['additional_claims'][0], 'id':ident})
            s['citation_links'].append({'claim_id':ident,'citation_id':'src','supported':True})
        self.assertEqual(self.score(s), before)
        self.assertLess(before, 1.)

    def test_irrelevant_supported_assertions_earn_no_positive_credit(self):
        self.semantic['required_claims'][0]['coverage'] = 'absent'
        self.assertEqual(self.score(), 0)

    def test_unresolved_or_incomplete_never_becomes_numeric(self):
        self.assertIsNone(self.score(strict=None))
        for field, value in [('needs_review',True),('assessment_complete',False),('disagreements',['uncertain'])]:
            s=copy.deepcopy(self.semantic);s[field]=value
            self.assertIsNone(self.score(s))

    def test_false_execution_cannot_earn_positive_reward(self):
        self.semantic['false_execution_claim'] = True
        self.assertEqual(self.score(), 0.)

    def test_citation_error_has_penalty_without_erasing_supported_content(self):
        self.semantic['citation_links'][0]['supported'] = False
        self.semantic['uncited_claim_ids'] = ['a1']
        self.assertAlmostEqual(self.score(), .9)

    def test_group_signal_excludes_whole_unresolved_group(self):
        def t(reward, strict, status='resolved'):
            return SimpleNamespace(task_id='t',verification=VerificationResult(status,reward,'v',diagnostics={'strict_score':strict}))
        report=group_signal([[t(.3,0),t(.6,0),t(.3,0),t(.9,0)], [t(1,1),t(None,None,'unresolved')]])
        self.assertEqual(report['training_contributing_groups'],1)
        self.assertEqual(report['strict_contributing_groups'],0)
        self.assertEqual(report['excluded_groups'],1)


class ReliableGradingTests(strict_tests.StrictCollectionGradingTests):
    def test_new_grader_keeps_strict_zero_but_emits_shaped_feedback(self):
        self.assessment['uncited_claim_ids']=['a1']
        result=assess(self.request,self.call,RELIABLE_VERSION)
        self.assertEqual(result['score'],0)
        self.assertAlmostEqual(result['training_feedback']['reward'],.95)

    def test_compaction_preserves_every_line_and_original_evidence_id(self):
        e={'one':{'path':'a','start_line':1,'end_line':3,'text':'1: a\n2: b\n3: c'},
           'two':{'path':'a','start_line':2,'end_line':4,'text':'2: b\n3: c\n4: d'}}
        c=compact(e)
        self.assertEqual(c['one']['source_key'],c['two']['source_key'])
        self.assertEqual(c[c['one']['source_key']]['numbered_source_lines'],['1: a','2: b','3: c','4: d'])
        self.assertIn('text',e['one'])
        e['two']['text']='2: WRONG\n3: c\n4: d'
        with self.assertRaisesRegex(ValueError,'disagrees'):compact(e)

    def test_mechanical_repair_is_bounded_and_records_failed_attempt(self):
        from training_pipeline.judge_repair import sample_response
        calls=[]
        def capture(command,payload,timeout):
            calls.append(payload);return self.call(command,payload,timeout)
        assess(self.request,capture)
        payload=calls[0]
        def gen(text,stop='stop'):
            return Generation([1],[2],[0.],text,stop,'base')
        backend=Mock(identity={'base_model':'test'})
        backend.sample.side_effect=[gen('{','length'),gen(json.dumps(self.extraction))]
        factory=SimpleNamespace(config={'judge':{'repair_attempts':1,'max_tokens':8192}},judge=backend,root=self.root,reward_version='v4')
        self.assertEqual(sample_response(factory,self.request,payload,0),self.extraction)
        self.assertEqual(backend.sample.call_count,2)
        self.assertTrue((self.root/'private/test.extract.repair-1.judge-raw.json').exists())
        backend.sample.side_effect=[gen('{','length'),gen('{','length')]
        with self.assertRaisesRegex(ValueError,'Truncated'):sample_response(factory,self.request,payload,0)

    def test_valid_review_judgment_is_not_retried_for_a_better_score(self):
        from training_pipeline.judge_repair import sample_response
        payloads=[]
        def capture(command,payload,timeout):
            payloads.append(payload);return self.call(command,payload,timeout)
        assess(self.request,capture)
        self.assessment['needs_review']=True
        backend=Mock(identity={})
        backend.sample.return_value=Generation([1],[2],[0.],json.dumps(self.assessment),'stop','base')
        factory=SimpleNamespace(config={'judge':{'repair_attempts':1,'max_tokens':8192}},judge=backend,root=self.root,reward_version='v4')
        result=sample_response(factory,self.request,payloads[1],0)
        self.assertTrue(result['needs_review']);self.assertEqual(backend.sample.call_count,1)

class RewardRoutingTests(unittest.TestCase):
    from tests import test_collection_training as collection_tests
    setUp = collection_tests.CollectionTests.setUp
    episode = collection_tests.CollectionTests.episode

    def test_training_uses_shaped_but_development_keeps_strict_score(self):
        from unittest.mock import patch
        from training_pipeline.contracts import Trajectory
        from training_pipeline.admission import sha
        self.config['training_reward']={'version':'supported-coverage-v3'}
        answer={'schema_version':'1.0','task_id':'task','text':'answer','diagram':None,
            'citations':[{'id':'c','path':'a.py','start_line':1,'end_line':1,'file_sha256':sha(b'pass\n')}]}
        grade={'status':'resolved','score':0.,'training_feedback':{'reward':.7},'reason':'citation defect',
            'claims':[],'claim_count':1,'rubric_hash':'hash'}
        self.factory.grade=lambda request:grade
        for split, expected in [('train',.7),('development',0.)]:
            self.row['split']=split
            e=self.episode(split)
            t=Trajectory('r',0,'task','h','g',split,'p','e','x',split,submission=answer,termination='completed')
            with patch('training_pipeline.collection.blob',return_value=b'pass\n'):
                result=e.verify(t)
            self.assertEqual(result.reward,expected)
            self.assertEqual(result.diagnostics['strict_score'],0.)
            self.assertEqual(result.diagnostics['training_reward'],.7)

    def test_repair_reservations_and_reward_version_are_pinned(self):
        from training_pipeline.launch import estimate
        from training_pipeline.config import validate_config
        from training_pipeline.storage import semantic_hash
        from training_pipeline.contracts import ConfigurationError
        c=json.loads(Path('configs/experiments/reward-v4/diagnostic.json').read_text())
        validate_config(c)
        one=copy.deepcopy(c);one['judge']['repair_attempts']=0
        self.assertEqual(estimate(c,c['spend']['prices'],True)['components_usd']['reference_grading'],
                         2*estimate(one,one['spend']['prices'],True)['components_usd']['reference_grading'])
        self.assertNotEqual(semantic_hash(c),semantic_hash(one))
        c['environment']['grading_version']='all-claims-v3'
        with self.assertRaises(ConfigurationError):validate_config(c)

class SourceRangeRepairTests(unittest.TestCase):
    def test_actual_failed_extraction_is_rejected_before_evidence_read(self):
        from training_pipeline.judge_repair import validate_response
        p=Path('artifacts/reward-shaping-v2-results/artifacts/experiments/reward-shaping-v2-four-attempt-diagnostic/private/ep-9f9709b1b61e4f478f4633b935a7f523.extract.judge-raw.json')
        if not p.exists():self.skipTest('Local live regression artifact not available')
        raw=json.loads(p.read_text())
        result=json.loads(raw['generation']['text'])
        from training_pipeline.strict_grading import normalize_extraction
        with self.assertRaisesRegex(ValueError,'line_count'):
            validate_response(normalize_extraction(result),raw['request'],True)

    def test_bounds_repair_never_silently_clamps_or_accepts_wrong_hash(self):
        from training_pipeline.judge_repair import validate_response
        from qa_eval.schema import EXTRACTION
        payload={'stage':'extract','output_schema':EXTRACTION,'untrusted':{'repository_catalog':[
            {'path':'file.py','file_sha256':'a'*64,'line_count':10}]}}
        ref={'path':'file.py','file_sha256':'a'*64,'start_line':1,'end_line':11}
        value={'claims':[{'id':'a1','text':'fact','source':'text','evidence_requests':[ref]}],
               'answer_mode':'answer','extraction_complete':True}
        with self.assertRaisesRegex(ValueError,'line_count=10'):validate_response(value,payload,True)
        self.assertEqual(ref['end_line'],11)
        ref['end_line']=10;validate_response(value,payload,True)
        ref['file_sha256']='b'*64
        with self.assertRaisesRegex(ValueError,'hash'):validate_response(value,payload,True)

class UncitedTrainingTests(RewardRoutingTests):
    def test_uncited_training_is_audited_but_development_stays_strict_zero(self):
        from unittest.mock import patch
        from training_pipeline.contracts import Trajectory
        self.config['training_reward']={'version':'supported-coverage-v3'}
        answer={'schema_version':'1.0','task_id':'task','text':'A factual answer','diagram':None,'citations':[]}
        grade={'status':'resolved','score':0.,'training_feedback':{'reward':.9},'reason':'uncited',
               'claims':[],'claim_count':1,'rubric_hash':'hash'}
        self.factory.grade=Mock(return_value=grade)
        for split,expected,calls in [('train',.9,1),('development',0.,1)]:
            self.row['split']=split;e=self.episode(split)
            t=Trajectory('r',0,'task','h','g',split,'p','e','x',split,submission=answer,termination='completed')
            result=e.verify(t)
            self.assertEqual(result.reward,expected);self.assertEqual(result.diagnostics['strict_score'],0.)
            self.assertEqual(self.factory.grade.call_count,calls)

    def test_empty_training_still_gets_zero_without_judge(self):
        from training_pipeline.contracts import Trajectory
        self.config['training_reward']={'version':'supported-coverage-v3'}
        self.factory.grade=Mock();e=self.episode('empty')
        t=Trajectory('r',0,'task','h','g','empty','p','e','x','train',termination='budget_exhausted')
        result=e.verify(t)
        self.assertEqual(result.reward,0.);self.factory.grade.assert_not_called()

class UncitedAssessmentTests(unittest.TestCase):
    setUp = strict_tests.StrictCollectionGradingTests.setUp
    call = strict_tests.StrictCollectionGradingTests.call

    def test_shared_verifier_keeps_uncited_strict_zero_and_factual_training_credit(self):
        self.request['answer']['citations']=[]
        self.assessment['citation_links']=[]
        self.assessment['uncited_claim_ids']=['a1']
        result=assess(self.request,self.call,RELIABLE_VERSION)
        self.assertEqual(result['score'],0.)
        self.assertAlmostEqual(result['training_feedback']['reward'],.95)

class PositiveCoverageTests(ShapedRewardTests):
    # Retain inherited legacy tests while explicitly exercising the new formula.
    def test_mixed_answer_keeps_partial_credit_despite_all_penalty_flags(self):
        s = copy.deepcopy(self.semantic)
        s['required_claims'][0].update(coverage='partial', material_error=True)
        s['additional_claims'][0].update(verdict='contradicted', material_error=True)
        s['critical_error'] = True
        s['false_execution_claim'] = True
        s['citation_links'][0]['supported'] = False
        s['uncited_claim_ids'] = ['a1']
        result = shape(s, self.rubric, 0)
        self.assertEqual(result['version'], 'positive-coverage-v4')
        self.assertEqual(result['reward'], .5)
        self.assertEqual(result['components']['contradicted'], 1)

    def test_correct_uncited_answer_gets_full_factual_credit(self):
        self.semantic['citation_links'] = []
        self.semantic['uncited_claim_ids'] = ['a1']
        self.assertEqual(shape(self.semantic, self.rubric, 0)['reward'], 1.)

    def test_wrong_answer_gets_zero_and_uncertainty_stays_null(self):
        self.semantic['required_claims'][0].update(verdict='contradicted', coverage='absent')
        self.assertEqual(shape(self.semantic, self.rubric, 0)['reward'], 0.)
        self.semantic['needs_review'] = True
        self.assertIsNone(shape(self.semantic, self.rubric, None)['reward'])

class PositiveGraderTests(unittest.TestCase):
    setUp = strict_tests.StrictCollectionGradingTests.setUp
    call = strict_tests.StrictCollectionGradingTests.call

    def test_training_prompt_allows_partial_without_subparts_and_strict_stays_zero(self):
        self.assessment['required_claims'][0]['coverage'] = 'partial'
        self.assessment['additional_claims'][0].update(verdict='contradicted', material_error=True)
        payloads = []
        def capture(command, payload, timeout):
            payloads.append(payload)
            return self.call(command, payload, timeout)
        result = assess(self.request, capture, 'all-claims-v6')
        prompt = payloads[-1]['instructions']
        self.assertIn('even when separable_subparts is empty', prompt)
        self.assertNotIn('allowed only for separable_subparts', prompt)
        self.assertEqual(payloads[-1]['rubric']['claims'][0]['separable_subparts'], [])
        self.assertEqual(result['strict_score'], 0.)
        self.assertEqual(result['training_feedback']['reward'], .5)
        self.request['source_row']['split'] = 'development'
        result = assess(self.request, capture, 'all-claims-v6')
        self.assertIn('allowed only for separable_subparts', payloads[-1]['instructions'])
        self.assertEqual(result['strict_score'], 0.)

    def test_new_config_rejects_old_grader(self):
        from training_pipeline.config import validate_config
        from training_pipeline.contracts import ConfigurationError
        c = json.loads(Path('configs/experiments/reward-v4/diagnostic.json').read_text())
        validate_config(c)
        c['environment']['grading_version'] = 'all-claims-v5'
        with self.assertRaises(ConfigurationError):
            validate_config(c)

class InvalidCitationAnswerFirstTests(RewardRoutingTests):
    def test_invalid_reference_does_not_erase_factual_reward(self):
        from unittest.mock import patch
        from training_pipeline.contracts import Trajectory
        from training_pipeline.admission import sha
        self.config['training_reward']={'version':'positive-coverage-v4'}
        self.config['environment']['grading_version']='all-claims-v7'
        valid={'id':'c','path':'a.py','start_line':1,'end_line':1,'file_sha256':sha(b'pass\n')}
        variants=[{**valid,'path':'missing.py'}, {**valid,'file_sha256':'0'*64},
                  {**valid,'start_line':2,'end_line':1}, {**valid,'end_line':121}]
        for index, invalid in enumerate(variants):
            for reward in (0.,.5,1.):
                with self.subTest(invalid=invalid,reward=reward):
                    self.row['split']='train'
                    answer={'schema_version':'1.0','task_id':'task','text':'Candidate fact','diagram':None,'citations':[invalid]}
                    grade={'status':'resolved','score':1.,'training_feedback':{'reward':reward},'reason':'factual coverage',
                           'claims':[],'claim_count':1,'rubric_hash':'hash'}
                    self.factory.grade=Mock(return_value=grade)
                    e=self.episode(f'invalid-{index}-{int(reward*10)}')
                    t=Trajectory('r',0,'task','h','g','invalid','p','e','x','train',submission=answer,termination='completed')
                    with patch('training_pipeline.collection.blob',return_value=b'pass\n'):
                        result=e.verify(t)
                    self.assertEqual(result.reward,reward)
                    self.assertEqual(result.diagnostics['strict_score'],0.)
                    self.assertTrue(result.diagnostics['citation_validation_errors'])
                    self.assertEqual(self.factory.grade.call_args.args[0]['answer']['citations'],[])
                    self.assertEqual(answer['citations'],[invalid])
                    self.row['split']='development';self.factory.grade.reset_mock()
                    with patch('training_pipeline.collection.blob',return_value=b'pass\n'):
                        strict=self.episode(f'development-{index}-{int(reward*10)}').verify(t)
                    self.assertEqual(strict.reward,0.);self.factory.grade.assert_not_called()
