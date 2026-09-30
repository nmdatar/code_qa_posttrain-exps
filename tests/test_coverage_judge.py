import unittest
from training_pipeline.coverage_judge import merge_coverage

class IndependentCoverageTests(unittest.TestCase):
    def test_uncited_mentioned_helper_gets_pinned_source(self):
        from unittest.mock import patch
        from training_pipeline.coverage_judge import mentioned_function_evidence
        from training_pipeline.admission import sha
        text=b'def track_value(value):\n    observed.append(value)\n'
        request={'answer':{'text':'track_value records observed values.'},'rubric':{'evidence':[]},
                 'observed_files':{'tracing.py':sha(text)},'source_row':{'snapshot_root':'root',
                 'public':{'repository':{'commit':'pin'}},'image_result':{'snapshot_files':{'tracing.py':sha(text)}}}}
        with patch('training_pipeline.coverage_judge.blob',return_value=text):refs=mentioned_function_evidence(request)
        self.assertEqual(refs,[{'path':'tracing.py','start_line':1,'end_line':2,'file_sha256':sha(text)}])
    def test_factory_routes_training_coverage_separately_and_keeps_evaluation_strict(self):
        import json,tempfile
        from pathlib import Path
        from types import SimpleNamespace
        from unittest.mock import patch
        from training_pipeline.collection import CollectionFactory
        config=json.loads(Path('tests/fixtures/experiment_contracts/reward-diagnostic.json').read_text())
        config['environment']['grading_version']='all-claims-v7'
        request={'episode_id':'case','source_row':{'split':'train'},'rubric':{'claims':[{'id':'r1'}],'rubric_hash':'hash'}}
        coverage={'status':'resolved','score':.5,'reason':'one fact'}
        with tempfile.TemporaryDirectory() as tmp:
            factory=CollectionFactory(config,Path(tmp),None,judge=SimpleNamespace())
            with patch('training_pipeline.strict_grading.assess',side_effect=ValueError('strict format failure')),patch('training_pipeline.coverage_judge.assess_coverage',return_value=coverage) as call:
                result=factory.grade(request)
                self.assertEqual(result['training_feedback']['reward'],.5)
                self.assertEqual(result['claim_count'],1)
                self.assertEqual(result['rubric_hash'],'hash')
                request['source_row']['split']='development'
                with self.assertRaises(ValueError):factory.grade(request)
                self.assertEqual(call.call_count,1)
    def test_verified_candidate_source_reaches_reward_without_citation_requirements(self):
        from unittest.mock import patch
        from training_pipeline.coverage_judge import coverage_request
        from training_pipeline.admission import sha
        source=b'helper collects values\n'
        request={'answer':{'text':'helper collects values'},'rubric':{'version':'v','claims':[],'evidence':[]},
                 'answer_evidence':[{'path':'helper.py','start_line':1,'end_line':1,'file_sha256':sha(source)}],
                 'source_row':{'snapshot_root':'root','public':{'repository':{'commit':'pin'}},
                               'image_result':{'snapshot_files':{'helper.py':sha(source)}}}}
        with patch('training_pipeline.coverage_judge.blob',return_value=source):
            result=coverage_request(request)
        self.assertTrue(any(e['text']=='helper collects values' for e in result['rubric']['evidence']))
        self.assertEqual(request['rubric']['evidence'],[])
    def test_group_signal_preserves_unknown_strict_scores(self):
        from types import SimpleNamespace as NS
        from training_pipeline.shaped_reward import group_signal
        group=[NS(task_id='t',verification=NS(status='resolved',reward=r,diagnostics={'strict_score':None})) for r in [0,.5]]
        result=group_signal([group])['groups'][0]
        self.assertTrue(result['training_contributes']);self.assertIsNone(result['strict_variance'])
        self.assertEqual(result['strict_scores'],[None,None])
    def test_strict_failure_does_not_erase_verified_factual_coverage(self):
        strict={'status':'unresolved','strict_score':None,'score':None,'reason':'invalid citation id'}
        coverage={'status':'resolved','score':.5,'reason':'one of two facts','claims':[]}
        r=merge_coverage(strict,coverage)
        self.assertEqual(r['status'],'resolved');self.assertIsNone(r['strict_score'])
        self.assertEqual(r['strict_status'],'unresolved');self.assertEqual(r['training_feedback']['reward'],.5)
        self.assertEqual(strict['reason'],'invalid citation id')
    def test_uncertain_reference_never_becomes_zero_reward(self):
        r=merge_coverage({'status':'resolved','strict_score':0,'reason':'strict failure'},
                         {'status':'unresolved','score':None,'reason':'source gap'})
        self.assertIsNone(r['training_feedback']['reward']);self.assertFalse(r['training_feedback']['eligible'])
    def test_extra_errors_do_not_change_reward_or_strict_result(self):
        r=merge_coverage({'status':'resolved','strict_score':0,'score':0,'reason':'false extra',
                         'training_feedback':{'components':{'contradicted':1}}},
                        {'status':'resolved','score':.5,'reason':'one correct fact'})
        self.assertEqual(r['strict_score'],0);self.assertEqual(r['training_feedback']['reward'],.5)
        self.assertEqual(r['training_feedback']['components']['contradicted'],1)
