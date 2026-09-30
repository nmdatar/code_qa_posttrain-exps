"""Frozen benchmark bindings, blind judge input, and persisted honest scores."""
import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from fastapi.testclient import TestClient
from product_api.app import create_app, RunManager
from product_api.benchmarks import freeze, public
from product_api.grading import grade_answer, main, JUDGE_MODEL
from product_api.worker import atomic_write
from tests import test_product as fixtures


class ProductGradingTests(unittest.TestCase):
    setUp = fixtures.ProductTests.setUp
    tearDown = fixtures.ProductTests.tearDown

    def benchmark(self):
        ref={'path':'example.py','start_line':1,'end_line':3,'file_sha256':self.manifest['snapshot_files']['example.py']}
        return {'id':'known','repo_id':'fixture','question':'What is the value?', 'claim_count':2,
            'split':'development','reference_status':'draft','human_reviewed':False,
            'task':{'permitted_tools':['read_file']},
            'reference':{'record':{'claims':[
                {'id':'value','text':'value is 1','weight':3,'evidence':[ref]},
                {'id':'frozen','text':'frozen is True','weight':1,'evidence':[ref]}]}}}

    def fake_judge(self):
        class Judge:
            identity = {'base_model':JUDGE_MODEL}
            def __init__(self, config=None): self.payloads=[]
            def sample(self,messages,*args):
                payload=json.loads(messages[1]['content']); self.payloads.append(payload)
                verdicts={'claims':[
                    {'id':'value','verdict':'supported','answer_span_ids':['a1'],'evidence_ids':['e1'],'reason':'The answer states the value.'},
                    {'id':'frozen','verdict':'missing','answer_span_ids':[],'evidence_ids':[],'reason':'Frozen status is omitted.'}]}
                return SimpleNamespace(text=json.dumps(verdicts),stop_reason='stop',prompt=[1,2],tokens=[3])
            def close(self): pass
        return Judge

    def test_weighted_score_and_blind_input(self):
        benchmark=freeze(self.benchmark(), self.repo)
        judge=self.fake_judge()()
        grade=grade_answer(judge,benchmark,{'text':'The value is 1.','model':'secret-model-name'})
        self.assertEqual(grade['score'],.75)
        self.assertEqual(grade['claims'][0]['answer_quotes'],['The value is 1.'])
        self.assertNotIn('secret-model-name',json.dumps(judge.payloads))
        self.assertNotIn('reference_answer',public(self.benchmark()))

    def test_source_mismatch_refuses_benchmark(self):
        row=self.benchmark()
        row['reference']['record']['claims'][0]['evidence'][0]['file_sha256']='wrong'
        with self.assertRaises(ValueError):freeze(row,self.repo)

    def test_model_score_or_hallucinated_passage_rejected(self):
        benchmark=freeze(self.benchmark(),self.repo)
        judge=self.fake_judge()()
        judge.sample=lambda *args:SimpleNamespace(text='{"score":1}',stop_reason='stop')
        with self.assertRaises(ValueError):grade_answer(judge,benchmark,{'text':'value is 1'})

    def test_scored_creation_binds_question_and_keeps_gold_out_of_workers(self):
        models=[{'id':'a','kind':'base','base_model':'test/model','ready':True},
                {'id':'b','kind':'base','base_model':JUDGE_MODEL,'ready':True}]
        storage=self.root/'runs'
        with patch('product_api.app.repo_catalog',return_value=[self.repo]), TestClient(create_app(storage)) as client:
            manager=client.app.state.manager
            manager.benchmarks={'known':self.benchmark()}
            body={'repo_id':'fixture','left_model_id':'a','right_model_id':'b','question':'What is the value?','benchmark_id':'known'}
            with patch.object(manager,'catalogs',return_value={'models':models}),patch.object(manager,'launch'):
                self.assertEqual(client.post('/api/comparisons',json={**body,'question':'Different'}).status_code,422)
                self.assertEqual(client.post('/api/comparisons',json={**body,'preview':True}).status_code,422)
                response=client.post('/api/comparisons',json=body)
            self.assertEqual(response.status_code,201,response.text)
            pair=response.json()
            self.assertEqual(pair['grading']['status'],'pending')
            for run in pair['runs']:
                config=json.loads((storage/run['id']/'config.json').read_text())
                self.assertEqual(config['question'],'What is the value?')
                self.assertNotIn('rubric',config)
                self.assertNotIn('reference_answer',config)
                self.assertEqual(config['permitted_tools'],['read_file'])
                atomic_write(storage/run['id']/'result.json',{'status':'completed','answer':{'text':'The value is 1.'},'metrics':{}})
            with patch('product_api.grading.TinkerJudge',self.fake_judge()):main(storage,pair['id'])
            result=client.get('/api/comparisons/'+pair['id']).json()
            self.assertEqual(result['grading']['status'],'completed')
            self.assertEqual([g['score'] for g in result['grading']['scores'].values()],[.75,.75])
            self.assertEqual(len(client.get('/api/comparisons').json()),1)

    def test_missing_answer_never_gets_invented_score(self):
        storage=self.root/'missing';(storage/'comparisons').mkdir(parents=True)
        for i in ('one','two'):
            (storage/i).mkdir()
            atomic_write(storage/i/'result.json',{'answer':None})
        benchmark=freeze(self.benchmark(),self.repo)
        atomic_write(storage/'comparisons/test.json',{'id':'test','run_ids':['one','two'],'benchmark':benchmark,'judge':{'base_model':JUDGE_MODEL}})
        with patch('product_api.grading.TinkerJudge') as factory:main(storage,'test')
        factory.assert_not_called()
        result=json.loads((storage/'comparisons/test.grade.json').read_text())
        self.assertTrue(all(g['score'] is None and g['status']=='not_scored' for g in result['scores'].values()))


    def test_grader_waits_for_both_answers_and_launches_once(self):
        storage=self.root/'supervisor'
        with patch('product_api.app.repo_catalog',return_value=[self.repo]):
            manager=RunManager(storage)
        try:
            record={'id':'pair','run_ids':['one','two'],'benchmark':{'id':'known'},'judge':{'base_model':JUDGE_MODEL}}
            atomic_write(storage/'comparisons/pair.json',record)
            for ident in record['run_ids']:(storage/ident).mkdir()
            atomic_write(storage/'one/result.json',{'answer':{'text':'one'}})
            with patch('product_api.app.subprocess.Popen') as launch, patch('product_api.app.threading.Thread'):
                manager.maybe_grade('pair')
                launch.assert_not_called()
                atomic_write(storage/'two/result.json',{'answer':{'text':'two'}})
                manager.maybe_grade('pair')
                manager.maybe_grade('pair')
                launch.assert_called_once()
        finally:
            manager.grading_processes.clear()
            manager.owner_lock.close()

    def test_restart_preserves_partial_grades_without_resubmission(self):
        storage=self.root/'restart'; (storage/'comparisons').mkdir(parents=True)
        atomic_write(storage/'comparisons/pair.json',{'id':'pair','benchmark':{'id':'known'}})
        scores={'one':{'status':'resolved','score':.75}}
        atomic_write(storage/'comparisons/pair.grade.json',{'status':'running','scores':scores})
        with patch('product_api.app.repo_catalog',return_value=[self.repo]), patch('product_api.app.subprocess.Popen') as launch:
            manager=RunManager(storage)
            try:
                grade=json.loads((storage/'comparisons/pair.grade.json').read_text())
                self.assertEqual(grade['status'],'interrupted')
                self.assertEqual(grade['scores'],scores)
                launch.assert_not_called()
            finally:manager.owner_lock.close()
