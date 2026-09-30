import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from dataset_builder.collection_review import validate_review

class ReviewBindingTests(unittest.TestCase):
    def setUp(self):
        self.ref={'id':'task','repository':{'commit':'a'*40}}
        self.quality={'reference_sha256':'reference','question_sha256':'question'}
        self.review={'task_id':'task',**self.quality,'status':'supported','material_errors':[],
            'checked_claims':[{'claim':'return is 2','verdict':'supported','reason':'literal return','evidence':{'path':'a.py','start_line':1,'end_line':2}}]}
        self.data=b'def f():\n    return 2\n';self.files={'a.py':hashlib.sha256(self.data).hexdigest()}
    def validate(self):
        with patch('dataset_builder.collection_review.read_git_file',return_value=self.data):
            return validate_review(self.review,self.ref,self.quality,Path('/source'),self.files)
    def test_binds_exact_source(self):
        self.assertEqual(self.validate()['verified_evidence'][0]['file_sha256'],self.files['a.py'])
    def test_rejects_stale_reference(self):
        self.review['reference_sha256']='old'
        with self.assertRaisesRegex(ValueError,'Stale'):self.validate()
    def test_rejects_unavailable_source(self):
        self.review['checked_claims'][0]['evidence']['path']='submodule/missing.py'
        with self.assertRaisesRegex(ValueError,'not in solver snapshot'):self.validate()
    def test_rejects_wrong_source_bytes(self):
        self.files['a.py']='wrong'
        with self.assertRaisesRegex(ValueError,'hash mismatch'):self.validate()
    def test_rejects_out_of_bounds_span(self):
        self.review['checked_claims'][0]['evidence']['end_line']=3
        with self.assertRaisesRegex(ValueError,'outside file'):self.validate()
    def test_cannot_accept_material_error(self):
        self.review['material_errors']=['Wrong return value']
        with self.assertRaisesRegex(ValueError,'material errors'):self.validate()

class CorrectionIndependenceTests(unittest.TestCase):
    def setUp(self):
        import json
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        self.folder=self.root/'reports/task-generation-1000'
        self.row={'task_id':'task','reference_sha256':'corrected','question_sha256':'question','original_reference_sha256':'original','status':'supported'}
        self.queue={'author_worker':0,'reviewer_worker':1,'tasks':[self.row]}
        self.report={'reviewer_worker':1,'human_reviewed':False,'tasks':[self.row]}
    def run_check(self):
        from dataset_builder.build import write_json
        from dataset_builder.collection_review import correction_records
        write_json(self.folder/'correction-queue/worker-1/batch-000.json',self.queue)
        write_json(self.folder/'correction-review/worker-1/batch-000.json',self.report)
        return correction_records(self.root)
    def test_independent_review_keeps_lineage(self):
        self.assertEqual(self.run_check()['task']['original_reference_sha256'],'original')
    def test_same_author_rejected(self):
        self.queue['author_worker']=1
        with self.assertRaisesRegex(ValueError,'author'):self.run_check()
    def test_stale_corrected_text_rejected(self):
        self.report['tasks']=[{**self.row,'reference_sha256':'different'}]
        with self.assertRaisesRegex(ValueError,'hash mismatch'):self.run_check()
    def test_missing_task_rejected(self):
        self.report['tasks']=[]
        with self.assertRaisesRegex(ValueError,'missing tasks'):self.run_check()
