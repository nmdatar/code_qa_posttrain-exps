import hashlib
import unittest
from dataset_builder.collection_release import grading_records

class GradingExportTests(unittest.TestCase):
    def test_unresolved_upstream_answer_excluded_and_correction_selected(self):
        refs=[{'id':'bad','reference_kind':'upstream_benchmark_draft','reference_answer':'wrong'},
              {'id':'fixed','reference_kind':'upstream_benchmark_draft','reference_answer':'wrong original',
               'selected_reference_answer':'corrected','selected_reference_sha256':hashlib.sha256(b'corrected').hexdigest(),
               'correction_status':'supported','grading':{'enabled':True}},
              {'id':'local','reference_kind':'local_claims_draft','record':{'claims':['verified local claim']}}]
        corrected={'checked_claims':[{'claim':'corrected claim'}],'verified_evidence':[{'path':'a.py','start_line':1,'end_line':1}]}
        result=grading_records(refs,{}, {'fixed':corrected})
        self.assertEqual({r['task_id'] for r in result},{'fixed','local'})
        self.assertEqual(result[0]['reference_answer'],'corrected')
        self.assertNotIn('wrong original',str(result))
        self.assertEqual(result[0]['reviewed_claims'],corrected['checked_claims'])
    def test_original_supported_uses_original_review_evidence(self):
        ref={'id':'good','reference_kind':'upstream_benchmark_draft','selected_reference_answer':'answer','selected_reference_sha256':'hash','grading':{}}
        result=grading_records([ref],{'good':{'checked_claims':['claim'],'verified_evidence':['span']}},{})
        self.assertEqual(result[0]['verified_evidence'],['span'])
