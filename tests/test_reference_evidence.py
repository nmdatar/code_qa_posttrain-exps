import unittest
from unittest.mock import patch
from training_pipeline.reference_evidence import cited_ranges,enrich
from training_pipeline.admission import sha

class EvidenceTests(unittest.TestCase):
    def test_parses_disjoint_ranges(self):
        self.assertEqual(list(cited_ranges('See pkg/a.py:3-8,12-15 and b.py:20.')),[('pkg/a.py',3,8),('pkg/a.py',12,15),('b.py',20,20)])
    def test_hash_bound_ranges_and_rejected_bad_references(self):
        content=b'x\n'*1000;h=sha(content)
        row={'snapshot_root':'root','public':{'repository':{'commit':'pin'}},'image_result':{'snapshot_files':{'a.py':h}}}
        ref={'reference_answer':'a.py:1-900 a.py:1-900 missing.py:2 a.py:1001','verified_evidence':[]}
        with patch('training_pipeline.reference_evidence.blob',return_value=content):new,rejected=enrich(ref,row)
        self.assertEqual([(r['start_line'],r['end_line']) for r in new['verified_evidence']],[(1,600),(601,900)])
        self.assertEqual(len(rejected),2);self.assertEqual(ref['verified_evidence'],[])
        self.assertTrue(all(r['file_sha256']==h for r in new['verified_evidence']))
        row['image_result']['snapshot_files']['a.py']='wrong'
        with patch('training_pipeline.reference_evidence.blob',return_value=content),self.assertRaises(ValueError):enrich(ref,row)
