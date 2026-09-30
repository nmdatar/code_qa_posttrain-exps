import copy
import unittest
from training_pipeline.atomic_claim_experiment import validate_split, apply_split

class AtomicClaimsTests(unittest.TestCase):
    def test_rejects_missing_parents_and_invented_quote(self):
        claims=[{'id':'0','text':'A and B'}]
        with self.assertRaises(ValueError):validate_split({'claims':[]},claims)
        with self.assertRaises(ValueError):validate_split({'claims':[{'id':'0','facts':[{'text':'C','quote':'C'}]}]},claims)

    def test_imported_evidence_and_qualification_preserved(self):
        original={'task_id':'x','reviewed_claims':[{'claim':'A and B','verdict':'qualified','reason':'Only when C',
            'evidence':{'path':'a.py','start_line':1,'end_line':3}}],'reference_answer':'reference'}
        saved=copy.deepcopy(original)
        result=apply_split(original,{'0':[{'text':'A','quote':'A'},{'text':'B','quote':'B'}]})
        self.assertEqual(original,saved)
        self.assertEqual([c['claim'] for c in result['reviewed_claims']],['A','B'])
        for c in result['reviewed_claims']:
            self.assertEqual(c['evidence'],original['reviewed_claims'][0]['evidence'])
            self.assertEqual(c['reason'],'Only when C')

    def test_native_parent_weight_conserved(self):
        r={'record':{'claims':[{'id':'one','text':'A and B','weight':3,'evidence':[],'separable_subparts':[]}]}}
        x=apply_split(r,{'one':[{'text':'A','quote':'A'},{'text':'B','quote':'B'}]})
        self.assertEqual(sum(c['weight'] for c in x['record']['claims']),3)
        self.assertEqual(len({c['id'] for c in x['record']['claims']}),2)
