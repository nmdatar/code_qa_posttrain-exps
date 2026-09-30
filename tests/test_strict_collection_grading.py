import copy
from pathlib import Path
import tempfile
import unittest

from qa_eval.demo import fixture
from qa_eval.schema import ASSESSMENT
from training_pipeline.strict_grading import assess
from training_pipeline.claim_grading import reference_rubric


class StrictCollectionGradingTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        task,answer,config,metrics,semantic=fixture(self.root)
        ref=task['claims'][0]['evidence'][0]
        content=(self.root/ref['path']).read_text().splitlines()
        evidence={**ref,'text':'\n'.join(content[ref['start_line']-1:ref['end_line']])}
        reference={'reviewed_claims':[{'claim':task['claims'][0]['text'],'verdict':'supported','evidence':ref}]}
        self.request={'episode_id':'test','question':task['question'],'answer':answer,
            'rubric':reference_rubric(reference,[evidence]),'answer_evidence':[evidence],
            'source_row':{'id':task['id'],'split':'train','public':{'repository':task['repository']},
                'image_result':{'snapshot_files':{ref['path']:ref['file_sha256']}},'snapshot_root':str(self.root)}}
        self.extraction={'claims':[{'id':'a1','text':'Sets cancelled to True.','source':'text','evidence_requests':[]}],
                         'answer_mode':'answer','extraction_complete':True}
        self.assessment={k:copy.deepcopy(semantic[k]) for k in ASSESSMENT['properties']}
        self.calls=[]

    def call(self,command,payload,timeout):
        self.calls.append(payload['stage'])
        return copy.deepcopy(self.extraction if payload['stage']=='extract' else self.assessment)

    def test_uses_shared_two_stage_verifier(self):
        result=assess(self.request,self.call)
        self.assertEqual(self.calls,['extract','assess'])
        self.assertEqual(result['score'],1)
        self.assertFalse(result['human_reviewed'])
        self.assertFalse(result['independent_evaluation'])

    def test_extra_falsehood_cannot_be_diluted_by_correct_required_claim(self):
        self.assessment['additional_claims'][0].update(verdict='contradicted',material_error=True)
        self.assertEqual(assess(self.request,self.call)['score'],0)

    def test_uncited_or_unsupported_extra_assertion_blocks_credit(self):
        self.assessment['uncited_claim_ids']=['a1']
        self.assertEqual(assess(self.request,self.call)['score'],0)
        self.assessment['uncited_claim_ids']=[]
        self.assessment['additional_claims'][0]['verdict']='insufficient'
        self.assertEqual(assess(self.request,self.call)['score'],0)

    def test_omitted_extracted_assertion_is_rejected(self):
        self.extraction['claims'].append({'id':'a2','text':'Unrelated extra fact.','source':'text','evidence_requests':[]})
        with self.assertRaisesRegex(ValueError,'omitted extracted'):
            assess(self.request,self.call)

    def test_incomplete_extraction_or_review_never_becomes_zero_reward(self):
        self.extraction['extraction_complete']=False
        with self.assertRaisesRegex(ValueError,'Incomplete claim extraction'):
            assess(self.request,self.call)
        self.extraction['extraction_complete']=True
        self.assessment['needs_review']=True
        result=assess(self.request,self.call)
        self.assertEqual(result['status'],'unresolved');self.assertIsNone(result['score'])

    def test_invented_evidence_and_false_execution_rejected(self):
        self.assessment['additional_claims'][0]['evidence_keys']=['unknown']
        with self.assertRaisesRegex(ValueError,'nonexistent evidence'):
            assess(self.request,self.call)
        self.assessment['additional_claims'][0]['evidence_keys']=self.assessment['required_claims'][0]['evidence_keys']
        self.assessment['false_execution_claim']=True
        self.assertEqual(assess(self.request,self.call)['score'],0)

    def test_requested_source_hash_must_match_inventory(self):
        ref=copy.deepcopy(self.request['answer_evidence'][0]);ref.pop('text');ref.pop('symbol',None);ref['file_sha256']='0'*64
        self.extraction['claims'][0]['evidence_requests']=[ref]
        with self.assertRaisesRegex(ValueError,'pinned inventory'):
            assess(self.request,self.call)

    def test_empty_optional_symbol_is_normalized_without_changing_evidence(self):
        from training_pipeline.strict_grading import normalize_extraction
        for symbol in ('',None):
            value={'claims':[{'evidence_requests':[{'path':'a.py','symbol':symbol,'start_line':1}]}]}
            result=normalize_extraction(value)
            self.assertNotIn('symbol',result['claims'][0]['evidence_requests'][0])
            self.assertIn('symbol',value['claims'][0]['evidence_requests'][0])
            self.assertEqual(result['claims'][0]['evidence_requests'][0]['start_line'],1)
        value={'claims':[{'evidence_requests':[{'symbol':'real_symbol'}]}]}
        self.assertEqual(normalize_extraction(value),value)
