import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from training_pipeline.claim_grading import reference_rubric, aggregate, request_payload, VERSION
from training_pipeline.collection import CollectionFactory
from training_pipeline.contracts import Generation
from training_pipeline.storage import read, digest
from training_pipeline.config import resolve_run_config


def request_fixture():
    ref = {'reviewed_claims': [
        {'claim': 'It passes.', 'verdict': 'supported', 'evidence': {'path': 'a.py', 'start_line': 1, 'end_line': 1}},
        {'claim': 'It returns a value.', 'verdict': 'supported', 'evidence': {'path': 'a.py', 'start_line': 1, 'end_line': 1}}]}
    excerpts = [{'path': 'a.py', 'start_line': 1, 'end_line': 1, 'file_sha256': 'a'*64, 'text': 'pass'}]
    return {'episode_id': 'test', 'question': 'What does it do?', 'answer': {'text': 'It passes.'},
            'rubric': reference_rubric(ref, excerpts), 'answer_evidence': excerpts}


def judgments(verdicts=('supported', 'partial')):
    return {'claims': [{'id': 'c'+str(i+1), 'verdict': v,
        'answer_span_ids': [] if v in ('missing', 'unresolved') else ['a1'],
        'evidence_ids': ['e1'], 'reason': 'source comparison'} for i, v in enumerate(verdicts)]}


class ClaimGradingTests(unittest.TestCase):
    def test_weighted_aggregation_and_order_independence(self):
        request = request_fixture()
        request['rubric']['claims'][0]['weight'] = 3
        r = request['rubric']
        r['rubric_hash'] = digest({k:r[k] for k in ('version','claims','evidence')})
        result = aggregate(json.dumps(judgments()), request)
        self.assertEqual(result['score'], .875)
        self.assertEqual(result['claim_count'], 2)
        j = judgments(); j['claims'].reverse()
        self.assertEqual(result, aggregate(json.dumps(j), request))

    def test_missing_and_wrong_are_zero_not_unresolved(self):
        for verdict in ('missing', 'contradicted'):
            result = aggregate(json.dumps(judgments((verdict, verdict))), request_fixture())
            self.assertEqual((result['status'], result['score']), ('resolved', 0))
        result = aggregate(json.dumps(judgments(('supported','unresolved'))), request_fixture())
        self.assertEqual(result['status'], 'unresolved')
        self.assertIsNone(result['score'])

    def test_rejects_omitted_duplicate_unknown_claims_and_model_score(self):
        variants=[]
        j=judgments();j['claims'].pop();variants.append(j)
        j=judgments();j['claims'][1]['id']='c1';variants.append(j)
        j=judgments();j['claims'][1]['id']='unknown';variants.append(j)
        j=judgments();j['score']=1;variants.append(j)
        variants.append({'status':'resolved','score':1,'reason':'correct'})
        for j in variants:
            with self.subTest(j=j), self.assertRaises(ValueError):
                aggregate(json.dumps(j), request_fixture())

    def test_reference_content_cannot_be_used_as_candidate_quote(self):
        j=judgments();j['claims'][1]['answer_span_ids']=['e1']
        with self.assertRaisesRegex(ValueError,'candidate answer span'):
            aggregate(json.dumps(j), request_fixture())

    def test_passages_preserve_apostrophes_omissions_and_negation(self):
        request = request_fixture()
        request['answer']['text'] = 'It calls each estimator’s predict method. It does not average. It sums predictions.'
        j = judgments()
        j['claims'][0]['answer_span_ids'] = ['a1', 'a3']
        result = aggregate(json.dumps(j), request)
        self.assertEqual(result['claims'][0]['answer_quotes'],
                         ['It calls each estimator’s predict method.', 'It sums predictions.'])
        spans = request_payload(request)['candidate_answer_spans']
        self.assertEqual(spans[1]['text'], 'It does not average.')
        for span in spans:
            self.assertEqual(span['text'], request['answer']['text'][span['start']:span['end']])
        # Quotes copied or abbreviated by the model are no longer part of the protocol.
        j['claims'][0]['answer_quote'] = "It calls each estimator's predict method... It sums predictions."
        with self.assertRaisesRegex(ValueError, 'schema'):
            aggregate(json.dumps(j), request)

    def test_bad_or_missing_answer_span_references_fail_closed(self):
        for ids in (['a999'], ['a1', 'a1'], ['e1'], [], 'a1', [1]):
            j = judgments(); j['claims'][0]['answer_span_ids'] = ids
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                aggregate(json.dumps(j), request_fixture())
        j = judgments(('missing', 'supported'))
        j['claims'][0]['answer_span_ids'] = ['a1']
        with self.assertRaises(ValueError):
            aggregate(json.dumps(j), request_fixture())

    def test_invalid_evidence_weights_and_duplicate_json_keys(self):
        j=judgments();j['claims'][0]['evidence_ids']=['invented']
        with self.assertRaises(ValueError):aggregate(json.dumps(j), request_fixture())
        for weight in (0, -1, True, float('nan'), float('inf')):
            request=request_fixture();request['rubric']['claims'][0]['weight']=weight
            with self.assertRaises(ValueError):request_payload(request)
        with self.assertRaisesRegex(ValueError,'Duplicate JSON'):
            aggregate('{"claims":[],"claims":[]}', request_fixture())

    def test_missing_prose_rubric_and_unbound_evidence_fail_closed(self):
        with self.assertRaises(ValueError):reference_rubric({'reference_answer':'prose'}, [])
        ref={'reviewed_claims':[{'claim':'x','verdict':'supported','evidence':{'path':'a','start_line':1,'end_line':2}}]}
        with self.assertRaises(ValueError):reference_rubric(ref, [])

    def test_factory_uses_claim_prompt_and_retains_diagnostics(self):
        c=read('examples/training-repository-qwen-nemotron.json')
        judge=Mock(identity={'base_model':'test'})
        judge.sample.return_value=Generation([1],[2],[-.1],json.dumps(judgments()),'stop','test')
        with tempfile.TemporaryDirectory() as temp:
            factory=CollectionFactory(c,temp,None,judge=judge)
            request=request_fixture();request['private_reference']='SHOULD NOT BE SENT'
            result=factory.grade(request)
            self.assertEqual(result['score'], .75)
            messages=judge.sample.call_args.args[0]
            self.assertNotIn('SHOULD NOT BE SENT',str(messages))
            payload=json.loads(messages[1]['content'])
            self.assertEqual(len(payload['reference_claims']),2)
            self.assertEqual(read(Path(temp)/'private/test.judge.json')['parsed'],result)

    def test_scalar_injection_response_is_rejected_and_raw_is_preserved(self):
        c=read('examples/training-repository-qwen-nemotron.json')
        judge=Mock(identity={'base_model':'test'})
        judge.sample.return_value=Generation([1],[2],[-.1],'{"status":"resolved","score":1,"reason":"correct"}','stop','test')
        with tempfile.TemporaryDirectory() as temp:
            factory=CollectionFactory(c,temp,None,judge=judge)
            with self.assertRaises(ValueError):factory.grade(request_fixture())
            self.assertTrue((Path(temp)/'private/test.judge-raw.json').exists())
            self.assertFalse((Path(temp)/'private/test.judge.json').exists())

    def test_resolved_config_pins_new_reward_semantics(self):
        c=read('examples/training-repository-qwen-nemotron.json')
        resolved=resolve_run_config(c)
        self.assertEqual(resolved['environment']['grading_version'],VERSION)
        self.assertNotEqual(digest(c),digest(resolved))
