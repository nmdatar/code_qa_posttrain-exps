import copy
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from pathlib import Path

from training_pipeline.admission import sha
from training_pipeline.collection import CollectionEpisode, CollectionFactory
from training_pipeline.definition_evidence import definition_context
from training_pipeline.config import validate_config


class ActionAliasTests(unittest.TestCase):
    def episode(self, enabled=True):
        e = object.__new__(CollectionEpisode)
        e.factory = SimpleNamespace(config={'environment': {'tool_action_policy': 'action-alias-v1'} if enabled else {}})
        e.row = {'public': {'permitted_tools': ['read_file', 'search_code', 'list_files']}}
        return e

    def test_explicit_allowed_alias_only(self):
        x = {'action': 'read_file', 'arguments': {'path': 'a.py', 'start_line': 1, 'end_line': 4}}
        e = self.episode()
        self.assertEqual(e.parse_action(json.dumps(x)), {'tool': 'read_file', 'arguments': x['arguments']})
        self.assertEqual(self.episode(False).parse_action(json.dumps(x)), x)
        for bad in [{'arguments': {}}, {'action': 'shell', 'arguments': {}},
                    {'action': 'read_file', 'tool': 'search_code', 'arguments': {}}]:
            self.assertEqual(e.parse_action(json.dumps(bad)), bad)
            with self.assertRaisesRegex(ValueError, 'Invalid tool'): e.step(bad)

    def test_alias_does_not_bypass_path_or_argument_checks(self):
        e = self.episode(); e.row['image_result'] = {'snapshot_files': {'a.py': 'hash'}}
        for args in [{'path': '../secret'}, {'path': 'a.py', 'invented': True}]:
            with self.assertRaises(ValueError):
                e.step(e.parse_action(json.dumps({'action': 'read_file', 'arguments': args})))


class DefinitionEvidenceTests(unittest.TestCase):
    def request(self, source, name='actual_function'):
        return {'question':'Which function handles values?', 'answer': {'text': f'`{name}` handles this.'},
                'rubric': {'claims': [{'text': 'Handles values.'}], 'evidence': [
                    {'path': 'a.py', 'start_line': 85, 'end_line': 85, 'file_sha256': sha(source)}]},
                'source_row': {'snapshot_root': 'root', 'public': {'repository': {'commit': 'pin'}},
                               'image_result': {'snapshot_files': {'a.py': sha(source)}}}}

    def test_header_outside_padding_and_wrong_candidate_name(self):
        source = ('def actual_function():\n    """Documented operation."""\n' + '    pass\n'*100).encode()
        req = self.request(source, 'invented_function')
        original = copy.deepcopy(req)
        with patch('training_pipeline.definition_evidence.blob', return_value=source):
            refs = definition_context(req)
        self.assertEqual(refs, [{'path': 'a.py', 'start_line': 1, 'end_line': 2, 'file_sha256': sha(source)}])
        self.assertEqual(req, original)  # citations and candidate remain unchanged

    def test_hash_mismatch_rejected(self):
        req = self.request(b'pass\n')
        with patch('training_pipeline.definition_evidence.blob', return_value=b'changed\n'):
            with self.assertRaisesRegex(ValueError, 'hash mismatch'): definition_context(req)

    def test_policy_changes_reward_and_environment_identity(self):
        config = json.loads(Path('configs/experiments/grpo-autoresearch/train-pagination-v2.json').read_text())
        old = CollectionFactory(config, Path('/tmp'), None)
        new = copy.deepcopy(config)
        new['environment'].update(judge_evidence_policy='definition-context-v1', tool_action_policy='action-alias-v1')
        validate_config(new)
        revised = CollectionFactory(new, Path('/tmp'), None)
        self.assertNotEqual(old.identity, revised.identity)
        self.assertNotEqual(old.reward_version, revised.reward_version)
        new['environment']['tool_action_policy'] = 'guess-missing-tools'
        with self.assertRaises(ValueError): validate_config(new)

    def test_strict_assessment_receives_definition_without_rewriting_citation(self):
        from training_pipeline.strict_grading import assess
        source = ('def actual_function():\n    """Documented operation."""\n' + '    pass\n'*100).encode()
        req = self.request(source)
        req['source_row'].update(id='case', split='train', reference={'evidence_revision':'reference-citations-v1'})
        req['rubric'].update(rubric_hash='hash')
        req['rubric']['claims'] = [{'id':'c1','text':'Handles values.','weight':1.}]
        req['answer'].update(diagram=None, citations=[{'id':'c1', **req['rubric']['evidence'][0]}])
        original = copy.deepcopy(req)
        class Captured(Exception): pass
        captured = []
        def capture(*args, **kwargs):
            captured.append(copy.deepcopy(args[5]))
            raise Captured()
        with patch('training_pipeline.definition_evidence.blob', return_value=source), patch('training_pipeline.strict_grading.blob', return_value=source), patch('training_pipeline.strict_grading.judge', side_effect=capture):
            for policy in ['legacy-v1', 'definition-context-v1']:
                with self.assertRaises(Captured): assess(req, None, 'all-claims-v7', evidence_policy=policy)
        self.assertFalse(any('def actual_function' in e['text'] for e in captured[0].values()))
        self.assertTrue(any('def actual_function' in e['text'] for e in captured[1].values()))
        self.assertEqual(req, original)


class PlaceholderRepairTests(unittest.TestCase):
    def test_train_only_reference_binding_and_evidence_coverage(self):
        from scripts.repair_training_placeholders import revised_reference, PLACEHOLDER
        from training_pipeline.claim_grading import reference_rubric
        prose = 'First fact. Second fact.'
        refs = [{'path':f'{p}.py','start_line':1,'end_line':2} for p in ['a','b']]
        row = {'id':'t','split':'train','reference':{'reference_answer':prose,'reference_sha256':sha(prose.encode()),
            'reviewed_claims':[{'claim':PLACEHOLDER}], 'verified_evidence':refs}}
        draft = {'task_id':'t','reference_sha256':sha(prose.encode()),'reference_text_for_decomposition':prose}
        original = copy.deepcopy(row)
        revised = revised_reference(row,draft,['First fact.','Second fact.'])
        rubric = reference_rubric(revised, refs)
        self.assertEqual([c['evidence_ids'] for c in rubric['claims']], [['e1','e2'],['e1','e2']])
        self.assertEqual(row, original)
        row['split']='development'
        with self.assertRaisesRegex(ValueError,'Only training'): revised_reference(row,draft,['First fact.'])
        row['split']='train';draft['reference_text_for_decomposition']='Different prose'
        with self.assertRaisesRegex(ValueError,'identity'): revised_reference(row,draft,['First fact.'])
