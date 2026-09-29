import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from training_pipeline.admission import assess, sha, verified_source


class AdmissionTests(unittest.TestCase):
    def setUp(self):
        self.repo = {'commit': 'a'*40, 'family_id': 'org/repo', 'url': 'https://github.com/org/repo'}
        self.task = {'user_prompt': 'What does f do?', 'repository': self.repo, 'permitted_tools': ['read_file']}
        self.quality = {'runtime_status': 'passed', 'grading_reference_available': True,
            'reference_admission': 'independently_reviewed_correction_draft', 'semantic_review': 'reject',
            'correction_review': 'supported', 'execution_requirement_screen': 'no_explicit_execution_request_detected',
            'question_sha256': sha(b'What does f do?'), 'selected_reference_sha256': sha(b'Corrected answer')}
        self.grading = {'kind': 'source_reference_comparison', 'reference_answer': 'Corrected answer',
            'reference_sha256': sha(b'Corrected answer'), 'verified_evidence': [
                {'path': 'a.py', 'file_sha256': sha(b'pass\n'), 'start_line': 1, 'end_line': 1}]}
        self.env = {'repository': self.repo, 'runtime_checked': True, 'capability': 'source_reading', 'image_id': 'im-1'}
        self.evidence = {'source': {'repository': self.repo, 'snapshot_path': '/unused'},
            'result': {'image_digest': 'im-1', 'commit': 'a'*40, 'snapshot_files': {'a.py': sha(b'pass\n')}},
            'isolation': {'passed': True, 'image_id': 'im-1'}, 'attestation': {'status': 'passed', 'image_id': 'im-1'}}

    def assess(self):
        with patch('training_pipeline.admission.blob', return_value=b'pass\n'):
            return assess(self.task, self.quality, self.grading, self.env, self.evidence)

    def test_corrected_reference_can_supersede_rejected_original(self):
        before = copy.deepcopy(self.quality)
        self.assertEqual(self.assess(), [])
        self.assertEqual(self.quality, before)
        self.assertNotIn('human_reviewed', self.quality)

    def test_unresolved_correction_excluded(self):
        self.quality['correction_review'] = 'needs_review'
        self.assertIn('selected_reference_not_supported', self.assess())

    def test_reference_question_and_evidence_bindings(self):
        self.grading['reference_answer'] += ' changed'
        self.task['user_prompt'] += ' changed'
        self.grading['verified_evidence'][0]['end_line'] = 2
        self.assertTrue({'selected_reference_hash', 'question_review_binding', 'source_evidence_mismatch'} <= set(self.assess()))

    def test_execution_and_image_mismatch_excluded(self):
        self.task['permitted_tools'].append('python_probe')
        self.evidence['attestation']['image_id'] = 'im-other'
        self.assertTrue({'execution_requirement_in_source_only_environment', 'environment_evidence_binding'} <= set(self.assess()))

    def test_source_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'data').write_text('original')
            (root/'manifest.json').write_text(json.dumps({'artifacts': {'data': sha(b'original')}}))
            verified_source(root)
            (root/'data').write_text('modified')
            with self.assertRaisesRegex(ValueError, 'mismatch'):
                verified_source(root)

    def test_export_keeps_private_references_and_disjoint_families(self):
        from training_pipeline.admission import export_rollouts
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp)/'source'
            source.mkdir()
            tasks = [{'id': str(i), 'user_prompt': 'Question '+str(i), 'split': 'development',
                      'repository': {'family_id': 'org/repo'+str(i//2)}, 'environment_id': 'env'} for i in range(20)]
            files = {'public/tasks.jsonl': tasks, 'public/environments.jsonl': [{'environment_id': 'env'}],
                     'private/quality.jsonl': [{'task_id': t['id']} for t in tasks],
                     'private/grading.jsonl': [{'task_id': t['id'], 'reference_answer': 'PRIVATE ANSWER'} for t in tasks]}
            hashes = {}
            for name, rows in files.items():
                path = source/name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
                hashes[name] = sha(path.read_bytes())
            (source/'manifest.json').write_text(json.dumps({'artifacts': hashes}))
            before = (source/'public/tasks.jsonl').read_bytes()
            with patch('training_pipeline.admission.assess', return_value=[]):
                report = export_rollouts(source, Path(tmp)/'export')
            root = Path(tmp)/'export'
            verified_source(root)
            self.assertEqual(before, (source/'public/tasks.jsonl').read_bytes())
            self.assertEqual(report['admitted'], 20)
            self.assertFalse(report['human_reviewed'])
            rows = [json.loads(x) for x in (root/'public/tasks.jsonl').read_text().splitlines()]
            family_splits = {}
            for row in rows:
                family_splits.setdefault(row['repository']['family_id'], set()).add(row['split'])
            self.assertTrue(all(len(splits) == 1 for splits in family_splits.values()))
            self.assertEqual({r['split'] for r in rows}, {'train', 'development'})
            self.assertNotIn('PRIVATE ANSWER', (root/'rl/train.jsonl').read_text())
            with self.assertRaisesRegex(ValueError, 'already exists'):
                export_rollouts(source, root)
