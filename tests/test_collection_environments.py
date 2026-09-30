import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from dataset_builder import collection_environments as envs
from dataset_builder.contracts import canonical_hash
from tests.test_dataset_investigate import bundle as make_bundle


class CollectionEnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = make_bundle(Path(self.tmp.name))
        self.built = envs._read(self.root / 'environment-build/result.json')
        self.environment = envs._read(self.root / 'public/environment.json')

    def test_ready_requires_immutable_image_and_current_recipe(self):
        self.assertTrue(envs._ready(self.built, self.environment))
        self.assertFalse(envs._ready({**self.built, 'image_digest': 'latest'}, self.environment))
        self.assertFalse(envs._ready(self.built, {**self.environment, 'recipe': {'changed': True}}))

    def test_verified_text_rejects_modified_source(self):
        self.assertEqual(envs._select_text(self.environment, self.built), ('code.py', 1, 'value = 7'))
        (Path(self.environment['snapshot_path']) / 'code.py').write_text('modified = 1\n')
        with self.assertRaisesRegex(ValueError, 'No verified'):
            envs._select_text(self.environment, self.built)

    def test_quarantine_is_saved_instead_of_raising(self):
        (self.root / 'public/tasks.jsonl').write_text('{}\n')
        result = envs.process_bundle(self.root)
        self.assertEqual(result['status'], 'quarantined')
        self.assertIn('hash mismatch', result['error'])
        self.assertEqual(envs._read(self.root / 'private/collection-environment-status.json'), result)

    def test_runtime_is_not_teacher_and_resume_rejects_tampered_evidence(self):
        def tool(attempt, name, args):
            field = {'list_files': 'files', 'search_code': 'matches', 'read_file': 'lines'}[name]
            event = {'name': name, 'status': 'ok', 'record_sha256': name,
                     'observation': {'stdout': json.dumps({field: ['nonempty']}), 'sandbox_id': name}}
            with (attempt / 'trajectory.jsonl').open('a') as stream:
                stream.write(json.dumps(event) + '\n')
            return event
        def replay(attempt, output):
            result = {'status': 'passed', 'checks': [{'matched': True}] * 3}
            envs._save(output, result)
            return result
        with patch.object(envs.investigate, 'tool', side_effect=tool), patch.object(envs.investigate, 'replay', side_effect=replay):
            result = envs.check_source_runtime(self.root, self.built, self.environment)
        self.assertFalse(result['answer_generated'])
        self.assertFalse(result['teacher_investigation'])
        self.assertTrue(envs._runtime_reusable(result, self.built, self.root))
        self.assertFalse(list(self.root.glob('private/source-runtime-attempts/*/answer.json')))
        (self.root / result['attempt_path'] / 'trajectory.jsonl').write_text('tampered')
        self.assertFalse(envs._runtime_reusable(result, self.built, self.root))

    def test_empty_observation_rejected(self):
        with patch.object(envs.investigate, 'tool', return_value={'status': 'ok', 'observation': {'stdout': '{"files": []}'}}):
            with self.assertRaisesRegex(ValueError, 'Empty source runtime'):
                envs.check_source_runtime(self.root, self.built, self.environment)

    def test_bound_checks_reused_and_failure_rerun(self):
        binding = {'environment_hash': canonical_hash(self.built), 'image_id': self.built['image_digest']}
        envs._save(self.root / 'private/isolation-report.json', {**binding, 'passed': True, 'checks': [{'a': True}, {'a': True}]})
        envs._save(self.root / 'private/image-attestation.json', {**binding, 'status': 'passed', 'observation': {'source_files_checked': 1}})
        with patch.object(envs.isolation, 'check_isolation') as isolate, patch.object(envs.attest, 'attest') as attest, patch.object(envs, 'check_source_runtime', return_value={'replayed_operations': 3}):
            self.assertEqual(envs.process_bundle(self.root)['status'], 'passed')
            isolate.assert_not_called()
            attest.assert_not_called()
        envs._save(self.root / 'private/isolation-report.json', {**binding, 'environment_hash': 'stale', 'passed': True})
        with patch.object(envs.isolation, 'check_isolation', return_value={**binding, 'passed': False}) as isolate:
            self.assertEqual(envs.process_bundle(self.root)['status'], 'quarantined')
            isolate.assert_called_once()


if __name__ == '__main__':
    unittest.main()
