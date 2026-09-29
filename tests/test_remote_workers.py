import copy
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from agent_harness.contracts import FinalAnswer, ModelResponse, RunLimits, Usage
from agent_harness.models import ScriptedModel
from agent_harness.remote_workers import run_rollout, run_grade
from qa_eval.dataset import freeze
from qa_eval.demo import fixture
from qa_eval.security import bindings, digest, file_hash, load_json, unseal

ROLLOUT_KEY = b'rollout-test-only-' + b'x' * 32
GRADER_KEY = b'grader-test-only-' + b'y' * 32


class RemoteWorkerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.public = self.root / 'public'
        self.public.mkdir()
        task, answer, experiment, _, semantic = fixture(self.root / 'source')
        task['permitted_tools'] = ['read_file', 'search_code']
        experiment['frozen'] = False
        experiment = freeze(experiment, [task], {'synthetic': True})
        semantic.update(bindings(task, answer), experiment_hash=digest(experiment))
        bundle = self.public / 'repo.bundle'
        subprocess.run(['git', '-C', str(self.root / 'source'), 'bundle', 'create', str(bundle), 'HEAD'], check=True, capture_output=True)
        self.job = {'schema_version': 1, 'run_id': 'test-run', 'episode_id': 'episode-1',
                    'group_id': 'group-1', 'task_id': task['id'], 'task_hash': digest(task),
                    'question': task['question'],
                    'repository': {'bundle': 'repo.bundle', 'bundle_sha256': file_hash(bundle), 'commit': task['repository']['commit']},
                    'permitted_tools': task['permitted_tools'],
                    'limits': asdict(RunLimits(max_tool_calls=10, max_output_tokens=1000)),
                    'max_submission_bytes': 20000,
                    'model': {'kind': 'scripted', 'actions': [
                        {'type': 'tool_call', 'name': 'read_file', 'arguments': {'path': 'executor.py', 'start_line': 1, 'line_count': 2}},
                        {'type': 'final_answer', 'value': answer}]}, 'environment_manifest': None}
        self.private_path = self.root / 'private' / 'test-run' / 'tasks' / (task['id'] + '.json')
        self.private_path.parent.mkdir(parents=True)
        self.private_path.write_text(json.dumps({'task': task, 'experiment': experiment, 'role': 'evaluation', 'semantic': semantic, 'synthetic': True}))
        self.answer = answer

    def rollout(self, **kw):
        return run_rollout(self.job, public_root=self.public, rollout_root=self.root / 'rollouts', signing_key=ROLLOUT_KEY, **kw)

    def grade(self, **kw):
        return run_grade(self.job, public_root=self.public, private_root=self.root / 'private', rollout_root=self.root / 'rollouts', grade_root=self.root / 'grades', rollout_key=ROLLOUT_KEY, grader_key=GRADER_KEY, **kw)

    def test_full_episode_separate_grade_and_idempotent_resume(self):
        first = self.rollout()
        self.assertEqual(first['run_id'], 'test-run')
        self.assertEqual(self.rollout(model_factory=lambda _: self.fail('Should resume')), first)
        grade = self.grade()
        self.assertEqual(grade['status'], 'resolved')
        self.assertGreaterEqual(grade['reward'], .9)
        self.assertEqual(self.grade(), grade)
        saved = unseal(load_json(self.root / 'grades' / grade['report_path']), 'GradeReport', GRADER_KEY)
        self.assertEqual(saved['report']['metrics']['tool_calls'], ['read_file'])
        trajectory = (self.root / 'rollouts/test-run/episodes/episode-1/trajectory.jsonl').read_text()
        self.assertNotIn('critical_errors', trajectory)
        self.assertNotIn('Claims cancellation kills', trajectory)
        self.assertFalse((self.root / 'rollouts/test-run/episode-1/grade.json').exists())

    def test_unknown_usage_is_unresolved(self):
        self.rollout(model_factory=lambda _: ScriptedModel([ModelResponse(FinalAnswer(self.answer), Usage())]))
        grade = self.grade()
        self.assertEqual(grade['status'], 'unresolved')
        self.assertIsNone(grade['reward'])

    def test_policy_exhaustion_is_failed_not_excluded(self):
        self.job['limits']['max_steps'] = 1
        self.job['model']['actions'] = self.job['model']['actions'][:1]
        self.rollout()
        grade = self.grade()
        self.assertEqual(grade['status'], 'resolved')
        self.assertEqual(grade['reward'], 0)

    def test_tampered_envelope_rejected(self):
        self.rollout()
        path = self.root / 'rollouts/test-run/episode-1/rollout.json'
        value = load_json(path)
        value['payload']['submission']['text'] = 'Tampered'
        path.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, 'signature'):
            self.grade()

    def test_changed_public_job_cannot_resume_or_grade(self):
        self.rollout()
        self.job['question'] += ' changed'
        with self.assertRaisesRegex(ValueError, 'different public job'):
            self.rollout()
        with self.assertRaisesRegex(ValueError, 'different public job'):
            self.grade()

    def test_changed_private_task_rejected(self):
        self.rollout()
        record = load_json(self.private_path)
        record['task']['question'] = 'Changed private question'
        self.private_path.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError, 'Private task'):
            self.grade()

    def test_corrupt_bundle_rejected_before_policy(self):
        (self.public / 'repo.bundle').write_bytes(b'corrupt')
        with self.assertRaisesRegex(ValueError, 'bundle hash'):
            self.rollout(model_factory=lambda _: self.fail('Must not invoke model'))

    def test_private_fields_rejected(self):
        self.job['claims'] = []
        with self.assertRaises(ValueError):
            self.rollout()

    def test_semantic_replay_requires_explicit_synthetic_record(self):
        self.rollout()
        record = load_json(self.private_path)
        del record['synthetic']
        self.private_path.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError, 'synthetic'):
            self.grade()

    def test_changed_grade_config_cannot_reuse_cached_report(self):
        self.rollout()
        self.grade()
        record = load_json(self.private_path)
        record['experiment']['judge_version'] = 'new-version'
        self.private_path.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError, 'grading configuration'):
            self.grade()

    def test_public_budget_inflation_rejected_by_grader(self):
        self.job['limits']['max_tool_calls'] = 11
        self.rollout()
        with self.assertRaisesRegex(ValueError, 'budgets'):
            self.grade()

    def test_same_signing_key_rejected(self):
        with self.assertRaisesRegex(ValueError, 'distinct'):
            run_grade(self.job, public_root=self.public, private_root=self.root / 'private', rollout_root=self.root / 'rollouts', grade_root=self.root / 'grades', rollout_key=ROLLOUT_KEY, grader_key=ROLLOUT_KEY)


if __name__ == '__main__':
    unittest.main()
