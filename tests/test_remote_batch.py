import copy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from agent_harness.code_tools import GitRepository
from agent_harness.remote_batch import prepare_batch
from agent_harness.remote_contracts import validate_job, validate_model
from agent_harness.remote_workers import repository_snapshot
from qa_eval.dataset import freeze
from qa_eval.demo import fixture


class RemoteBatchTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.repo = self.root / 'repo'
        task, answer, experiment, _, _ = fixture(self.repo)
        task['permitted_tools'] = ['read_file', 'search_code']
        task['critical_errors'] = ['PRIVATE_GRADING_SENTINEL']
        experiment['frozen'] = False
        experiment = freeze(experiment, [task], {'synthetic': True})
        self.task, self.experiment, self.answer = task, experiment, answer
        self.task_path, self.experiment_path = self.root / 'task.json', self.root / 'experiment.json'
        self.task_path.write_text(json.dumps(task))
        self.experiment_path.write_text(json.dumps(experiment))
        self.output = self.root / 'bundle'
        self.config = {'run_id': 'run-1', 'episodes_per_task': 2, 'max_rollouts': 2,
            'max_graders': 1, 'coordinator_seconds': 100, 'synthetic': True,
            'model': {'kind': 'scripted', 'actions': [{'type': 'final_answer', 'value': answer}]},
            'tasks': [{'task': str(self.task_path), 'experiment': str(self.experiment_path), 'repository': str(self.repo)}]}

    def prepare(self):
        return prepare_batch(self.config, self.output)

    def git(self, *args):
        return subprocess.run(['git', '-C', str(self.repo), *args], check=True, capture_output=True, text=True).stdout.strip()

    def test_public_private_separation_and_bundle_snapshot(self):
        commit = self.task['repository']['commit']
        expected = (self.repo / 'executor.py').read_bytes()
        self.git('checkout', '-qb', 'unrelated-private-branch')
        (self.repo / 'private.txt').write_text('PRIVATE_BRANCH_SENTINEL')
        self.git('add', 'private.txt')
        self.git('-c', 'user.name=Test', '-c', 'user.email=test@example.invalid', '-c', 'commit.gpgsign=false', 'commit', '-qm', 'unrelated private file')
        (self.repo / 'executor.py').write_text('DIRTY_WORKTREE_SENTINEL')
        (self.repo / 'untracked.txt').write_text('UNTRACKED_SENTINEL')
        result = self.prepare()
        self.assertEqual(len(result['jobs']), 2)
        job = result['jobs'][0]
        bundle = self.output / 'public' / job['repository']['bundle']
        self.assertEqual(hashlib.sha256(bundle.read_bytes()).hexdigest(), job['repository']['bundle_sha256'])
        for path in (self.output / 'public').rglob('*.json'):
            self.assertNotIn('PRIVATE_GRADING_SENTINEL', path.read_text())
            self.assertNotIn('critical_errors', path.read_text())
            self.assertNotIn(str(self.repo), path.read_text())
        private = json.loads((self.output / 'private/run-1/tasks' / (self.task['id'] + '.json')).read_text())
        self.assertEqual(private['task']['critical_errors'], ['PRIVATE_GRADING_SENTINEL'])
        with repository_snapshot(job, self.output / 'public') as clone:
            self.assertEqual((clone / 'executor.py').read_bytes(), expected)
            self.assertFalse((clone / 'private.txt').exists())
            self.assertFalse((clone / 'untracked.txt').exists())
            self.assertEqual(GitRepository(clone, commit).files(), ['executor.py'])
            branches = subprocess.check_output(['git', '-C', str(clone), 'branch', '-a'], text=True)
            self.assertNotIn('unrelated-private-branch', branches)
        self.assertEqual((self.repo / 'executor.py').read_text(), 'DIRTY_WORKTREE_SENTINEL')
        self.assertEqual(result['jobs'][0]['repository'], result['jobs'][1]['repository'])
        self.assertEqual(len(list((self.output / 'public').rglob('*.bundle'))), 1)

    def test_task_budgets_are_clamped_and_group_stable(self):
        self.config['limits'] = {'max_tool_calls': 99, 'max_output_tokens': 99999}
        result = self.prepare()
        first, second = result['jobs']
        self.assertNotEqual(first['episode_id'], second['episode_id'])
        self.assertEqual(first['group_id'], second['group_id'])
        self.assertEqual(first['limits']['max_tool_calls'], self.task['budgets']['max_tool_calls'])
        self.assertEqual(first['limits']['max_output_tokens'], self.task['budgets']['max_output_tokens'])
        self.assertEqual(first['max_submission_bytes'], self.task['budgets']['max_submission_bytes'])

    def test_duplicate_task_rejected_before_output(self):
        self.config['tasks'].append(dict(self.config['tasks'][0]))
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            self.prepare()
        self.assertFalse(self.output.exists())

    def test_frozen_experiment_membership_required(self):
        experiment = dict(self.experiment)
        experiment['frozen'] = False
        self.experiment_path.write_text(json.dumps(experiment))
        with self.assertRaisesRegex(ValueError, 'frozen'):
            self.prepare()
        self.assertFalse(self.output.exists())

    def test_final_test_cannot_be_training(self):
        self.config['tasks'][0]['role'] = 'training'
        with self.assertRaisesRegex(ValueError, 'train-split'):
            self.prepare()
        self.assertFalse(self.output.exists())

    def test_scripted_requires_explicit_synthetic(self):
        self.config.pop('synthetic')
        with self.assertRaisesRegex(ValueError, 'synthetic'):
            self.prepare()
        self.assertFalse(self.output.exists())

    def test_preloaded_semantics_require_synthetic(self):
        self.config['model'] = {'kind': 'http', 'model': 'test', 'base_url': 'https://example.invalid/v1'}
        self.config['synthetic'] = False
        self.config['tasks'][0]['semantic'] = str(self.root / 'unused-private.json')
        with self.assertRaisesRegex(ValueError, 'synthetic'):
            self.prepare()
        self.assertFalse(self.output.exists())

    def test_environment_projection_omits_logs_and_build_secrets(self):
        files = GitRepository(self.repo, self.task['repository']['commit']).snapshot_hashes()
        environment = {'backend': 'modal', 'commit': self.task['repository']['commit'],
            'capability': 'execution', 'image_digest': 'im-test', 'snapshot_files': files,
            'snapshot_sha256': hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest(),
            'status': 'ready', 'readiness': {'exit_code': 0, 'timed_out': False, 'truncated': False,
                'stdout': 'PRIVATE_BUILD_SENTINEL', 'stderr': 'PRIVATE_BUILD_SENTINEL'},
            'runtime_policy': {'network': 'blocked', 'host_mounts': False, 'secrets': False,
                'source_read_only': True, 'fresh_sandbox_per_operation': True},
            'install_commands': ['PRIVATE_BUILD_SENTINEL'], 'private_key': 'PRIVATE_BUILD_SENTINEL'}
        env_path = self.root / 'environment.json'
        env_path.write_text(json.dumps(environment))
        self.config['tasks'][0]['environment_manifest'] = str(env_path)
        result = self.prepare()
        self.assertNotIn('PRIVATE_BUILD_SENTINEL', json.dumps(result))
        self.assertEqual(result['jobs'][0]['environment_manifest']['snapshot_files'], files)
        self.assertEqual(result['jobs'][0]['environment_manifest']['readiness'],
                         {'exit_code': 0, 'timed_out': False, 'truncated': False})

    def test_unknown_configuration_and_invalid_limit_types(self):
        cases = [None, [], {**self.config, 'secrets': {}}, {**self.config, 'max_rollouts': True},
                 {**self.config, 'synthetic': 'true'}, {**self.config, 'limits': {'max_steps': False}},
                 {**self.config, 'tasks': [{}]}, {**self.config, 'limits': {'wall_time_seconds': float('nan')}}]
        for config in cases:
            with self.subTest(config=config), self.assertRaises(ValueError):
                prepare_batch(config, self.output)
            self.assertFalse(self.output.exists())

    def test_remote_job_rejects_bad_hashes_paths_and_private_fields(self):
        job = self.prepare()['jobs'][0]
        variants = []
        for key, value in [('task_hash', None), ('schema_version', True), ('question', ' '),
                           ('claims', []), ('max_submission_bytes', True)]:
            variant = copy.deepcopy(job)
            variant[key] = value
            variants.append(variant)
        for key, value in [('commit', 123), ('bundle_sha256', []), ('bundle', '../private/task.json')]:
            variant = copy.deepcopy(job)
            variant['repository'][key] = value
            variants.append(variant)
        for variant in variants:
            with self.subTest(variant=variant), self.assertRaises(ValueError):
                validate_job(variant)

    def test_model_required_fields_types_and_credentials(self):
        cases = [None, [], {'kind': 'http'}, {'kind': 'http', 'model': [], 'base_url': 'https://example.invalid'},
                 {'kind': 'http', 'model': 'x', 'base_url': 'https://user:secret@example.invalid'},
                 {'kind': 'http', 'model': 'x', 'base_url': 'https://example.invalid', 'api_key': 'secret'},
                 {'kind': 'http', 'model': 'x', 'base_url': 'https://example.invalid', 'timeout_seconds': True},
                 {'kind': 'http', 'model': 'x', 'base_url': 'https://example.invalid', 'input_price_per_million': True},
                 {'kind': 'scripted'}, {'kind': 'scripted', 'actions': []},
                 {'kind': 'scripted', 'actions': [{'type': 'tool_call', 'name': 'x', 'arguments': []}]},
                 {'kind': 'tinker', 'base_model': 'x'},
                 {'kind': 'tinker', 'base_model': 'x', 'renderer_name': 'r', 'temperature': []}]
        for model in cases:
            with self.subTest(model=model), self.assertRaises(ValueError):
                validate_model(model)
        validate_model({'kind': 'tinker', 'base_model': 'x', 'renderer_name': 'r', 'temperature': 1.0})
        validate_model({'kind': 'http', 'model': 'x', 'base_url': 'https://example.invalid/v1'})

    def test_tinker_episode_seeds_are_distinct_and_reproducible(self):
        self.config['model'] = {'kind': 'tinker', 'base_model': 'test', 'renderer_name': 'test', 'seed': 42}
        result = self.prepare()
        seeds = [job['model']['seed'] for job in result['jobs']]
        self.assertEqual(len(set(seeds)), 2)
        self.assertTrue(all(0 <= seed < 2 ** 31 for seed in seeds))
        second = prepare_batch(self.config, self.root / 'bundle-2')
        self.assertEqual(seeds, [job['model']['seed'] for job in second['jobs']])
        self.assertEqual(self.config['model']['seed'], 42)

    def test_total_job_cap_rejected_before_output(self):
        self.config['episodes_per_task'] = 1001
        with self.assertRaisesRegex(ValueError, '1000'):
            self.prepare()
        self.assertFalse(self.output.exists())
