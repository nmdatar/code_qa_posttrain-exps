import copy
import unittest

from training_pipeline.benchmark import task_manifest
from training_pipeline.contracts import ConfigurationError
from training_pipeline.task_expansion import create, validate


class TaskExpansionTests(unittest.TestCase):
    def setUp(self):
        self.data = {'identity': 'test', 'tasks': [
            {'id': str(i), 'family_id': str(i % 3), 'split': 'train'} for i in range(12)]}
        self.previous = task_manifest(self.data, 6)
        self.manifest = create(self.data, self.previous)
        self.config = {'benchmark': {'manifest_hash': self.manifest['manifest_hash']},
                       'execution': {'operation': 'benchmark'}}

    def test_exact_disjoint_complement(self):
        tasks = validate(self.manifest, self.config, self.data)
        old, new = set(self.previous['task_ids']), {t['id'] for t in tasks}
        self.assertFalse(old & new)
        self.assertEqual(old | new, {t['id'] for t in self.data['tasks']})

    def test_tampering_and_wrong_execution_rejected(self):
        bad = copy.deepcopy(self.manifest)
        bad['task_ids'][0] = self.previous['task_ids'][0]
        with self.assertRaises(ConfigurationError):
            validate(bad, self.config, self.data)
        self.config['execution']['operation'] = 'run'
        with self.assertRaises(ConfigurationError):
            validate(self.manifest, self.config, self.data)

    def test_heldout_and_different_release_rejected(self):
        self.data['identity'] = 'changed'
        with self.assertRaises(ConfigurationError):
            validate(self.manifest, self.config, self.data)
        self.data['identity'] = 'test'
        selected = self.manifest['task_ids'][0]
        next(t for t in self.data['tasks'] if t['id'] == selected)['split'] = 'development'
        with self.assertRaises(ConfigurationError):
            validate(self.manifest, self.config, self.data)
