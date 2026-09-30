import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from training_pipeline.cohorts import build_cohorts, select_tasks
from training_pipeline.contracts import ConfigurationError
from training_pipeline.storage import atomic_json, digest, Tracker
from training_pipeline.task_subset import apply_subset
from tests.test_training_pipeline import trajectory


class TaskSubsetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.data = {'identity':'source', 'tasks':[
            {'id':f't{i}', 'family_id':'train', 'split':'train'} for i in range(4)],
            'development':[{'id':f'e{i}', 'family_id':'eval', 'split':'development'} for i in range(8)]}
        self.cohorts = build_cohorts(self.data, 4)
        atomic_json(self.root/'cohorts.json', self.cohorts)
        pool = {'data_identity':'source', 'task_ids':['t0','t1']}
        pool['manifest_hash'] = digest(pool)
        self.manifest = {'kind':'selected-task-pilot-v1', 'data_identity':'source',
                         'training_pool':pool, 'training_ids':['t1'],
                         'evaluation_ids':self.cohorts['selection'][:2]}
        self.config = {'evaluation':{'cohort_manifest':str(self.root/'cohorts.json'),
                       'cohort_sha256':self.cohorts['manifest_hash'], 'max_tasks':2}}
        self.save()

    def save(self):
        p = self.root/'subset.json'; atomic_json(p, self.manifest)
        self.config['task_subset'] = {'manifest':str(p), 'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}

    def test_pinned_subset_preserves_confirmation(self):
        data = apply_subset(self.config, self.data)
        self.assertEqual([t['id'] for t in data['tasks']], ['t1'])
        self.assertEqual(data['snapshot_tasks'], self.data['tasks'])
        selected, identity = select_tasks(data, self.config['evaluation'])
        self.assertEqual([t['id'] for t in selected], self.manifest['evaluation_ids'])
        self.assertIn('subset_sha256', identity)
        confirm, _ = select_tasks(data, self.config['evaluation'], 'confirmation')
        self.assertEqual([t['id'] for t in confirm], self.cohorts['confirmation'])

    def test_training_and_eval_leakage_rejected(self):
        self.manifest['training_ids'] = ['t2']; self.save()
        with self.assertRaises(ConfigurationError): apply_subset(self.config, self.data)
        self.manifest['training_ids'] = ['t1']
        self.manifest['evaluation_ids'] = self.cohorts['confirmation'][:1]; self.save()
        with self.assertRaises(ConfigurationError): apply_subset(self.config, self.data)

    def test_manifest_tampering_rejected(self):
        (self.root/'subset.json').write_text('{}')
        with self.assertRaises(ConfigurationError): apply_subset(self.config, self.data)

    def test_final_only_flush_preserves_every_raw_trace(self):
        tracker = Tracker(self.root/'run', {'mode':'disabled','project':'test','flush_every':0}, 'test')
        with patch.object(tracker, 'flush_answers') as flush:
            for i in range(32): tracker.trajectory(trajectory(i, .5))
            flush.assert_not_called()
            self.assertEqual(len(list((tracker.root/'trajectories').glob('*.json'))),32)
            tracker.finish('complete')
            flush.assert_called_once()
