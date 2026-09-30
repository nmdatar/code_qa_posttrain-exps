import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from training_pipeline.judge_viewer import collect, write_report, log_report
from training_pipeline.storage import Tracker, atomic_json


class JudgeViewerTests(unittest.TestCase):
    def test_attempts_scores_escaping_and_prompt_exclusion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            atomic_json(root/'trajectories/ep.json', {'episode_id':'ep', 'task_id':'task',
                'verification':{'status':'resolved', 'reward':.5, 'diagnostics':{
                    'strict_score':None, 'strict_status':'unresolved', 'training_reward':.5,
                    'private_secret':'DO_NOT_EXPORT'}}})
            raw = {'generation':{'text':'</pre><script>alert(1)</script>', 'prompt':'PRIVATE_PROMPT',
                                 'tokens':[1, 2], 'stop_reason':'length'},
                   'request':{'untrusted':{'evidence':'PRIVATE_SOURCE'}}, 'validation_error':'Truncated'}
            atomic_json(root/'private/ep.assess.judge-raw.json', raw)
            atomic_json(root/'private/ep.assess.repair-1.judge-raw.json', {
                'generation':{'text':'{"assessment_complete": true}', 'stop_reason':'stop'}})
            atomic_json(root/'private/ep.coverage-0.json', {'generation':{'text':'{"claims": []}'}})
            path, records = write_report(root)
            self.assertEqual([(a['stage'],a['attempt']) for a in records[0]['attempts']],
                             [('assess',0), ('assess',1), ('coverage',0)])
            self.assertIsNone(records[0]['strict_score'])
            self.assertEqual(records[0]['training_reward'], .5)
            self.assertEqual(records[0]['attempts'][0]['output_tokens'], 2)
            html = path.read_text()
            self.assertIn('&lt;script&gt;', html)
            self.assertNotIn('<script>alert(1)', html)
            for forbidden in ('PRIVATE_PROMPT','PRIVATE_SOURCE','DO_NOT_EXPORT'):
                self.assertNotIn(forbidden, path.with_suffix('.json').read_text())
            logged = []
            log_report(SimpleNamespace(log=logged.append),
                       SimpleNamespace(Table=lambda **kw:kw, Html=lambda text, **kw:text), path, records)
            self.assertEqual(len(logged[0]['judge_answers']['data']), 3)

    def test_missing_traces_and_corrupt_attempt_remain_visible(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'private').mkdir()
            (root/'private/old.judge-raw.json').write_text('{broken')
            records = collect(root)
            self.assertEqual(records[0]['episode_id'], 'old')
            self.assertIn('read_error', records[0]['attempts'][0])

    def test_tracker_generates_locally_and_observability_failure_is_nonfatal(self):
        with tempfile.TemporaryDirectory() as tmp:
            tracker = Tracker(tmp, {'mode':'disabled'}, 'test')
            tracker.flush_answers()
            self.assertTrue((Path(tmp)/'judge-viewer.html').exists())
            with patch('training_pipeline.judge_viewer.write_report', side_effect=OSError('unavailable')):
                tracker.flush_answers()
            events = [json.loads(line) for line in (Path(tmp)/'events.jsonl').read_text().splitlines()]
            self.assertEqual(events[-1]['artifact'], 'judge-viewer')


if __name__ == '__main__':
    unittest.main()
