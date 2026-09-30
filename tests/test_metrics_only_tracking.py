import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock
from training_pipeline.storage import Tracker, read
from tests.test_training_pipeline import trajectory

class MetricsOnlyTrackingTests(unittest.TestCase):
    def test_large_records_stay_local_and_charts_remain(self):
        with tempfile.TemporaryDirectory() as tmp:
            wb = MagicMock()
            wb.init.return_value.url = 'https://example.test/run'
            tracker = Tracker(tmp, {'mode':'online','flush_every':1}, 'metrics', wb)
            # Exercise the formerly sampled upload path deterministically.
            from unittest.mock import patch
            with patch('training_pipeline.storage.digest', return_value='0'*64):
                tracker.trajectory(trajectory(0, 1))
            tracker.event('training_batch', attempted_batches=1, optimizer_step=0,
                          mean_reward=.5, scoring_coverage=1,
                          diagnostics={'text':'x'*1000000}, usage=[1]*10000)
            tracker.event('evaluation', optimizer_step=3, mean_correctness_score=.75,
                          completion_rate=1, results=[{'answer':'private'}])
            tracker.artifact(Path(tmp)/'answers.json', 'answers')
            tracker.finish('complete')
            wb.Table.assert_not_called()
            wb.Html.assert_not_called()
            wb.Artifact.assert_not_called()
            remote = wb.init.return_value
            payloads = [c.args[0] for c in remote.log.call_args_list]
            self.assertTrue(any(p.get('training/mean_reward')==.5 for p in payloads))
            self.assertTrue(any(p.get('evaluation/mean_correctness_score')==.75 for p in payloads))
            self.assertFalse(any(p.get('event')=='trajectory' for p in payloads))
            self.assertLess(len(json.dumps(payloads)), 3000)
            events = [json.loads(line) for line in (Path(tmp)/'events.jsonl').read_text().splitlines()]
            self.assertEqual(next(e for e in events if e['event']=='training_batch')['diagnostics']['text'], 'x'*1000000)
            self.assertEqual(len(read(Path(tmp)/'answers.json')['data']), 1)
            self.assertEqual(len(list((Path(tmp)/'trajectories').glob('*.json'))), 1)

    def test_upload_failure_does_not_lose_local_metric(self):
        with tempfile.TemporaryDirectory() as tmp:
            wb = MagicMock()
            tracker = Tracker(tmp, {'mode':'online'}, 'failure', wb)
            wb.init.return_value.log.side_effect = RuntimeError('quota')
            tracker.event('training_batch', attempted_batches=1, mean_reward=.5)
            events = [json.loads(line) for line in (Path(tmp)/'events.jsonl').read_text().splitlines()]
            self.assertTrue(any(e['event']=='training_batch' and e['mean_reward']==.5 for e in events))
            self.assertTrue(any(e['event']=='tracking_failure' for e in events))

if __name__ == '__main__': unittest.main()
