import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "artifacts/efficiency-rl-v2-source"))
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock
from training_pipeline.config import validate_config
from training_pipeline.storage import Tracker
from training_pipeline.efficiency_metrics import summarize

class IterationReportingTests(unittest.TestCase):
    def test_aggregate_tracking_retains_private_data_only_locally(self):
        with tempfile.TemporaryDirectory() as tmp:
            tracker=Tracker(tmp,{'mode':'disabled','aggregate_only':True},'test')
            tracker.remote=Mock()
            tracker.event('trajectory',task_id='private-task',reward=1)
            tracker.remote.log.assert_not_called()
            tracker.event('evaluation',optimizer_step=4,mean_output_tokens=123.,scoring_coverage=1.,
                          artifact='private-path',results=[{'answer':'private-answer'}])
            logged=tracker.remote.log.call_args.args[0]
            self.assertEqual(logged['evaluation/mean_output_tokens'],123)
            self.assertNotIn('results',logged);self.assertNotIn('artifact',logged)
            tracker.answer_rows=[['local-answer']]
            tracker.flush_answers()
            self.assertEqual(json.loads((Path(tmp)/'answers.json').read_text())['data'],[['local-answer']])
            tracker.artifact(Path(tmp)/'answers.json','answers')
            tracker.remote.log_artifact.assert_not_called()
            self.assertIn('private-task',(Path(tmp)/'events.jsonl').read_text())

    def test_config_requires_boolean_aggregate_flag(self):
        c=json.loads(Path('configs/experiments/efficiency-rl-v1/efficiency.json').read_text())
        c['tracking']['aggregate_only']=True
        validate_config(c)
        c['tracking']['aggregate_only']='yes'
        with self.assertRaises(ValueError):validate_config(c)

    def test_runtime_excludes_grading_and_provisioning(self):
        t=NS(usage={'input_tokens':100,'output_tokens':50,'tool_calls':1,'tool_seconds':1,
                    'generation_seconds':7,'action_seconds':2,'latency_seconds':30},
             verification=NS(status='resolved',diagnostics={'tier':'accepted'}))
        self.assertEqual(summarize([t])['mean_model_action_seconds'],9)
        del t.usage['action_seconds']
        self.assertIsNone(summarize([t])['mean_model_action_seconds'])

if __name__=='__main__':unittest.main()
