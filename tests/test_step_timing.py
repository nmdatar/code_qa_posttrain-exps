import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock
from tests.test_training_pipeline import FakeBackend, PRICES
from training_pipeline.smoke import smoke_config
from training_pipeline.orchestrator import Pipeline
from training_pipeline.storage import Tracker

class StepTimingTests(unittest.TestCase):
    def test_completed_batch_timing_precedes_evaluation_and_is_logged(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            config=smoke_config(root/'run',PRICES,root/'ledger.json')
            Pipeline(config,backend=FakeBackend()).run()
            events=[json.loads(x) for x in (Path(config['output'])/'events.jsonl').read_text().splitlines()]
            timings=[e for e in events if e['event']=='training_step_timing']
            self.assertTrue(timings)
            for e in timings:
                self.assertGreaterEqual(e['batch_seconds'],sum(e[k] for k in ('collection_and_grading_seconds','update_seconds','checkpoint_seconds')))
            for i,e in enumerate(events):
                if e['event']=='evaluation' and e['optimizer_step']>0:
                    self.assertTrue(any(x['event']=='training_step_timing' and x['optimizer_step']==e['optimizer_step'] for x in events[:i]))
            wandb=MagicMock();wandb.init.return_value.url='https://example.test/run'
            tracker=Tracker(root/'remote',{'mode':'online'},'timing',wandb_module=wandb)
            fields={k:v for k,v in timings[0].items() if k not in ('event','event_id','run_id')}
            tracker.event('training_step_timing',**fields)
            logged=wandb.init.return_value.log.call_args.args[0]
            self.assertEqual(logged['training/batch_seconds'],fields['batch_seconds'])
            self.assertEqual(logged['attempted_batches'],fields['attempted_batches'])
