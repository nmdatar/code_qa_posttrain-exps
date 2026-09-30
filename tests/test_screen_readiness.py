import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from training_pipeline.storage import Tracker, read
from tests.test_training_pipeline import trajectory
from tests.test_collection_training import CollectionTests


class ReadinessLoggingTests(unittest.TestCase):
    def test_all_answers_include_rewards_and_phase_and_short_artifact_names(self):
        wb = Mock()
        with tempfile.TemporaryDirectory() as root:
            tracker = Tracker(root, {'mode':'online','project':'test','upload_policy':'full'}, 'r'*100, wb)
            tracker.context = {'phase':'evaluation','optimizer_step':3}
            tracker.trajectory(trajectory(0, 1))
            tracker.trajectory(trajectory(1, 0))
            tracker.finish('stopped_initial_zero_batches')
            table = read(Path(root)/'answers.json')
            self.assertEqual(len(table['data']), 2)
            rows = [dict(zip(table['columns'], r)) for r in table['data']]
            self.assertEqual([r['reward'] for r in rows], [1,0])
            self.assertEqual(rows[0]['optimizer_step'], 3)
            self.assertEqual(rows[0]['phase'], 'evaluation')
            wb.init.return_value.finish.assert_called_once_with(exit_code=0)
            p=Path(root)/('x'*100+'.json');p.write_text('{}');tracker.artifact(p,'evaluation')
            self.assertLess(len(wb.Artifact.call_args.args[0]),128)
            self.assertTrue(any(c.kwargs.get('step_metric')=='optimizer_step' for c in wb.init.return_value.define_metric.call_args_list))


class ReadinessProtocolTests(CollectionTests):
    def test_visible_budgets_match_actual_limits_and_do_not_mutate_task(self):
        self.row['public']['budgets']['max_tool_calls']=40
        self.factory.config['limits']['max_tool_calls']=5
        ep=self.episode()
        visible=json.loads(ep.messages[1]['content'])
        self.assertEqual(visible['budgets']['max_tool_calls'],5)
        self.assertEqual(self.row['public']['budgets']['max_tool_calls'],40)
        ep.prepare_generation(1)
        self.assertIn('Submit your answer now',ep.messages[-1]['content'])
        ep.close()

    def test_judge_failure_keeps_diagnostic_and_null_reward(self):
        self.test_invalid_judge_remains_unresolved()
        self.assertEqual(read(self.root/'private/ep.judge-error.json')['detail'],'bad json')
