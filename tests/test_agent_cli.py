import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from agent_harness.cli import main
from agent_harness.contracts import FinalAnswer
from agent_harness.models import ScriptedModel
from qa_eval.demo import fixture


class CLITests(unittest.TestCase):
    def invoke(self, args):
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(io.StringIO()):
            code = main(args)
        return code, stream.getvalue()

    def test_offline_demo_runs_list_read_and_final(self):
        with tempfile.TemporaryDirectory() as tmp, patch('urllib.request.OpenerDirector.open', side_effect=AssertionError('network forbidden')):
            code, output = self.invoke(['demo', '--output', tmp, '--episode-id', 'demo'])
            self.assertEqual(code, 0)
            summary = json.loads(output)
            result = json.loads(Path(summary['result']).read_text())
            self.assertEqual(result['metrics']['tool_names'], ['list_files', 'read_file'])
            self.assertEqual(result['metrics']['cost_usd'], 0)
            self.assertEqual(result['termination_reason'], 'completed')
            self.assertTrue(Path(summary['trajectory']).exists())

    def test_existing_episode_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(self.invoke(['demo', '--output', tmp, '--episode-id', 'same'])[0], 0)
            before = (Path(tmp) / 'same' / 'result.json').read_bytes()
            self.assertEqual(self.invoke(['demo', '--output', tmp, '--episode-id', 'same'])[0], 2)
            self.assertEqual((Path(tmp) / 'same' / 'result.json').read_bytes(), before)

    def test_task_commit_and_submission_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task, answer, *_ = fixture(root / 'repo')
            task_path = root / 'task.json'
            task_path.write_text(json.dumps(task))
            args = ['run', '--repo', str(root / 'repo'), '--commit', task['repository']['commit'],
                    '--output', str(root / 'runs'), '--task', str(task_path),
                    '--model', 'fake', '--base-url', 'https://example.invalid/v1']
            with patch('agent_harness.cli.ChatCompletionsModel', return_value=ScriptedModel([FinalAnswer(answer)])):
                code, output = self.invoke(args)
            self.assertEqual(code, 0)
            result = json.loads(Path(json.loads(output)['result']).read_text())
            self.assertEqual(result['submission'], answer)
            task['budgets']['max_submission_bytes'] = 10
            task_path.write_text(json.dumps(task))
            with patch('agent_harness.cli.ChatCompletionsModel', return_value=ScriptedModel([FinalAnswer(answer)])):
                code, output = self.invoke(args)
            self.assertEqual(code, 1)
            self.assertEqual(json.loads(output)['termination_reason'], 'agent_error')

    def test_task_commit_mismatch_fails_before_model_call(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task, answer, *_ = fixture(root / 'repo')
            commit = task['repository']['commit']
            task['repository']['commit'] = 'a' * 40
            (root / 'task.json').write_text(json.dumps(task))
            with patch('agent_harness.cli.ChatCompletionsModel') as model:
                code, output = self.invoke(['run', '--repo', str(root / 'repo'), '--commit', commit,
                    '--output', str(root / 'runs'), '--task', str(root / 'task.json'),
                    '--model', 'fake', '--base-url', 'https://example.invalid/v1'])
            self.assertEqual(code, 2)
            model.return_value.generate.assert_not_called()
