import json
from pathlib import Path
import tempfile
import unittest

from agent_harness import ArtifactStore, FinalAnswer, ModelResponse, Usage, ToolRegistry
from agent_harness.code_tools import GitRepository, code_understanding_tools
from agent_harness.models import ScriptedModel
from agent_harness.contracts import ToolCall
from qa_eval.grading import evaluate
from qa_eval.security import seal
from agent_harness.qa_adapter import run_qa_episode, request_from_task, submission_validator
from qa_eval.demo import fixture, KEY
from qa_eval.security import unseal


class FinalModel:
    def __init__(self, value, usage=Usage(10, 20, 0.0)):
        self.value, self.usage, self.messages = value, usage, []

    def generate(self, messages, tools, max_output_tokens, timeout_seconds=None):
        self.messages = messages
        return ModelResponse(FinalAnswer(self.value), self.usage)


class QABridgeTests(unittest.TestCase):
    def test_public_request_never_serializes_private_claims(self):
        with tempfile.TemporaryDirectory() as tmp:
            task, answer, *_ = fixture(Path(tmp) / 'repo')
            task['critical_errors'] = ['PRIVATE_SENTINEL']
            request = request_from_task(task, {})
            self.assertNotIn('PRIVATE_SENTINEL', request.question)
            self.assertNotIn('claims', request.question)
            self.assertLessEqual(request.limits.max_tool_calls, task['budgets']['max_tool_calls'])

    def test_signs_only_host_run_and_binds_submission(self):
        with tempfile.TemporaryDirectory() as tmp:
            task, answer, *_ = fixture(Path(tmp) / 'repo')
            model = FinalModel(answer)
            result = run_qa_episode(task=task, resources={"repository": GitRepository(Path(tmp) / "repo", task["repository"]["commit"])}, model=model,
                registry=ToolRegistry(), artifacts=ArtifactStore(Path(tmp) / 'runs'),
                experiment_id='test', key=KEY)
            self.assertIsNone(result.unresolved_reason)
            metrics = unseal(result.metrics_envelope, 'EpisodeMetrics', KEY)
            self.assertEqual(metrics['input_tokens'], 10)
            self.assertEqual(metrics['output_tokens'], 20)
            self.assertEqual(metrics['tool_calls'], [])
            self.assertEqual(metrics['probes'], [])
            self.assertNotIn(KEY.decode(errors='ignore'), json.dumps(model.messages))

    def test_unknown_cost_is_not_fabricated(self):
        with tempfile.TemporaryDirectory() as tmp:
            task, answer, *_ = fixture(Path(tmp) / 'repo')
            result = run_qa_episode(task=task, resources={"repository": GitRepository(Path(tmp) / "repo", task["repository"]["commit"])}, model=FinalModel(answer, Usage(10, 20)),
                registry=ToolRegistry(), artifacts=ArtifactStore(Path(tmp) / 'runs'),
                experiment_id='test', key=KEY)
            self.assertIsNone(result.metrics_envelope)
            self.assertIn('unavailable', result.unresolved_reason)

    def test_research_to_existing_grader_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task, answer, config, _, semantic = fixture(root / 'repo')
            result = run_qa_episode(task=task,
                resources={'repository': GitRepository(root / 'repo', task['repository']['commit'])},
                model=ScriptedModel([ToolCall('read_file', {'path': 'executor.py', 'line_count': 2}), FinalAnswer(answer)]),
                registry=ToolRegistry(code_understanding_tools()), artifacts=ArtifactStore(root / 'runs'),
                experiment_id=config['id'], key=KEY)
            report, _ = evaluate(task, result.submission, result.metrics_envelope, config,
                root / 'repo', KEY, semantic_envelope=seal('SemanticAssessment', semantic, KEY))
            self.assertEqual(report['tier'], 'accepted')

    def test_invalid_answer_remains_gradable_policy_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task, _, *_ = fixture(root / 'repo')
            result = run_qa_episode(task=task,
                resources={'repository': GitRepository(root / 'repo', task['repository']['commit'])},
                model=FinalModel({'bad': 'submission'}), registry=ToolRegistry(),
                artifacts=ArtifactStore(root / 'runs'), experiment_id='test', key=KEY)
            self.assertEqual(result.episode.termination_reason, 'agent_error')
            self.assertEqual(result.submission['text'], '')
            self.assertIsNotNone(result.metrics_envelope)

    def test_mismatched_snapshot_fails_before_sampling(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task, answer, *_ = fixture(root / 'repo')
            repo = GitRepository(root / 'repo', task['repository']['commit'])
            task['repository']['commit'] = '0' * 40
            with self.assertRaisesRegex(ValueError, 'commit'):
                run_qa_episode(task=task, resources={'repository': repo}, model=FinalModel(answer),
                    registry=ToolRegistry(), artifacts=ArtifactStore(root / 'runs'),
                    experiment_id='test', key=KEY)

    def test_submission_cannot_supply_trusted_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            task, answer, *_ = fixture(Path(tmp) / 'repo')
            answer['input_tokens'] = 0
            with self.assertRaises(ValueError):
                submission_validator(task['id'])(answer)

    def test_task_id_and_size_are_checked(self):
        with tempfile.TemporaryDirectory() as tmp:
            task, answer, *_ = fixture(Path(tmp) / 'repo')
            with self.assertRaises(ValueError):
                submission_validator('other')(answer)
            with self.assertRaises(ValueError):
                submission_validator(task['id'], 1)(answer)


if __name__ == '__main__':
    unittest.main()
