import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from agent_harness.artifacts import ArtifactStore
from agent_harness.contracts import (FinalAnswer, ModelResponse, ResearchRequest, RunLimits,
                                     ToolCall, ToolContext, ToolObservation, ToolSpec, Usage)
from agent_harness.models import ModelActionError, ScriptedModel
from agent_harness.registry import ToolInputError, ToolRegistry, ToolScopeError
from agent_harness.runner import AgentRunner


class EchoTool:
    spec = ToolSpec("echo", "1", "text", ("read",),
                    {"type": "object", "properties": {"text": {"type": "string"}},
                     "required": ["text"], "additionalProperties": False}, required_resources=("public",))

    def __init__(self):
        self.seen_resources = None
        self.calls = 0

    def execute(self, arguments, context):
        self.calls += 1
        self.seen_resources = dict(context.resources)
        return ToolObservation("ok", arguments["text"])


class AgentCoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.store = ArtifactStore(self.temp.name)
        self.echo = EchoTool()
        self.registry = ToolRegistry([self.echo])

    def run_actions(self, actions, **kwargs):
        request = ResearchRequest("task", "Find the answer", resources={"public": "safe", "private": "never-policy"}, **kwargs)
        return AgentRunner(ScriptedModel(actions), self.registry, self.store).run(request)

    def events(self, result):
        return [json.loads(line) for line in Path(result.trajectory_path).read_text().splitlines()]

    def test_multiturn_evidence_and_usage_persist_without_private_resources(self):
        result = self.run_actions([
            ModelResponse(ToolCall("echo", {"text": "evidence"}), Usage(10, 2, .001)),
            ModelResponse(FinalAnswer({"answer": "done"}), Usage(20, 3, .002))])
        self.assertEqual(result.termination_reason, "completed")
        self.assertEqual(result.metrics["input_tokens"], 30)
        self.assertEqual(result.metrics["output_tokens"], 5)
        self.assertAlmostEqual(result.metrics["cost_usd"], .003)
        self.assertEqual(self.echo.seen_resources, {"public": "safe"})
        self.assertNotIn("never-policy", Path(result.trajectory_path).read_text())
        observation = next(e for e in self.events(result) if e["kind"] == "tool_observation")
        self.assertEqual(self.store.get(result.episode_id, observation["artifact_id"])["content"], "evidence")
        self.assertEqual(json.loads(Path(result.metrics_path).read_text()), result.metrics)

    def test_scope_is_intersection_and_rechecked_at_dispatch(self):
        for kwargs in ({"allowed_types": frozenset()}, {"allowed_capabilities": frozenset()},
                       {"permitted_tools": frozenset()}, {"resources": {}}):
            fields = {"resources": {"public": "safe"}, **kwargs}
            request = ResearchRequest("t", "q", **fields)
            self.assertEqual(self.registry.available(request), [])
            with self.assertRaises(ToolScopeError):
                self.registry.dispatch(ToolCall("echo", {"text": "x"}), ToolContext({}, self.store, "none"), request)
        result = self.run_actions([ToolCall("echo", {"text": "no"}), FinalAnswer("done")], permitted_tools=frozenset())
        self.assertEqual(self.echo.calls, 0)
        self.assertEqual(result.metrics["tool_calls"], 1)
        self.assertTrue(result.metrics["integrity_violations"])

    def test_invalid_arguments_are_observed_and_counted(self):
        result = self.run_actions([ToolCall("echo", {"text": 42}), FinalAnswer("recovered")])
        self.assertEqual(result.termination_reason, "completed")
        self.assertEqual(result.metrics["tool_calls"], 1)
        self.assertEqual(self.echo.calls, 0)
        self.assertEqual(next(e for e in self.events(result) if e["kind"] == "tool_observation")["observation"]["status"], "error")

    def test_duplicate_and_unsupported_schema_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.registry.register(EchoTool())
        bad = EchoTool()
        bad.spec = ToolSpec("bad", "1", "general", (), {"type": "object", "oneOf": []})
        with self.assertRaisesRegex(ValueError, "unsupported"):
            self.registry.register(bad)

    def test_explicit_entry_points_only(self):
        class Entry:
            name = "trusted"
            def load(self):
                return EchoTool
        class NeverLoad:
            name = "not-configured"
            def load(self):
                raise AssertionError("Unrequested plugins must never load")
        registry = ToolRegistry()
        with patch("agent_harness.registry.metadata.entry_points", return_value=[Entry(), NeverLoad()]):
            registry.load_plugins(["trusted"])
            with self.assertRaises(ValueError):
                registry.load_plugins(["missing"])
        self.assertEqual([s.name for s in registry.available(ResearchRequest("t", "q", {"public": "yes"}))], ["echo"])

    def test_artifacts_cannot_cross_episodes_or_be_overwritten(self):
        self.store.create_episode("one")
        self.store.create_episode("two")
        artifact = self.store.put("one", {"answer": 1})
        with self.assertRaises(FileNotFoundError):
            self.store.get("two", artifact)
        with self.assertRaises(ValueError):
            self.store.get("one", "../metrics")
        with self.assertRaises(FileExistsError):
            self.store.create_episode("one")
        path = self.store.episode_dir("one") / "artifacts" / (artifact + ".json")
        path.write_text("{}")
        with self.assertRaisesRegex(ValueError, "integrity"):
            self.store.get("one", artifact)

    def test_steps_tool_calls_and_token_limits(self):
        cases = [(RunLimits(max_steps=1), [ToolCall("echo", {"text": "x"})]),
                 (RunLimits(max_tool_calls=0), [ToolCall("echo", {"text": "x"})]),
                 (RunLimits(max_output_tokens=1), [ModelResponse(FinalAnswer("too many"), Usage(1, 2, 0))])]
        for limits, actions in cases:
            result = self.run_actions(actions, limits=limits)
            self.assertEqual(result.termination_reason, "budget_exhausted")
            self.assertIsNone(result.submission)

    def test_missing_usage_remains_unknown_and_reserves_remaining_budget(self):
        result = self.run_actions([ModelResponse(ToolCall("echo", {"text": "x"})), FinalAnswer("never")])
        self.assertEqual(result.termination_reason, "budget_exhausted")
        self.assertFalse(result.metrics["usage_complete"])
        self.assertIsNone(result.metrics["output_tokens"])
        self.assertEqual(result.metrics["steps"], 1)

    def test_policy_parse_failure_preserves_usage_and_is_not_infrastructure(self):
        class BrokenPolicy:
            def generate(self, *args, **kwargs):
                raise ModelActionError("malformed tool call", Usage(5, 3, .01))
        result = AgentRunner(BrokenPolicy(), self.registry, self.store).run(ResearchRequest("t", "q"))
        self.assertEqual(result.termination_reason, "agent_error")
        self.assertEqual(result.metrics["output_tokens"], 3)
        self.assertTrue(result.metrics["usage_complete"])

    def test_transport_failure_marks_usage_unknown(self):
        class TransportFailure:
            def generate(self, *args, **kwargs):
                raise OSError("offline")
        result = AgentRunner(TransportFailure(), self.registry, self.store).run(ResearchRequest("t", "q"))
        self.assertEqual(result.termination_reason, "infrastructure_error")
        self.assertIsNone(result.metrics["input_tokens"])
        self.assertIsNone(result.metrics["cost_usd"])

    def test_tool_deadline_interrupts_and_records_timeout(self):
        slow = EchoTool()
        slow.spec = ToolSpec("slow", "1", "text", (), {"type": "object"}, timeout_seconds=.01)
        slow.execute = lambda arguments, context: time.sleep(2)
        self.registry.register(slow)
        started = time.monotonic()
        result = self.run_actions([ToolCall("slow", {}), FinalAnswer("handled")])
        self.assertLess(time.monotonic() - started, 1)
        self.assertEqual(result.termination_reason, "completed")
        self.assertEqual(next(e for e in self.events(result) if e["kind"] == "tool_observation")["observation"]["status"], "timeout")

    def test_invalid_tool_output_is_infrastructure_failure(self):
        self.echo.spec = ToolSpec("echo", "1", "text", (), {"type": "object"}, output_schema={"type": "integer"})
        self.echo.execute = lambda arguments, context: ToolObservation("ok", "wrong")
        result = self.run_actions([ToolCall("echo", {})])
        self.assertEqual(result.termination_reason, "infrastructure_error")
        self.assertEqual(result.metrics["tool_calls"], 1)
        self.assertGreater(result.metrics["tool_seconds"], 0)

    def test_output_validation_and_sampling_alignment(self):
        runner = AgentRunner(ScriptedModel([FinalAnswer("bad")]), self.registry, self.store,
                             output_validator=lambda value: (_ for _ in ()).throw(ValueError("invalid output")))
        result = runner.run(ResearchRequest("t", "q"))
        self.assertEqual(result.termination_reason, "agent_error")
        result = self.run_actions([ModelResponse(FinalAnswer("x"), Usage(1, 1, 0), (1,), ())])
        self.assertEqual(result.termination_reason, "infrastructure_error")

    def test_malformed_tool_fields_are_agent_errors_with_usage(self):
        for action in (ToolCall([], {}), ToolCall("echo", []), ToolCall("echo", {}, [])):
            result = self.run_actions([ModelResponse(action, Usage(2, 3, .01))])
            self.assertEqual(result.termination_reason, "agent_error")
            self.assertEqual(result.metrics["output_tokens"], 3)

    def test_policy_sees_shrinking_budgets_and_can_finalize_without_tools(self):
        class ObservingModel(ScriptedModel):
            def __init__(self):
                super().__init__([ModelResponse(ToolCall("echo", {"text": "x"}), Usage(1, 3, 0)),
                                  ModelResponse(FinalAnswer("done"), Usage(1, 2, 0))])
                self.calls = []
            def generate(self, messages, tools, max_output_tokens, timeout_seconds=None):
                budget = json.loads(messages[0]["content"].split("Remaining budgets: ")[1])
                self.calls.append((budget, [tool.name for tool in tools]))
                return super().generate(messages, tools, max_output_tokens, timeout_seconds=timeout_seconds)
        model = ObservingModel()
        result = AgentRunner(model, self.registry, self.store).run(ResearchRequest(
            "t", "q", {"public": "safe"}, limits=RunLimits(max_steps=3, max_tool_calls=1, max_output_tokens=10)))
        self.assertEqual(result.termination_reason, "completed")
        self.assertEqual([call[0]["tool_calls"] for call in model.calls], [1, 0])
        self.assertEqual([call[0]["model_steps"] for call in model.calls], [3, 2])
        self.assertEqual([call[0]["output_tokens"] for call in model.calls], [10, 7])
        self.assertEqual(model.calls[1][1], [])
        self.assertLessEqual(model.calls[1][0]["wall_time_seconds"], model.calls[0][0]["wall_time_seconds"])

    def test_context_compaction_archives_whole_pairs(self):
        calls = [ToolCall("echo", {"text": str(i) + "x" * 500}) for i in range(4)]
        result = self.run_actions(calls + [FinalAnswer("done")], limits=RunLimits(max_context_chars=3000))
        self.assertEqual(result.termination_reason, "completed")
        events = self.events(result)
        self.assertTrue(any(e["kind"] == "context_compacted" for e in events))
        for event in events:
            if event["kind"] == "model_request":
                messages = event["messages"]
                for i, message in enumerate(messages):
                    if message["role"] == "tool":
                        self.assertEqual(messages[i - 1]["tool_calls"][0]["id"], message["tool_call_id"])


if __name__ == "__main__":
    unittest.main()
