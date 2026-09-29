"""Independent security regression tests for generic dispatch boundaries."""
import json
from pathlib import Path
import tempfile
import unittest

from agent_harness.artifacts import ArtifactStore
from agent_harness.contracts import ResearchRequest, ToolCall, ToolContext, ToolObservation, ToolSpec
from agent_harness.registry import ToolRegistry


class ObservingTool:
    spec = ToolSpec(name="inspect", version="1", type="generic", capabilities=("read",),
                    input_schema={"type": "object", "properties": {}, "additionalProperties": False},
                    required_resources=("public",), max_output_bytes=512)

    def __init__(self, error=None):
        self.seen = None
        self.error = error

    def execute(self, arguments, context):
        self.seen = dict(context.resources)
        return ToolObservation("error" if self.error else "ok", "response", error=self.error)


class DispatchSecurityTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.artifacts = ArtifactStore(Path(self.temporary.name))
        self.artifacts.create_episode("one")
        self.resources = {"public": "source", "private": "grading secret"}
        self.request = ResearchRequest("task", "question", resources=self.resources)
        self.context = ToolContext(self.resources, self.artifacts, "one")

    def test_tool_gets_only_declared_resource_handles(self):
        tool = ObservingTool()
        ToolRegistry([tool]).dispatch(ToolCall("inspect", {}), self.context, self.request)
        self.assertEqual(tool.seen, {"public": "source"})

    def test_oversized_error_is_bounded_like_content(self):
        from dataclasses import asdict
        tool = ObservingTool(error="x" * 10000)
        observation = ToolRegistry([tool]).dispatch(ToolCall("inspect", {}), self.context, self.request)
        self.assertLessEqual(len(json.dumps(asdict(observation)).encode()), tool.spec.max_output_bytes)
        self.assertTrue(observation.truncated)


class PolicyFailureTests(unittest.TestCase):
    def test_nonfinite_action_is_policy_failure_with_usage_retained(self):
        from agent_harness.contracts import ModelResponse, Usage
        from agent_harness.models import ScriptedModel
        from agent_harness.runner import AgentRunner
        with tempfile.TemporaryDirectory() as temporary:
            model = ScriptedModel([ModelResponse(ToolCall("inspect", {"x": float("nan")}),
                                                 usage=Usage(2, 3, .01))])
            result = AgentRunner(model, ToolRegistry(), ArtifactStore(temporary)).run(
                ResearchRequest("task", "question"))
            self.assertEqual(result.termination_reason, "agent_error")
            self.assertEqual(result.metrics["output_tokens"], 3)
            self.assertEqual(result.metrics["cost_usd"], .01)
