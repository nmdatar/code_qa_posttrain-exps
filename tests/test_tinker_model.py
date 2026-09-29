"""Offline sampling contract tests: no tokenizer downloads or paid requests."""
import unittest
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

from agent_harness.contracts import FinalAnswer, ModelActionError, ModelError, ToolCall, ToolSpec
from agent_harness.tinker_model import TinkerModel


class TinkerModelTests(unittest.TestCase):
    def model(self, parsed=None, *, clean=True, reason="stop", logprobs=(-.1, -.2)):
        model = TinkerModel(base_model="explicit/model", renderer_name="explicit_renderer",
                            seed=17, input_price_per_million=1, output_price_per_million=2)
        model._types = NS(SamplingParams=lambda **kw: NS(**kw))
        model._tool_call_type = NS(model_validate=lambda c: NS(**c))
        model._renderer = Mock()
        model._renderer.build_generation_prompt.return_value = NS(to_ints=lambda: [1, 2, 3])
        model._renderer.get_stop_sequences.return_value = [99]
        model._renderer.create_conversation_prefix_with_tools.return_value = [{"role": "system", "content": "rendered tools"}]
        model._renderer.parse_response.return_value = (parsed or {"role": "assistant", "content": '{"answer": "yes"}'}, NS(is_clean=clean))
        model._sampling = Mock()
        model._sampling.sample.return_value.result.return_value = NS(sequences=[NS(tokens=[4, 5], logprobs=logprobs, stop_reason=reason)])
        return model

    def test_exact_sampling_provenance_and_usage(self):
        model = self.model()
        result = model.generate([{"role": "user", "content": "question"}], [], 50, timeout_seconds=10)
        self.assertEqual(result.action, FinalAnswer({"answer": "yes"}))
        self.assertEqual(result.conditioning_token_ids, (1, 2, 3))
        self.assertEqual(result.token_ids, (4, 5))
        self.assertEqual(result.logprobs, (-.1, -.2))
        self.assertAlmostEqual(result.usage.cost_usd, .000007)
        args = model._sampling.sample.call_args.kwargs
        self.assertEqual(args["num_samples"], 1)
        self.assertEqual(args["sampling_params"].seed, 17)
        self.assertEqual(args["sampling_params"].stop, [99])
        self.assertEqual(args["sampling_params"].max_tokens, 50)
        self.assertLessEqual(model._sampling.sample.return_value.result.call_args.kwargs["timeout"], 10)

    def test_native_tool_prefix_and_history(self):
        call = NS(function=NS(name="read", arguments='{"path":"a"}'), id=None)
        model = self.model({"content": "", "tool_calls": [call]})
        tools = [ToolSpec("read", "1", "anything", ("read",), {"type": "object"}, description="Read")]
        messages = [{"role": "system", "content": "original"},
                    {"role": "assistant", "content": None, "tool_calls": [{"id": "old", "type": "function", "function": {"name": "read", "arguments": "{}"}}]},
                    {"role": "tool", "content": "data", "name": "read", "tool_call_id": "old"}]
        result = model.generate(messages, tools, 50)
        self.assertIsInstance(result.action, ToolCall)
        self.assertTrue(result.action.call_id.startswith("call_"))
        self.assertEqual(result.action.arguments, {"path": "a"})
        model._renderer.create_conversation_prefix_with_tools.assert_called_once_with(
            [{"name": "read", "description": "Read", "parameters": {"type": "object"}}], system_prompt="original")
        rendered = model._renderer.build_generation_prompt.call_args.args[0]
        self.assertEqual(rendered[1]["content"], "")
        self.assertIsNone(messages[1]["content"])
        self.assertEqual(rendered[2]["tool_call_id"], "old")

    def test_malformed_and_length_keep_training_data(self):
        for kwargs, reason in [({"clean": False}, "agent_error"), ({"reason": "length"}, "budget_exhausted"),
                               ({"parsed": {"content": "ignored", "unparsed_tool_calls": ["bad"]}}, "agent_error")]:
            with self.subTest(kwargs=kwargs):
                model = self.model(**kwargs)
                with self.assertRaises(ModelActionError) as ctx:
                    model.generate([], [], 2)
                self.assertEqual(ctx.exception.termination_reason, reason)
                self.assertEqual(ctx.exception.token_ids, (4, 5))
                self.assertEqual(ctx.exception.conditioning_token_ids, (1, 2, 3))
                self.assertEqual(ctx.exception.logprobs, (-.1, -.2))
                self.assertEqual(ctx.exception.usage.output_tokens, 2)

    def test_does_not_fabricate_missing_or_invalid_logprobs(self):
        for logprobs in [None, [-1], [float("nan"), -1]]:
            self.assertIsNone(self.model(logprobs=logprobs).generate([], [], 2).logprobs)

    def test_thinking_is_not_final_answer(self):
        model = self.model({"content": [{"type": "thinking", "thinking": "internal"}, {"type": "text", "text": "answer"}]})
        self.assertEqual(model.generate([], [], 2).action.value, "answer")

    def test_parallel_calls_and_bad_arguments_are_invalid(self):
        call = NS(function=NS(name="read", arguments="[]"), id="call")
        for calls in [[call], [call, call]]:
            with self.assertRaises(ModelActionError):
                self.model({"tool_calls": calls}).generate([], [], 2)

    def test_errors_do_not_expose_provider_credentials(self):
        model = self.model()
        model._sampling.sample.side_effect = RuntimeError("secret API credential")
        with self.assertRaises(ModelError) as ctx:
            model.generate([], [], 2)
        self.assertNotIn("secret", str(ctx.exception))

    def test_lazy_sdk_initialization_uses_exact_checkpoint(self):
        sdk = Mock()
        renderers = Mock()
        tool_type = Mock()
        modules = {"tinker": sdk, "tinker_cookbook": NS(renderers=renderers),
                   "tinker_cookbook.renderers.base": NS(ToolCall=tool_type)}
        model = TinkerModel(base_model="m", model_path="tinker://run/sampler/checkpoint", renderer_name="r")
        sdk.ServiceClient.assert_not_called()
        with patch.dict("sys.modules", modules):
            model._initialize()
            model._initialize()
        sdk.ServiceClient.assert_called_once_with()
        sampling = sdk.ServiceClient.return_value.create_sampling_client
        sampling.assert_called_once_with(model_path="tinker://run/sampler/checkpoint")
        renderers.get_renderer.assert_called_once_with("r", sampling.return_value.get_tokenizer.return_value)

    def test_explicit_configuration_and_alias(self):
        self.assertEqual(TinkerModel(model="m", renderer_name="r").base_model, "m")
        for config in [{}, {"base_model": "m", "model": "different"}, {"base_model": "m", "temperature": 0},
                       {"base_model": "m", "model_path": "/local"}, {"base_model": "m", "seed": True}]:
            with self.assertRaises(ValueError):
                TinkerModel(renderer_name="r", **config)


if __name__ == "__main__":
    unittest.main()
