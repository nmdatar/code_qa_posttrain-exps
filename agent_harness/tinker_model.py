"""Optional Tinker sampling with exact per-turn training provenance.

Validated against tinker==0.30.4 and tinker-cookbook==0.5.7. Credentials are
read by the SDK from TINKER_API_KEY, never from episode configuration.
"""
from __future__ import annotations
from copy import deepcopy
import math
import time
import uuid
from .contracts import FinalAnswer, ModelActionError, ModelError, ModelResponse, ToolCall, ToolCallBatch, Usage
from .models import _loads, _NonFiniteJSON


class TinkerModel:
    def __init__(self, *, base_model=None, model=None, model_path=None, renderer_name,
                 temperature=1.0, seed=None, input_price_per_million=None,
                 output_price_per_million=None, max_parallel_tool_calls=1):
        if model is not None and base_model is not None and model != base_model:
            raise ValueError("model and base_model disagree")
        base_model = base_model or model
        if not isinstance(base_model, str) or not base_model.strip():
            raise ValueError("An explicit base_model is required")
        if not isinstance(renderer_name, str) or not renderer_name.strip():
            raise ValueError("An explicit renderer_name is required")
        if model_path is not None and (not isinstance(model_path, str) or not model_path.startswith("tinker://")):
            raise ValueError("model_path must be a Tinker sampling checkpoint reference")
        if isinstance(temperature, bool) or not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be positive and finite for policy sampling")
        if seed is not None and (type(seed) is not int or seed < 0):
            raise ValueError("seed must be a nonnegative integer")
        for price in (input_price_per_million, output_price_per_million):
            if price is not None and (isinstance(price, bool) or not math.isfinite(price) or price < 0):
                raise ValueError("Token prices must be finite and nonnegative")
        if type(max_parallel_tool_calls) is not int or not 1 <= max_parallel_tool_calls <= 8:
            raise ValueError("max_parallel_tool_calls must be between 1 and 8")
        self.max_parallel_tool_calls = max_parallel_tool_calls
        self.base_model, self.model_path, self.renderer_name = base_model, model_path, renderer_name
        self.model_id = model_path or base_model
        self.temperature, self.seed = temperature, seed
        self.input_price, self.output_price = input_price_per_million, output_price_per_million
        self._sampling = self._renderer = self._types = self._tool_call_type = None

    def _initialize(self):
        if self._sampling is not None:
            return
        try:
            import tinker
            from tinker_cookbook import renderers
            from tinker_cookbook.renderers.base import ToolCall as RendererToolCall
        except ImportError:
            raise ModelError("Tinker sampling requires the pinned tinker optional dependencies") from None
        try:
            service = tinker.ServiceClient()
            # A checkpoint identifies its base model; never fall back to base weights.
            kwargs = {"model_path": self.model_path} if self.model_path else {"base_model": self.base_model}
            sampling = service.create_sampling_client(**kwargs)
            renderer = renderers.get_renderer(self.renderer_name, sampling.get_tokenizer())
        except Exception:
            raise ModelError("Tinker client or renderer initialization failed") from None
        self._service = service
        self._sampling, self._renderer = sampling, renderer
        self._types, self._tool_call_type = tinker.types, RendererToolCall

    def _prompt(self, messages, tools):
        converted = deepcopy(messages)
        for message in converted:
            if message.get("content") is None:
                message["content"] = ""
            if message.get("tool_calls"):
                message["tool_calls"] = [self._tool_call_type.model_validate(c) for c in message["tool_calls"]]
        if tools:
            system = ""
            if converted and converted[0]["role"] == "system":
                system = converted.pop(0)["content"]
            specs = [{"name": tool.name, "description": tool.description,
                      "parameters": tool.input_schema} for tool in tools]
            converted = self._renderer.create_conversation_prefix_with_tools(specs, system_prompt=system) + converted
        return self._renderer.build_generation_prompt(converted)

    def generate(self, messages, tools, max_output_tokens, timeout_seconds=None):
        if type(max_output_tokens) is not int or max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be a positive integer")
        if timeout_seconds is not None and (not math.isfinite(timeout_seconds) or timeout_seconds <= 0):
            raise ModelError("Model call deadline exhausted")
        started = time.monotonic()
        self._initialize()
        try:
            prompt = self._prompt(messages, tools)
            conditioning = tuple(prompt.to_ints())
            params = dict(max_tokens=max_output_tokens, temperature=self.temperature,
                          stop=self._renderer.get_stop_sequences())
            if self.seed is not None:
                params["seed"] = self.seed
            remaining = None if timeout_seconds is None else timeout_seconds - (time.monotonic() - started)
            if remaining is not None and remaining <= 0:
                raise ModelError("Model call deadline exhausted before sampling")
            future = self._sampling.sample(prompt=prompt, num_samples=1,
                                           sampling_params=self._types.SamplingParams(**params))
            response = future.result(timeout=remaining)
            if len(response.sequences) != 1:
                raise ValueError("Expected one sampled sequence")
            sequence = response.sequences[0]
            tokens = tuple(sequence.tokens)
            if not all(type(token) is int and token >= 0 for token in tokens):
                raise ValueError("Invalid sampled token IDs")
        except ModelError:
            raise
        except Exception:
            # Provider errors may contain headers: never persist their raw text.
            raise ModelError("Tinker sampling failed; usage unknown") from None
        cost = None
        if self.input_price is not None and self.output_price is not None:
            cost = (len(conditioning) * self.input_price + len(tokens) * self.output_price) / 1_000_000
        usage = Usage(input_tokens=len(conditioning), output_tokens=len(tokens), cost_usd=cost)
        raw_logprobs = sequence.logprobs
        logprobs = None if raw_logprobs is None else tuple(raw_logprobs)
        if logprobs is not None and (len(logprobs) != len(tokens) or any(
                isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in logprobs)):
            logprobs = None  # Never fabricate or misalign policy likelihoods.
        metadata = dict(conditioning_token_ids=conditioning, token_ids=tokens, logprobs=logprobs)
        if sequence.stop_reason == "length":
            raise ModelActionError("Model reached its output limit", usage, "budget_exhausted", **metadata)
        try:
            parsed, termination = self._renderer.parse_response(list(tokens))
            # MALFORMED is truthy: inspect the official enum property explicitly.
            if not termination.is_clean or parsed.get("unparsed_tool_calls"):
                raise ValueError("Malformed rendered action")
            calls = parsed.get("tool_calls") or []
            if calls:
                if not 1 <= len(calls) <= self.max_parallel_tool_calls:
                    raise ValueError("Too many tool actions")
                actions = []
                for call in calls:
                    arguments = _loads(call.function.arguments)
                    if not isinstance(arguments, dict) or not isinstance(call.function.name, str) or not call.function.name:
                        raise ValueError("Invalid tool action")
                    if call.id and not isinstance(call.id, str):
                        raise ValueError("Invalid call ID")
                    actions.append(ToolCall(call.function.name, arguments, call.id or "call_" + uuid.uuid4().hex))
                if len({a.call_id for a in actions}) != len(actions):
                    raise ValueError("Duplicate call ID")
                action = actions[0] if len(actions) == 1 else ToolCallBatch(tuple(actions))
            else:
                content = parsed.get("content", "")
                # Thinking stays in sampled tokens, but not in the final submission.
                if isinstance(content, list):
                    content = "".join(part["text"] for part in content if part["type"] == "text")
                if not isinstance(content, str) or not content.strip():
                    raise ValueError("Empty final answer")
                try:
                    value = _loads(content)
                except _NonFiniteJSON:
                    raise
                except ValueError:
                    value = content
                action = FinalAnswer(value)
        except Exception:
            raise ModelActionError("Model returned an invalid action", usage, **metadata) from None
        return ModelResponse(action, usage=usage, **metadata)
