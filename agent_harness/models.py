"""Provider adapters with bounded responses and no implicit paid retries."""
from __future__ import annotations
from collections import deque
import json
import math
import urllib.error
import urllib.parse
import urllib.request
from .contracts import ModelError, ModelActionError, FinalAnswer, ModelResponse, ToolCall, ToolCallBatch, ToolSpec, Usage

class _NonFiniteJSON(ValueError):
    pass


def _reject_constant(value):
    raise _NonFiniteJSON('Nonfinite JSON number')


def _finite_float(value):
    number = float(value)
    if not math.isfinite(number):
        raise _NonFiniteJSON('Nonfinite JSON number')
    return number


def _loads(value):
    return json.loads(value, parse_constant=_reject_constant, parse_float=_finite_float)


class ScriptedModel:
    """Deterministic offline actions, not a trained policy."""
    def __init__(self, actions):
        self._actions = deque(actions)

    def generate(self, messages, tools, max_output_tokens, *, timeout_seconds=None):
        if not self._actions:
            raise ModelError("Scripted actions exhausted")
        action = self._actions.popleft()
        if isinstance(action, ModelResponse):
            return action
        return ModelResponse(action=action, usage=Usage(input_tokens=0, output_tokens=0, cost_usd=0.0))

class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None  # Never forward credentials to a redirected destination.

class ChatCompletionsModel:
    """Native tool calling; base_url includes the API prefix, e.g. /v1.

    Missing usage and cost remain unknown. Exact sampled token IDs/logprobs
    are not available from this adapter and must not be inferred for training.
    """
    def __init__(self, *, model, base_url, api_key=None, timeout_seconds=120,
                 max_response_bytes=4 * 1024 * 1024, max_parallel_tool_calls=1,
                 input_price_per_million=None, output_price_per_million=None):
        parsed = urllib.parse.urlsplit(base_url)
        if (parsed.scheme not in {'http', 'https'} or not parsed.hostname
                or parsed.username or parsed.password or parsed.query or parsed.fragment):
            raise ValueError('base_url must be an HTTP(S) API URL without credentials or query')
        if parsed.scheme == 'http' and parsed.hostname not in {'localhost', '127.0.0.1', '::1'}:
            raise ValueError('Remote model endpoints must use HTTPS')
        if not model or not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError('A model and positive finite timeout are required')
        if type(max_response_bytes) is not int or max_response_bytes <= 0:
            raise ValueError('max_response_bytes must be a positive integer')
        for price in (input_price_per_million, output_price_per_million):
            if price is not None and (not math.isfinite(price) or price < 0):
                raise ValueError('Token prices must be finite and nonnegative')
        if type(max_parallel_tool_calls) is not int or not 1 <= max_parallel_tool_calls <= 8:
            raise ValueError("max_parallel_tool_calls must be between 1 and 8")
        self.max_parallel_tool_calls = max_parallel_tool_calls
        self.model, self.base_url, self._api_key = model, base_url.rstrip('/'), api_key
        self.timeout_seconds, self.max_response_bytes = timeout_seconds, max_response_bytes
        self.input_price, self.output_price = input_price_per_million, output_price_per_million

    def generate(self, messages, tools, max_output_tokens, *, timeout_seconds=None):
        if type(max_output_tokens) is not int or max_output_tokens <= 0:
            raise ValueError('max_output_tokens must be a positive integer')
        timeout = self.timeout_seconds if timeout_seconds is None else min(self.timeout_seconds, timeout_seconds)
        if not math.isfinite(timeout) or timeout <= 0:
            raise ModelError('Model call deadline exhausted')
        body = dict(model=self.model, messages=messages, max_tokens=max_output_tokens)
        if tools:
            body['tools'] = [dict(type='function', function=dict(name=t.name, description=t.description,
                                                               parameters=t.input_schema)) for t in tools]
            body['parallel_tool_calls'] = self.max_parallel_tool_calls > 1
        headers = {'Content-Type': 'application/json'}
        if self._api_key:
            headers['Authorization'] = 'Bearer ' + self._api_key
        request = urllib.request.Request(self.base_url + '/chat/completions', data=json.dumps(body).encode(), headers=headers)
        try:
            with urllib.request.build_opener(_NoRedirect()).open(request, timeout=timeout) as response:
                raw = response.read(self.max_response_bytes + 1)
            if len(raw) > self.max_response_bytes:
                raise ModelError('Model response exceeds byte limit; usage unknown')
            payload = _loads(raw)
        except ModelError:
            raise
        except urllib.error.HTTPError as exc:
            raise ModelError(f'Model endpoint returned HTTP {exc.code}; usage unknown') from None
        except (OSError, ValueError, urllib.error.URLError):
            raise ModelError('Model request failed or returned invalid JSON; usage unknown') from None
        try:
            return self._parse(payload)
        except (ValueError, TypeError, KeyError, IndexError, AttributeError):
            raise ModelError('Model response has an invalid action or usage; usage unknown') from None

    def _parse(self, payload):
        usage = self._usage(payload)
        choices = payload['choices']
        if not isinstance(choices, list) or len(choices) != 1:
            raise ValueError('Expected one choice')
        choice = choices[0]
        if choice.get('finish_reason') == 'length':
            raise ModelActionError('Model reached its output limit', usage, 'budget_exhausted')
        if choice.get('finish_reason') not in {'stop', 'tool_calls'}:
            raise ModelActionError('Model did not provide a usable action', usage)
        message = choice['message']
        try:
            action = self._action(message)
        except (ValueError, TypeError, KeyError, IndexError, AttributeError):
            raise ModelActionError('Model returned an invalid action', usage) from None
        return ModelResponse(action=action, usage=usage)

    def _action(self, message):
        calls = message.get('tool_calls') or []
        if calls:
            if not isinstance(calls, list) or not 1 <= len(calls) <= self.max_parallel_tool_calls:
                raise ValueError('Too many tool actions')
            actions = []
            for call in calls:
                if call.get('type') != 'function':
                    raise ValueError('Expected function action')
                function = call['function']
                arguments = _loads(function['arguments'])
                if not isinstance(arguments, dict) or not isinstance(function['name'], str) or not function['name']:
                    raise ValueError('Invalid tool action')
                if not isinstance(call.get('id'), str) or not call['id']:
                    raise ValueError('Missing call ID')
                actions.append(ToolCall(function['name'], arguments, call['id']))
            if len({a.call_id for a in actions}) != len(actions):
                raise ValueError('Duplicate call ID')
            action = actions[0] if len(actions) == 1 else ToolCallBatch(tuple(actions))
        else:
            content = message['content']
            if not isinstance(content, str) or not content.strip():
                raise ValueError('Missing final content')
            try:
                value = _loads(content)
            except _NonFiniteJSON:
                raise
            except ValueError:
                value = content
            action = FinalAnswer(value=value)
        return action

    def _usage(self, payload):
        reported = payload.get('usage') or {}
        def count(key):
            value = reported.get(key)
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError('Invalid usage')
            return value
        input_tokens, output_tokens = count('prompt_tokens'), count('completion_tokens')
        cost = None
        if all(x is not None for x in (input_tokens, output_tokens, self.input_price, self.output_price)):
            cost = (input_tokens * self.input_price + output_tokens * self.output_price) / 1_000_000
        return Usage(input_tokens=input_tokens, output_tokens=output_tokens, cost_usd=cost)
