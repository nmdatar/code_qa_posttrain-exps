import io
import json
import unittest
from unittest.mock import patch
import urllib.error
from agent_harness.contracts import FinalAnswer, ToolCall, ToolSpec
from agent_harness.models import ChatCompletionsModel, ScriptedModel, ModelError, ModelActionError


def payload(message=None, reason='stop', usage=None):
    value = {'choices': [{'finish_reason': reason, 'message': message or {'content': '{"answer":"done"}'}}]}
    if usage is not None:
        value['usage'] = usage
    return value


class ModelTests(unittest.TestCase):
    def model(self, **kwargs):
        return ChatCompletionsModel(model='test-v1', base_url='https://example.invalid/v1', **kwargs)

    def generate(self, data, **kwargs):
        with patch('agent_harness.models.urllib.request.build_opener') as opener:
            opener.return_value.open.return_value = io.BytesIO(json.dumps(data).encode())
            return self.model(**kwargs).generate([], [], 20)

    def test_scripted_usage_and_exhaustion(self):
        model = ScriptedModel([FinalAnswer('done')])
        result = model.generate([], [], 10)
        self.assertEqual(result.usage.cost_usd, 0)
        with self.assertRaises(ModelError):
            model.generate([], [], 10)

    def test_missing_usage_and_cost_are_unknown(self):
        result = self.generate(payload())
        self.assertIsNone(result.usage.input_tokens)
        self.assertIsNone(result.usage.output_tokens)
        self.assertIsNone(result.usage.cost_usd)
        self.assertIsNone(result.token_ids)

    def test_native_tool_call_and_usage(self):
        call = {'id': 'one', 'type': 'function', 'function': {'name': 'read', 'arguments': '{"path":"a"}'}}
        result = self.generate(payload({'tool_calls': [call]}, 'tool_calls',
                                       {'prompt_tokens': 100, 'completion_tokens': 50}),
                               input_price_per_million=2, output_price_per_million=4)
        self.assertEqual(result.action, ToolCall('read', {'path': 'a'}, 'one'))
        self.assertAlmostEqual(result.usage.cost_usd, .0004)

    def test_malformed_actions_preserve_usage(self):
        for arguments in ('not json', '[]', '{"x":NaN}', '{"x":Infinity}', '{"x":1e999}'):
            call = {'id': 'one', 'type': 'function', 'function': {'name': 'read', 'arguments': arguments}}
            with self.assertRaises(ModelActionError) as error:
                self.generate(payload({'tool_calls': [call]}, 'tool_calls', {'prompt_tokens': 2, 'completion_tokens': 3}))
            self.assertEqual(error.exception.usage.output_tokens, 3)
            self.assertEqual(error.exception.termination_reason, 'agent_error')

    def test_nonfinite_final_json_is_policy_failure(self):
        for content in ('{"answer":NaN}', '{"answer":1e999}'):
            with self.assertRaises(ModelActionError) as error:
                self.generate(payload({'content': content}, usage={'completion_tokens': 3}))
            self.assertEqual(error.exception.usage.output_tokens, 3)

    def test_output_limit_preserves_usage(self):
        with self.assertRaises(ModelActionError) as error:
            self.generate(payload(reason='length', usage={'prompt_tokens': 2, 'completion_tokens': 20}))
        self.assertEqual(error.exception.termination_reason, 'budget_exhausted')
        self.assertEqual(error.exception.usage.output_tokens, 20)

    def test_multiple_actions_rejected(self):
        call = {'id': 'one', 'type': 'function', 'function': {'name': 'read', 'arguments': '{}'}}
        with self.assertRaises(ModelActionError):
            self.generate(payload({'tool_calls': [call, call]}, 'tool_calls'))

    def test_invalid_usage_not_fabricated(self):
        for count in (-1, True, 1.5):
            with self.assertRaises(ModelError):
                self.generate(payload(usage={'prompt_tokens': count}))

    def test_request_schema_timeout_and_output_cap(self):
        spec = ToolSpec('read', '1', 'generic', ('read',), {'type': 'object'}, description='Read')
        with patch('agent_harness.models.urllib.request.build_opener') as opener:
            opener.return_value.open.return_value = io.BytesIO(json.dumps(payload()).encode())
            self.model(api_key='secret', timeout_seconds=30).generate([], [spec], 7, timeout_seconds=2)
            args, kwargs = opener.return_value.open.call_args
            body = json.loads(args[0].data)
            self.assertEqual(kwargs['timeout'], 2)
            self.assertEqual(body['max_tokens'], 7)
            self.assertFalse(body['parallel_tool_calls'])
            self.assertEqual(body['tools'][0]['function']['name'], 'read')
            self.assertEqual(opener.return_value.open.call_count, 1)

    def test_http_error_is_sanitized_without_retry(self):
        with patch('agent_harness.models.urllib.request.build_opener') as opener:
            opener.return_value.open.side_effect = urllib.error.HTTPError('https://secret', 401, 'secret', {}, None)
            with self.assertRaises(ModelError) as error:
                self.model(api_key='secret').generate([], [], 5)
            self.assertNotIn('secret', str(error.exception))
            self.assertEqual(opener.return_value.open.call_count, 1)

    def test_response_byte_limit(self):
        with self.assertRaisesRegex(ModelError, 'byte limit'):
            self.generate(payload(), max_response_bytes=10)

    def test_endpoint_validation(self):
        for url in ('http://remote.invalid/v1', 'https://user:secret@example.com', 'https://example.com?key=secret'):
            with self.assertRaises(ValueError):
                ChatCompletionsModel(model='test', base_url=url)
