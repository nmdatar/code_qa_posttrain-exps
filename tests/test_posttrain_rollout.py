import json
import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path
from posttrain.backends import FakeBackend, TinkerBackend, training_rows
from posttrain.environments import FakeEnvironment, InfrastructureError
from posttrain.rollout import run_episode


def task():
    return {'id': 't1', 'environment_id': 'e1', 'split': 'train',
            'user_prompt': 'What is here?', 'system_prompt': 'Inspect sources.',
            'permitted_tools': ['list_files', 'read_file', 'search_code'],
            'private_reference': 'SECRET NEVER SHOW', 'budgets': {'max_output_tokens': 2000}}


class RolloutTests(unittest.TestCase):
    def test_single_json_fence_preserves_sampled_tokens(self):
        raw='```json\n{"final_answer":"Supported answer."}\n```'
        result=run_episode(FakeBackend(responses=[raw]),FakeEnvironment(),task())
        self.assertEqual(result['termination'],'completed')
        self.assertEqual(result['actions'][0]['text'],raw)
        self.assertEqual(result['actions'][0]['token_ids'],list(raw.encode()))
        invalid=run_episode(FakeBackend(responses=['Prose '+raw]),FakeEnvironment(),task())
        self.assertEqual(invalid['termination'],'invalid_action')

    def test_visible_tool_episode_private_separation(self):
        with tempfile.TemporaryDirectory() as root:
            result = run_episode(FakeBackend(), FakeEnvironment(), task(), output_dir=root)
            self.assertEqual(result['termination'], 'completed')
            self.assertEqual(result['tool_calls'], 1)
            self.assertEqual(len(result['actions']), 2)
            self.assertNotIn('SECRET', json.dumps(result))
            self.assertEqual(json.loads((Path(root)/'episode.json').read_text()), result)
            for action in result['actions']:
                self.assertEqual(len(action['token_ids']), len(action['logprobs']))

    def test_masks_do_not_retrain_context_or_previous_actions(self):
        result = run_episode(FakeBackend(), FakeEnvironment(), task())
        rows = training_rows([result], [1.])
        self.assertEqual(sum(sum(r['mask']) for r in rows), result['output_tokens'])
        self.assertAlmostEqual(sum(sum(r['advantages']) for r in rows), 1.)
        for row, action in zip(rows, result['actions']):
            self.assertEqual(row['target_tokens'], (action['prompt_tokens'] + action['token_ids'])[1:])
            self.assertEqual(sum(row['mask'][:len(action['prompt_tokens'])-1]), 0)

    def test_invalid_model_action_is_not_infrastructure_failure(self):
        result = run_episode(FakeBackend(responses=['not json']), FakeEnvironment(), task())
        self.assertEqual(result['termination'], 'invalid_action')
        self.assertEqual(len(result['actions']), 1)

    def test_provider_failure_retains_partial_trajectory(self):
        class Broken(FakeEnvironment):
            def call(self, *args): raise InfrastructureError('provider unavailable')
        result = run_episode(FakeBackend(), Broken(), task())
        self.assertEqual(result['termination'], 'infrastructure_error')
        self.assertEqual(len(result['actions']), 1)

    def test_context_and_tool_budgets(self):
        result = run_episode(FakeBackend(), FakeEnvironment(), task(), {'context_tokens': 1})
        self.assertEqual(result['termination'], 'context_budget')
        self.assertEqual(result['output_tokens'], 0)
        result = run_episode(FakeBackend(), FakeEnvironment(), task(), {'max_tool_calls': 0})
        self.assertEqual(result['termination'], 'tool_budget')
        self.assertEqual(result['tool_calls'], 0)

    def test_group_bindings_and_zero_update(self):
        b = FakeBackend(); a = run_episode(b, FakeEnvironment(), task())
        other = dict(a, task_id='different')
        with self.assertRaises(ValueError): training_rows([a, other], [-1, 1])
        old = b.policy_id
        self.assertFalse(b.update([a], [0.])['updated'])
        self.assertEqual(b.policy_id, old)
        self.assertTrue(b.update([a], [1.])['updated'])
        self.assertNotEqual(b.policy_id, old)

    def test_resume_fork_and_sampling_checkpoint(self):
        b = FakeBackend(); a = run_episode(b, FakeEnvironment(), task())
        b.update([a], [1.]); saved = b.save('one')
        resumed = FakeBackend().load(saved, optimizer=True)
        self.assertEqual(resumed.step, 1); self.assertEqual(resumed.policy_id, b.policy_id)
        forked = FakeBackend().load(saved)
        self.assertEqual(forked.step, 0); self.assertEqual(forked.policy_id, b.policy_id)
        with self.assertRaises(ValueError): FakeBackend().load(saved['sampling_state'], optimizer=True)

    def test_sft_masks(self):
        a = run_episode(FakeBackend(), FakeEnvironment(), task())
        rows = training_rows([a], algorithm='sft')
        self.assertAlmostEqual(sum(sum(r['weights']) for r in rows), 1.)
        self.assertTrue(all('advantages' not in r for r in rows))

    def test_sft_render_rejects_nontraining_and_masks_history(self):
        b = FakeBackend()
        example = {'id': 'sft1', 'split': 'train', 'status': 'accepted', 'provenance': {'teacher': 'fixture'},
                   'messages': [{'role': 'user', 'content': 'Q'}, {'role': 'assistant', 'content': 'A'},
                                {'role': 'user', 'content': 'tool observation'}, {'role': 'assistant', 'content': 'B'}]}
        trajectory = b.render_sft(example)
        rows = training_rows([trajectory], algorithm='sft')
        self.assertEqual(sum(sum(row['mask']) for row in rows), 2)
        with self.assertRaises(ValueError): training_rows([trajectory], [1.])
        example['split'] = 'development'
        with self.assertRaises(ValueError): b.render_sft(example)

    def test_timing_and_turn_limit(self):
        result = run_episode(FakeBackend(), FakeEnvironment(), task(), {'max_turns': 1})
        self.assertEqual(result['termination'], 'turn_budget')
        self.assertIsNone(result['time_to_first_token_seconds'])
        self.assertGreaterEqual(result['time_to_first_response_seconds'], 0.)
        self.assertGreaterEqual(result['tool_seconds'], 0.)

    def test_tinker_finite_loss_before_optimizer(self):
        b = TinkerBackend('model', None, 'renderer')
        calls = []
        class Future:
            def result(self): return SimpleNamespace(metrics={'loss:sum': float('nan')})
        b.training = SimpleNamespace(forward_backward=lambda *a: Future(),
                                     optim_step=lambda *a: calls.append('optimizer'))
        with self.assertRaises(ValueError): b._update_call(None, [], 'grpo', .001)
        self.assertEqual(calls, [])

    def test_tinker_update_refreshes_only_after_acknowledged_optimizer(self):
        b = TinkerBackend('model', None, 'renderer')
        calls = []
        class Future:
            def __init__(self, value): self.value = value
            def result(self): return self.value
        sdk = SimpleNamespace(types=SimpleNamespace(
            ModelInput=SimpleNamespace(from_ints=lambda x:x),
            Datum=lambda **kwargs:kwargs, AdamParams=lambda **kwargs:kwargs))
        b._sdk = lambda:sdk
        b._paid = lambda kind, operation: operation()
        b._refresh = lambda name:calls.append('refresh')
        b.training = SimpleNamespace(
            forward_backward=lambda *a: Future(SimpleNamespace(metrics={'loss:sum': .5})),
            optim_step=lambda *a: (calls.append('optimizer') or Future(None)))
        result = run_episode(FakeBackend(), FakeEnvironment(), task())
        b.policy_id = result['policy_id']
        update = b.update([result], [1.])
        self.assertEqual(calls, ['optimizer', 'refresh'])
        self.assertTrue(update['updated'])
        def fail(*a): raise ConnectionError('Unknown optimizer outcome')
        calls.clear(); b.training.optim_step = fail
        with self.assertRaises(ConnectionError): b.update([result], [1.])
        self.assertEqual(calls, [])

    def test_paid_calls_require_explicit_positive_reservations(self):
        b = TinkerBackend('model', None, 'renderer')
        invoked = []
        with self.assertRaises(ValueError): b._paid('sample', lambda: invoked.append(1))
        self.assertEqual(invoked, [])

if __name__ == '__main__': unittest.main()
