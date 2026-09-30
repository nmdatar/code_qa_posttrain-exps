"""Token-exact model backends. Live calls require an explicit budget ledger.

Tinker SDK contract: https://tinker-docs.thinkingmachines.ai/tinker/sdk-cheatsheet/
Importance sampling sums losses, so weights below normalize per trajectory/batch.
"""
import hashlib
import json
import math
import re
import uuid
from pathlib import Path
import threading


def training_rows(trajectories, advantages=None, algorithm='grpo'):
    if algorithm not in ('grpo', 'sft') or not trajectories:
        raise ValueError('Nonempty SFT or GRPO batch required')
    if algorithm == 'grpo':
        if advantages is None or len(advantages) != len(trajectories):
            raise ValueError('One advantage per trajectory required')
        bindings = {(t['task_id'], t['policy_id']) for t in trajectories}
        if len(bindings) != 1:
            raise ValueError('Mixed task or policy group')
    rows = []
    for i, trajectory in enumerate(trajectories):
        if trajectory.get('split') not in (None, 'train'):
            raise ValueError('Held-out trajectories cannot enter training')
        actions = trajectory['actions']
        total = sum(len(a['token_ids']) for a in actions)
        if not total:
            raise ValueError('Empty trajectory')
        advantage = float(advantages[i]) if algorithm == 'grpo' else 1.0
        if not math.isfinite(advantage):
            raise ValueError('Nonfinite advantage')
        weight = advantage / total / len(trajectories)
        for action in actions:
            if algorithm == 'grpo' and action.get('sft_only'):
                raise ValueError('SFT target records cannot supply RL behavior logprobs')
            prompt, tokens = action['prompt_tokens'], action['token_ids']
            logprobs = action['logprobs']
            if not prompt or not tokens or len(tokens) != len(logprobs):
                raise ValueError('Invalid token/logprob alignment')
            if action.get('policy_id') != trajectory['policy_id']:
                raise ValueError('Action policy mismatch')
            if any(type(x) is not int or x < 0 for x in prompt + tokens):
                raise ValueError('Invalid token IDs')
            if any(not math.isfinite(x) or x > 0 for x in logprobs):
                raise ValueError('Invalid log probabilities')
            full = prompt + tokens
            n = len(prompt) - 1
            row = {'input_tokens': full[:-1], 'target_tokens': full[1:],
                   'mask': [0] * n + [1] * len(tokens)}
            if algorithm == 'grpo':
                row.update(logprobs=[0.] * n + logprobs,
                           advantages=[0.] * n + [weight] * len(tokens))
            else:
                row['weights'] = [0.] * n + [weight] * len(tokens)
            rows.append(row)
    return rows


def render_sft(backend, example):
    """Render admitted assistant targets, excluding every context token.

    Cookbook masks select only the last assistant message at each prefix. Each
    trained span becomes one action with its original preceding token context.
    Synthetic zero logprobs are marked SFT-only and never used by GRPO.
    """
    if example.get('split') != 'train' or example.get('status') != 'accepted' or not example.get('provenance'):
        raise ValueError('SFT requires accepted training examples with provenance')
    messages = example['messages']
    actions = []
    if isinstance(backend, TinkerBackend) and backend.renderer is None:
        backend.connect()
    for i, message in enumerate(messages):
        if message['role'] != 'assistant':
            continue
        if isinstance(backend, FakeBackend):
            tokens = list(message['content'].encode())
            actions.append({'prompt_tokens': backend.render(messages[:i]), 'token_ids': tokens,
                            'logprobs': [0.] * len(tokens), 'policy_id': backend.policy_id, 'sft_only': True})
        else:
            from tinker_cookbook.renderers import TrainOnWhat
            rendered, weights = backend.renderer.build_supervised_example(
                messages[:i+1], train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE)
            tokens, weights = rendered.to_ints(), weights.tolist()
            if len(tokens) != len(weights) or any(x not in (0, 1) for x in weights):
                raise ValueError('Unsupported SFT renderer weight shape')
            cursor = 0
            while cursor < len(tokens):
                if not weights[cursor]:
                    cursor += 1; continue
                start = cursor
                while cursor < len(tokens) and weights[cursor]: cursor += 1
                if start == 0: raise ValueError('SFT target has no conditioning context')
                actions.append({'prompt_tokens': tokens[:start], 'token_ids': tokens[start:cursor],
                                'logprobs': [0.] * (cursor-start), 'policy_id': backend.policy_id, 'sft_only': True})
    if not actions:
        raise ValueError('SFT example contains no assistant target tokens')
    return {'task_id': example['id'], 'split': 'train', 'policy_id': backend.policy_id, 'actions': actions,
            'input_tokens': sum(len(a['prompt_tokens']) for a in actions),
            'output_tokens': sum(len(a['token_ids']) for a in actions), 'termination': 'sft_example'}


class FakeBackend:
    """Deterministic byte-token simulator; never evidence of real model training."""
    def __init__(self, model='fake', budget=None, responses=None, **kwargs):
        self.model, self.step, self.weight = model, 0, 0.0
        self.responses = responses
        self._lock = threading.Lock()
        self._refresh()

    def render_sft(self, example):
        return render_sft(self, example)

    def _refresh(self):
        self.policy_id = 'fake-' + hashlib.sha256(f'{self.model}:{self.weight}'.encode()).hexdigest()[:20]

    def render(self, messages):
        return list(json.dumps(messages, sort_keys=True, ensure_ascii=False).encode())

    def sample(self, messages, max_tokens=1024, temperature=1., seed=0):
        n = sum(m['role'] == 'assistant' for m in messages)
        if self.responses:
            text = self.responses[min(n, len(self.responses)-1)]
        elif n == 0:
            text = json.dumps({'tool': 'list_files', 'arguments': {'path': '.', 'limit': 10}})
        else:
            text = json.dumps({'final_answer': 'Evidence inspected. ' + ('Supported example.' if seed % 2 else 'Insufficient evidence.')})
        tokens = list(text.encode())[:max_tokens]
        return {'text': bytes(tokens).decode(errors='replace'), 'prompt_tokens': self.render(messages),
                'token_ids': tokens, 'logprobs': [-math.log(256)] * len(tokens),
                'policy_id': self.policy_id, 'stop_reason': 'length' if len(text.encode()) > max_tokens else 'stop',
                'simulated': True}

    def update(self, trajectories, advantages=None, learning_rate=1e-4, algorithm='grpo'):
        rows = training_rows(trajectories, advantages, algorithm)
        if algorithm == 'grpo' and any(t['policy_id'] != self.policy_id for t in trajectories):
            raise ValueError('Off-policy group rejected')
        if algorithm == 'grpo' and all(a == 0 for a in advantages):
            return {'updated': False, 'reason': 'zero_advantages', 'loss': 0.}
        with self._lock:
            self.step += 1
            self.weight += learning_rate
            self._refresh()
        return {'updated': True, 'loss': math.log(256) if algorithm == 'sft' else -sum(advantages)/len(advantages),
                'optimizer_step': self.step, 'simulated': True, 'training_rows': len(rows)}

    def save(self, name):
        state = {'backend': 'fake', 'model': self.model, 'step': self.step, 'weight': self.weight}
        return {'training_state': dict(state, kind='training'), 'sampling_state': dict(state, kind='sampling'),
                'policy_id': self.policy_id}

    def load(self, ref, optimizer=False, purpose=None):
        if isinstance(ref, dict) and 'training_state' in ref:
            ref = ref['sampling_state' if purpose == 'evaluate' else 'training_state']
        if ref.get('backend') != 'fake' or ref.get('model') != self.model:
            raise ValueError('Checkpoint backend/model mismatch')
        if optimizer and ref.get('kind') != 'training':
            raise ValueError('Optimizer restore requires training checkpoint')
        self.weight, self.step = ref['weight'], ref['step'] if optimizer else 0
        self._refresh()
        return self


class TinkerBackend:
    """Lazy SDK adapter using an explicit cookbook renderer and budget bounds.

    Constructing the adapter performs no service call. ``connect`` creates a
    training client, unless ``load(..., purpose='evaluate')`` is used first.
    Exceptions from optimizer calls are deliberately not retried here.
    """
    def __init__(self, model, budget, renderer_name, rank=32, bounds=None, **kwargs):
        self.model, self.budget, self.renderer_name = model, budget, renderer_name
        self.rank, self.bounds = rank, bounds or {}
        self.training = self.sampling = self.service = self.renderer = None
        self.policy_id = None
        self.step = 0
        self._connect_lock = threading.RLock()

    def render_sft(self, example):
        return render_sft(self, example)

    def _paid(self, kind, call):
        upper = self.bounds.get(kind)
        if self.budget is None or upper is None or not math.isfinite(upper) or upper <= 0:
            raise ValueError('Explicit positive budget bound required for Tinker ' + kind)
        return self.budget.execute('tinker', upper, call, {'operation': kind, 'model': self.model})

    def _sdk(self):
        import tinker
        if self.service is None:
            self.service = tinker.ServiceClient(max_retries=0)
        return tinker

    def _renderer(self):
        from tinker_cookbook import renderers
        tokenizer = self.training.get_tokenizer() if self.training else self.sampling.get_tokenizer()
        self.renderer = renderers.get_renderer(self.renderer_name, tokenizer)

    def connect(self):
        with self._connect_lock:
            if self.training is None:
                self._sdk()
                self.training = self._paid('connect', lambda: self.service.create_lora_training_client(base_model=self.model, rank=self.rank))
                self._renderer()
                self._refresh('initial')
        return self

    def _refresh(self, name):
        name = re.sub(r'[^a-zA-Z0-9_-]', '-', Path(name).name)[:80] + '-' + uuid.uuid4().hex[:12]
        saved = self._paid('save', lambda: self.training.save_weights_for_sampler(name=name).result())
        self.sampling = self.service.create_sampling_client(model_path=saved.path)
        self.policy_id = saved.path

    def render(self, messages):
        if self.renderer is None:
            self.connect()
        # Our tool interface is JSON actions, not native Harmony recipients.
        # Fix the output channel in conditioning rather than interpreting analysis
        # text as a tool call. These prefix tokens have zero training loss.
        kwargs = {'prefill': '<|channel|>final<|message|>'} if self.renderer_name.startswith('gpt_oss') else {}
        prompt = self.renderer.build_generation_prompt(messages, **kwargs)
        return prompt.to_ints()

    def sample(self, messages, max_tokens=1024, temperature=1., seed=0):
        sdk = self._sdk()
        prompt = self.render(messages)
        params = sdk.types.SamplingParams(max_tokens=max_tokens, temperature=temperature,
                                          seed=seed, stop=self.renderer.get_stop_sequences())
        response = self._paid('sample', lambda: self.sampling.sample(
            prompt=sdk.types.ModelInput.from_ints(prompt), num_samples=1, sampling_params=params).result())
        seq = response.sequences[0]
        tokens, logprobs = list(seq.tokens), list(seq.logprobs or [])
        if len(tokens) != len(logprobs):
            raise ValueError('Tinker omitted sampled token log probabilities')
        parsed = self.renderer.parse_response(tokens)
        # Cookbook renderers return (message, success), not a raw decoded string.
        message = parsed[0] if isinstance(parsed, tuple) else parsed
        content = message.get('content', '')
        if isinstance(content, list):
            content = ''.join(part['text'] for part in content if part.get('type') == 'text')
        if not isinstance(content, str):
            raise ValueError('This adapter requires a text renderer')
        return {'text': content, 'prompt_tokens': prompt, 'token_ids': tokens, 'logprobs': logprobs,
                'policy_id': self.policy_id, 'stop_reason': seq.stop_reason, 'simulated': False,
                'parsed_message': message, 'parse_termination': str(parsed[1]) if isinstance(parsed, tuple) else None}

    def update(self, trajectories, advantages=None, learning_rate=1e-4, algorithm='grpo'):
        sdk = self._sdk()
        rows = training_rows(trajectories, advantages, algorithm)
        if algorithm == 'grpo' and any(t['policy_id'] != self.policy_id for t in trajectories):
            raise ValueError('Off-policy group rejected')
        if algorithm == 'grpo' and all(a == 0 for a in advantages):
            return {'updated': False, 'reason': 'zero_advantages', 'loss': 0.}
        if self.training is None:
            raise ValueError('Training client required')
        data = [sdk.types.Datum(model_input=sdk.types.ModelInput.from_ints(r['input_tokens']),
                               loss_fn_inputs={k:v for k,v in r.items() if k not in ('input_tokens', 'mask')}) for r in rows]
        result = self._paid('update', lambda: self._update_call(sdk, data, algorithm, learning_rate))
        self.step += 1
        # A refresh failure propagates; caller must recover from its last commit.
        self._refresh('update-' + str(self.step))
        metrics = dict(result.metrics)
        loss = metrics.get('loss:sum', metrics.get('loss'))
        if loss is None or not math.isfinite(float(loss)):
            raise ValueError('Training response did not include finite loss')
        return {'updated': True, 'loss': float(loss), 'optimizer_step': self.step, 'metrics': metrics}

    def _update_call(self, sdk, data, algorithm, learning_rate):
        result = self.training.forward_backward(data, 'importance_sampling' if algorithm == 'grpo' else 'cross_entropy').result()
        metrics = dict(result.metrics)
        loss = metrics.get('loss:sum', metrics.get('loss'))
        if loss is None or not math.isfinite(float(loss)):
            raise ValueError('Training returned nonfinite/missing loss before optimizer update')
        self.training.optim_step(sdk.types.AdamParams(learning_rate=learning_rate)).result()
        return result

    def save(self, name):
        name = re.sub(r'[^a-zA-Z0-9_-]', '-', Path(name).name)[:80] + '-' + uuid.uuid4().hex[:12]
        if self.training is None:
            raise ValueError('Training state not available')
        train = self._paid('save', lambda: self.training.save_state(name=name).result())
        sample = self._paid('save', lambda: self.training.save_weights_for_sampler(name=name).result())
        return {'training_state': train.path, 'sampling_state': sample.path, 'policy_id': self.policy_id}

    def load(self, ref, optimizer=False, purpose=None):
        self._sdk()
        if isinstance(ref, dict):
            ref = ref['sampling_state' if purpose == 'evaluate' else 'training_state']
        if purpose == 'evaluate':
            self.sampling = self.service.create_sampling_client(model_path=ref)
            self.policy_id = ref
        else:
            method = (self.service.create_training_client_from_state_with_optimizer if optimizer
                      else self.service.create_training_client_from_state)
            self.training = self._paid('connect', lambda: method(path=ref))
            self._refresh('restored')
        self._renderer()
        return self
