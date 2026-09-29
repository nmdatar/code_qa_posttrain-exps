"""Exact per-turn alignment without training prior assistant turns twice."""
from .contracts import LossRow
from .storage import digest


class ChatRenderer:
    def __init__(self, tokenizer, context_limit):
        self.tokenizer = tokenizer
        self.context_limit = context_limit
        self.identity = {'renderer': 'hf-chat-no-thinking-v1',
                         'template_hash': digest(tokenizer.get_chat_template()),
                         'tokenizer_class': type(tokenizer).__name__}
        self.prompt([{'role': 'user', 'content': 'Preflight'}])

    def render(self, messages, generation=False):
        return list(self.tokenizer.apply_chat_template(messages, tokenize=True,
            add_generation_prompt=generation, enable_thinking=False, return_dict=False))

    def prompt(self, messages):
        tokens = self.render(messages, True)
        if not tokens or len(tokens) >= self.context_limit:
            raise ValueError('Context budget exceeded')
        return tokens

    def supervised(self, example):
        messages = example['messages']
        selected = example.get('assistant_turns', [i for i, m in enumerate(messages) if m['role'] == 'assistant'])
        if not selected or len(set(selected)) != len(selected):
            raise ValueError('Empty or duplicate supervision selection')
        rows = []
        for i in selected:
            if type(i) is not int or not 0 < i < len(messages) or messages[i]['role'] != 'assistant':
                raise ValueError('Supervision must select an assistant turn with context')
            if not messages[i]['content'].strip():
                raise ValueError('Empty assistant target')
            prompt = self.prompt(messages[:i])
            full = self.render(messages[:i+1])
            # Some templates strip reasoning markers from completed assistant turns.
            # Reject rather than guessing offsets or re-tokenizing decoded samples.
            if full[:len(prompt)] != prompt:
                raise ValueError('Chat template does not preserve assistant generation prefix')
            completion = full[len(prompt):]
            if not completion or len(full) > self.context_limit:
                raise ValueError('Empty target or context overflow')
            rows.append(shifted(prompt, completion, [1.0] * len(completion)))
        count = sum(sum(r.weights) for r in rows)
        for row in rows:
            row.weights = [v / count for v in row.weights]
        return rows


def shifted(prompt, tokens, weights, logprobs=None):
    if not prompt or not tokens or len(tokens) != len(weights):
        raise ValueError('Invalid generation alignment')
    full = prompt + tokens
    prefix = len(prompt) - 1
    row = LossRow(full[:-1], full[1:], [0.0] * prefix + weights,
                  None if logprobs is None else [0.0] * prefix + logprobs)
    row.validate()
    return row


def sft_batch(renderer, examples):
    if not examples:
        raise ValueError('Empty SFT batch')
    rows = []
    for example in examples:
        if example.get('status') != 'accepted' or example.get('split') != 'train':
            raise ValueError('SFT requires accepted training examples')
        for row in renderer.supervised(example):
            row.weights = [w / len(examples) for w in row.weights]
            rows.append(row)
    return rows
