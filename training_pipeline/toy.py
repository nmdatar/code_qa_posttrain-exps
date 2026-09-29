"""Synthetic deterministic lookup fixture, never a repository training release."""
import json
from .contracts import VerificationResult
from .storage import digest

PROTOCOL = ('Lookup game. Use JSON only. First call {"tool":"lookup","key":"KEY"}. '
            'Then return {"answer":"VALUE","tag":"WORD"}, where WORD is any one lowercase word. '
            'The fixed reward is 0.8 for the correct value plus up to 0.2 for tag letters (a=0,...,z=25, mean/25).')


def task(index, split='train'):
    return {'id': f'toy-{split}-{index}', 'split': split, 'family_id': 'toy-' + split,
            'lineage_id': f'toy-{split}-{index}', 'key': f'key{index}', 'value': f'value{index*7+3}'}


def messages(t):
    return [{'role': 'system', 'content': PROTOCOL}, {'role': 'user', 'content': 'Look up ' + t['key']}]


def toy_inputs():
    tasks = [task(i) for i in range(8)]
    examples = []
    for t in tasks:
        conversation = messages(t) + [
            {'role': 'assistant', 'content': json.dumps({'tool': 'lookup', 'key': t['key']})},
            {'role': 'user', 'content': 'Tool observation: ' + json.dumps({'value': t['value']})},
            {'role': 'assistant', 'content': json.dumps({'answer': t['value'], 'tag': 'blue'})}]
        examples.append({**t, 'status': 'accepted', 'messages': conversation})
    result = {'sft': examples, 'tasks': tasks, 'development': [task(20, 'development'), task(21, 'development')]}
    return {**result, 'identity': digest(result)}


class ToyFactory:
    identity = 'toy-lookup-v1'
    reward_version = 'toy-lookup-and-tag-v1'

    def create(self, task, episode_id, trajectory_path):
        return ToyEpisode(task)


class ToyEpisode:
    def __init__(self, task):
        self.task = task
        self.messages = messages(task)
        self.looked_up = False
        self.closed = False

    def step(self, action):
        if set(action) == {'tool', 'key'} and action['tool'] == 'lookup':
            if action['key'] != self.task['key']:
                raise ValueError('Unknown key')
            self.looked_up = True
            return False, {'value': self.task['value']}
        if set(action) == {'answer', 'tag'} and all(isinstance(action[k], str) for k in action):
            return True, action
        raise ValueError('Invalid lookup action')

    def verify(self, trajectory):
        answer = trajectory.submission or {}
        correct = self.looked_up and answer.get('answer') == self.task['value']
        tag = answer.get('tag', '')
        quality = sum(ord(c)-97 for c in tag)/ (25*len(tag)) if tag and tag.isascii() and tag.isalpha() and tag.islower() else 0
        reward = .8*bool(correct) + .2*quality if correct else 0.0
        return VerificationResult('resolved', reward, ToyFactory.reward_version,
                                  diagnostics={'synthetic': True, 'correct': bool(correct), 'tag_score': quality})

    def close(self):
        self.closed = True
