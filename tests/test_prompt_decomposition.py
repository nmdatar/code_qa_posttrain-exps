import copy
import json
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch
from training_pipeline.prompt_decomposition import prepare, parse, augment, VERSION
from training_pipeline.storage import digest
from training_pipeline.contracts import Generation, ConfigurationError
from training_pipeline.config import validate_config

class DecompositionTests(unittest.TestCase):
    def config(self):
        c = json.loads(Path('configs/experiments/bash-correctness-grpo-15-v1/run.json').read_text())
        c['prompt_decomposition'] = {'version':VERSION}
        return c

    def test_config_and_parse(self):
        c = self.config()
        validate_config(c)
        c['prompt_decomposition']['version'] = 'unknown'
        with self.assertRaises(ConfigurationError): validate_config(c)
        for text in ('{}', '{"subsections":[]}', '{"subsections":[{"heading":"a","question":""}]}'):
            with self.assertRaises(ValueError): parse(text)

    def test_question_only_frozen_across_checkpoints(self):
        question = 'How does x work, and what calls it?'
        row = {'id':'t1', 'public':{'user_prompt':question}, 'reference':'SECRET GOLD', 'rubric':'SECRET RUBRIC'}
        sample = Generation([1], [2], [0.0], json.dumps({'subsections':[
            {'heading':'Behavior', 'question':'How does x work?'},
            {'heading':'Callers', 'question':'What calls x?'}]}), 'stop', 'teacher')
        calls = []
        teacher = SimpleNamespace(identity={'base_model':'teacher'}, sample=lambda *a:(calls.append(a) or sample))
        with TemporaryDirectory() as root, patch('training_pipeline.cohorts.select_tasks', return_value=([row], {})):
            data = {'tasks':[row]}
            first = prepare(self.config(), data, root, None, teacher)
            second = prepare(self.config(), data, root, None, teacher)
            self.assertEqual(first, second)
            self.assertEqual(len(calls), 1)
            self.assertNotIn('SECRET', json.dumps(calls))
            self.assertEqual(json.loads(calls[0][0][1]['content']), {'question':question})
            self.assertTrue(augment(question, first['t1']).startswith(question))
            with self.assertRaises(ConfigurationError): augment('changed', first['t1'])

    def test_only_solver_prompt_changes(self):
        from training_pipeline.collection import CollectionEpisode
        c = self.config()
        public = {'id':'t', 'user_prompt':'How does x work?', 'permitted_tools':['read_file'], 'budgets':{}}
        row = {'id':'t', 'public':public}
        original = copy.deepcopy(row)
        with TemporaryDirectory() as root, patch('training_pipeline.collection.EpisodeRecorder'):
            factory = SimpleNamespace(config=c, root=Path(root), prompt_decompositions={
                't':{'question_sha256':digest(public['user_prompt']), 'subsections':[{'heading':'Behavior','question':public['user_prompt']}]}})
            episode = CollectionEpisode(row, None, factory, 'e', Path(root)/'e.json')
            self.assertEqual(row, original)
            self.assertEqual(episode.row['public']['user_prompt'], public['user_prompt'])
            self.assertIn('### Behavior', json.loads(episode.messages[1]['content'])['user_prompt'])
            self.assertEqual(json.loads(episode.messages[1]['content'])['permitted_tools'], ['bash'])

if __name__ == '__main__': unittest.main()
