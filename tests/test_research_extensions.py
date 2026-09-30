"""Regression checks for planning isolation and the distillation loss contrast."""
import copy
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from training_pipeline.collection import CollectionEpisode
from training_pipeline.config import inputs, validate_config
from training_pipeline.harness_variants import validate
from training_pipeline.rendering import ChatRenderer
from training_pipeline.sft_collection import load_supervised


class ResearchExtensionTests(unittest.TestCase):
    def config(self, name):
        return json.loads(Path('configs/experiments/research-extensions-v1/' + name + '.json').read_text())

    def test_planning_changes_prompt_not_tools_or_public_task(self):
        row = {'id':'task', 'public':{'id':'task', 'permitted_tools':['read_file'], 'budgets':{'max_tool_calls':5}},
               'image_result':{'snapshot_files':{}}}
        original = copy.deepcopy(row)
        with TemporaryDirectory() as tmp, patch('training_pipeline.collection.EpisodeRecorder'):
            episodes = []
            for arm in ['control', 'decompose']:
                config = self.config('07-' + arm + '-selection-r1')
                factory = SimpleNamespace(root=Path(tmp), config=config)
                episodes.append(CollectionEpisode(row, None, factory, arm, Path(tmp)/arm))
            control, treatment = episodes
            self.assertEqual(row, original)
            self.assertEqual(control.messages[1], treatment.messages[1])
            self.assertEqual(control.row['public']['permitted_tools'], treatment.row['public']['permitted_tools'])
            self.assertNotIn('checklist', control.messages[0]['content'])
            self.assertIn('Derive it only from the user question', treatment.messages[0]['content'])
            self.assertEqual(control.limits, treatment.limits)

    def test_unknown_variants_fail_closed(self):
        config = self.config('07-control-selection-r1')
        config['harness']['planning'] = 'invented'
        with self.assertRaises(ValueError): validate(config['harness'])
        config = self.config('08-answer-only-sft')
        config['supervised']['loss_scope'] = 'invented'
        with self.assertRaises(ValueError): validate_config(config)
        tool = json.loads(Path('configs/experiments/current-v8/04-tool-only-sft.json').read_text())
        tool['supervised']['loss_scope'] = 'final-answer-only'
        with self.assertRaisesRegex(ValueError, 'complete investigations'): inputs(tool)

    def test_same_verified_context_only_final_target_has_loss(self):
        from transformers import AutoTokenizer
        full_config = self.config('08-investigation-sft')
        answer_config = self.config('08-answer-only-sft')
        full = inputs(full_config)['sft']
        answer = inputs(answer_config)['sft']
        self.assertEqual(len(full), len(answer))
        renderer = ChatRenderer(AutoTokenizer.from_pretrained(full_config['model']['base_model'], local_files_only=True), 8192)
        for f, a in zip(full, answer):
            self.assertEqual(f['messages'], a['messages'])
            self.assertEqual(f['proofs'], a['proofs'])
            self.assertEqual(f['task_id'], a['task_id'])
            self.assertGreater(len(f['assistant_turns']), 1)
            self.assertEqual(a['assistant_turns'], [f['assistant_turns'][-1]])
            rows = renderer.supervised(a)
            self.assertEqual(len(rows), 1)
            prefix = len(renderer.prompt(a['messages'][:a['assistant_turns'][0]])) - 1
            self.assertTrue(all(w == 0 for w in rows[0].weights[:prefix]))
            self.assertTrue(all(w > 0 for w in rows[0].weights[prefix:]))
        # Changing target selection cannot bypass original trajectory admission.
        with patch('training_pipeline.investigation_sft.candidate', side_effect=ValueError('Invalid original proof')):
            with self.assertRaisesRegex(ValueError, 'Invalid original proof'): inputs(answer_config)

    def test_rl_continuations_are_matched_and_point_to_distinct_parents(self):
        direct = self.config('08-direct-grpo')
        for arm in ['investigation', 'answer-only']:
            fork = self.config('08-' + arm + '-then-grpo')
            for key in ['model', 'seed', 'stages', 'limits', 'environment', 'judge', 'evaluation', 'training_reward']:
                self.assertEqual(direct[key], fork[key], key)
            parent = self.config('08-' + arm + '-sft')
            self.assertEqual(fork['execution']['checkpoint'], parent['output'] + '/checkpoints/best.json')


if __name__ == '__main__':
    unittest.main()
