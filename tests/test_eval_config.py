import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from eval_pipeline.run import main, DIMS


class EvalConfigTests(unittest.TestCase):
    def test_baseline_config_runs_only_evaluation_with_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task = dict(id='q1', repo='test/repo', commit_id='a'*40, question='Where?')
            (root/'tasks.jsonl').write_text(json.dumps(task)+'\n')
            (root/'references.jsonl').write_text(json.dumps(dict(id='q1', reference_answer='reference'))+'\n')
            (root/'manifest.json').write_text(json.dumps(dict(revision='pinned', source_sha256='hash', task_ids=['q1'], snapshots=[])))
            path = Path(__file__).parents[1]/'configs/baseline-qwen3.5-9b.json'
            env = dict(EVAL_DATA_DIR=str(root), BASELINE_BASE_URL='http://answer/v1',
                       JUDGE_MODEL='independent-judge', JUDGE_BASE_URL='http://judge/v1')
            responses = [('answer', {}), (json.dumps(dict(scores=dict.fromkeys(DIMS, 8), reason='ok')), {})]
            with patch.dict(os.environ, env), patch('eval_pipeline.run.retrieve', return_value='source'), \
                 patch('eval_pipeline.run.chat', side_effect=responses) as chat, \
                 patch('eval_pipeline.tracking.log_run') as log, contextlib.redirect_stdout(io.StringIO()):
                result = main(['--config', str(path), '--output', str(root/'out')])
            self.assertEqual(result, 0)
            self.assertEqual(chat.call_count, 2)
            self.assertEqual(chat.call_args_list[0].args[:2], ('http://answer/v1', 'Qwen/Qwen3.5-9B'))
            self.assertEqual(chat.call_args_list[1].args[:2], ('http://judge/v1', 'independent-judge'))
            config = log.call_args.args[1]
            self.assertEqual(config['mode'], 'retrieval')
            self.assertEqual(config['organization']['experiment_id'], 'baseline-v1')
            self.assertIn('baseline', config['organization']['tags'])
            self.assertEqual(log.call_args.args[4], 'online')
            self.assertNotIn('MODEL_API_KEY', (root/'out/config.json').read_text())

    def test_invalid_or_unresolved_configs_fail_before_work(self):
        for config in ({'mode': 'invalid'}, {'stages': []}, {'context_chars': True},
                       {'tags': 'baseline'}, {'base_url': '${UNSET_EVAL_TEST_ENDPOINT}'}):
            with self.subTest(config=config), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp)/'run.json'
                path.write_text(json.dumps(config))
                with patch.dict(os.environ, {}, clear=True), contextlib.redirect_stderr(io.StringIO()), \
                     patch('eval_pipeline.run.read_rows') as read:
                    with self.assertRaises(SystemExit):
                        main(['--config', str(path)])
                    read.assert_not_called()

    def test_overrides_and_config_relative_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'run.json'
            path.write_text(json.dumps({'data': './dataset', 'base_url': '${UNSET_EVAL_TEST_ENDPOINT}',
                                        'wandb': 'disabled'}))
            with patch('eval_pipeline.run.read_rows', side_effect=RuntimeError('stop')) as read:
                with self.assertRaisesRegex(RuntimeError, 'stop'):
                    main(['--config', str(path), '--base-url', 'http://override/v1'])
                self.assertEqual(read.call_args.args[0], (path.parent/'dataset/tasks.jsonl').resolve())


if __name__ == '__main__':
    unittest.main()
