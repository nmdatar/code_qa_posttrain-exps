import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from training_eval.cli import main


class TrainingCliTests(unittest.TestCase):
    def test_experiment_metadata_in_local_logs_and_wandb(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = json.loads((Path(__file__).parents[1] / 'examples/training-demo.json').read_text())
            config['tracking'].update(experiment_id='ablation-v1', run_name='baseline-seed7',
                                      tags=['baseline'], source_run_id='training-seed7',
                                      metadata={'variant': 'baseline'})
            for mode in ('disabled', 'online'):
                config['tracking']['mode'] = mode
                config_path = root / 'config.json'
                config_path.write_text(json.dumps(config))
                output = root / mode
                with patch('training_eval.cli.WandbSink') as sink, contextlib.redirect_stdout(io.StringIO()):
                    sink.return_value.delivery_key = 'test-sink'
                    self.assertEqual(main(['evaluate', '--config', str(config_path), '--plugin',
                        'training_eval.mocks:demo_inputs', '--output', str(output)]), 0)
                    if mode == 'online':
                        options = sink.call_args.kwargs
                        self.assertEqual(options['group'], 'ablation-v1')
                        self.assertEqual(options['name'], 'baseline-seed7')
                        self.assertEqual(options['job_type'], 'evaluation')
                        self.assertEqual(options['config']['organization']['source_run_id'], 'training-seed7')
                    else:
                        sink.assert_not_called()
                event = json.loads((output / 'logs/events.jsonl').read_text().splitlines()[0])
                self.assertEqual(event['kind'], 'run_metadata')
                self.assertEqual(event['payload']['experiment_id'], 'ablation-v1')
                self.assertEqual(event['payload']['metadata'], {'variant': 'baseline'})

    def test_configured_run_resume_and_standalone_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = json.loads((Path(__file__).parents[1] / 'examples/training-demo.json').read_text())
            config_path = root / 'config.json'
            config_path.write_text(json.dumps(config))
            def call(command, output, *extra):
                args = [command, '--config', str(config_path), '--plugin',
                        'training_eval.mocks:demo_inputs', '--output', str(root / output), *extra]
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(main(args), 0)
                return json.loads((root / output / 'result.json').read_text())
            paused = call('run', 'run', '--stop-after-batches', '3')
            self.assertEqual(paused['status'], 'paused')
            resumed = call('run', 'run', '--resume', paused['checkpoint'])
            self.assertEqual(resumed['status'], 'complete')
            config['model'] = {'checkpoint': resumed['checkpoint']}
            config_path.write_text(json.dumps(config))
            report = call('evaluate', 'evaluation')
            self.assertEqual(report['coverage'], 1)
            self.assertEqual(report['checkpoint'], resumed['checkpoint'])
            config['run']['run_id'] = 'new-training-fork'
            config_path.write_text(json.dumps(config))
            fork = call('run', 'fork', '--stop-after-batches', '1')
            self.assertEqual(fork['optimizer_step'], 1)


if __name__ == '__main__':
    unittest.main()
