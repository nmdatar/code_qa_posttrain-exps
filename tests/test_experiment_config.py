"""Config-to-runtime integration: independent judging, training knobs and run identity."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

from training_pipeline.config import validate_config, resolve_run_config
from training_pipeline.contracts import ConfigurationError, Generation
from training_pipeline.collection import CollectionFactory
from training_pipeline.launch import estimate
from training_pipeline.orchestrator import Pipeline
from training_pipeline.storage import Tracker, read, load_checkpoint, semantic_hash, run_label
from training_pipeline.smoke import smoke_config
from tests.test_training_pipeline import FakeBackend, Tokenizer, PRICES, trajectory


def collection_config():
    c = read(Path(__file__).parents[1] / 'examples/training-experiment.json')
    c['judge']['prices'] = dict(model=c['judge']['base_model'], prefill=2, sample=4,
                               source='test-fixture', checked_at='test-fixture')
    return c


class ExperimentConfigTests(unittest.TestCase):
    def test_config_loads_independent_models_and_validates_knobs(self):
        c = collection_config()
        self.assertNotEqual(c['model']['base_model'], c['judge']['base_model'])
        c['evaluation']['temperature'] = .7
        c['judge']['temperature'] = .2
        self.assertIs(validate_config(c), c)
        mutations = [lambda x: x['stages'][0]['optimizer'].update(beta1=1),
            lambda x: x['stages'][0]['optimizer'].update(eps=0),
            lambda x: x['stages'][0]['optimizer'].update(weight_decay=-1),
            lambda x: x['stages'][0]['optimizer'].update(grad_clip_norm=float('nan')),
            lambda x: x.update(group_retries=True),
            lambda x: x['judge'].update(temperature=-1),
            lambda x: x['judge']['prices'].update(model='wrong-model'),
            lambda x: x['tracking'].update(tags='not-a-list'),
            lambda x: x['stages'][0].update(temperature=.7)]
        for mutate in mutations:
            bad = copy.deepcopy(c); mutate(bad)
            with self.assertRaises(ConfigurationError):
                validate_config(bad)

    def test_auto_ids_paths_and_resume_identity(self):
        c = collection_config()
        # Shared project allocations are the default; path templating remains optional.
        self.assertEqual(resolve_run_config(c)['spend']['ledger'], c['spend']['ledger'])
        c['spend']['ledger'] = 'artifacts/test-ledgers/{run_id}.json'
        first, second = resolve_run_config(c), resolve_run_config(c)
        self.assertNotEqual(first['run_id'], second['run_id'])
        self.assertEqual(c['run_id'], 'auto')
        self.assertIn(first['run_id'], first['output'])
        self.assertIn(first['run_id'], first['spend']['ledger'])
        self.assertEqual(resolve_run_config(first), first)
        changed = copy.deepcopy(first)
        changed['judge']['base_model'] = 'different'
        self.assertNotEqual(semantic_hash(first), semantic_hash(changed))

    def test_collection_forwards_independent_judge_sampling_and_budget(self):
        c = collection_config()
        c['judge']['temperature'] = .25
        with tempfile.TemporaryDirectory() as directory:
            judge = Mock()
            judge.identity = {'base_model': c['judge']['base_model']}
            judge.sample.return_value = Generation([1], [2], [-.1],
                json.dumps({'status':'resolved','score':.75,'reason':'supported'}), 'stop', 'judge')
            ledger = Mock()
            with patch('training_pipeline.judge.TinkerJudge', return_value=judge) as constructor:
                factory = CollectionFactory(c, directory, ledger)
                factory.grade({'episode_id':'test'})
                constructor.assert_called_once_with(c['judge'], ledger)
                self.assertEqual(judge.sample.call_args.args[1:], (1024, .25))
                self.assertEqual(factory.judge_model, c['judge']['base_model'])
        c['group_retries'] = 0
        base = estimate(c, PRICES)
        self.assertEqual(base['components_usd']['reference_grading'],
            base['episodes_including_retries'] * (16384*2 + 1024*4)/1e6)
        c['group_retries'] = 3
        self.assertGreater(estimate(c, PRICES)['upper_estimate_usd'], base['upper_estimate_usd'])

    def test_wandb_name_config_group_and_resume(self):
        c = resolve_run_config(collection_config())
        c['tracking'].update(mode='online', entity='team', experiment_id='lr-sweep', tags=['grpo'])
        wandb = Mock()
        with tempfile.TemporaryDirectory() as directory:
            tracker = Tracker(directory, c['tracking'], c['run_id'], wandb, run_config=c)
            args = wandb.init.call_args.kwargs
            self.assertEqual(args['id'], c['run_id'])
            self.assertIn('Qwen3.5-4B__judge-'+c['judge']['base_model'].rsplit('/', 1)[-1], args['name'])
            self.assertIn('grpo-lr1e-05-b1-g4', args['name'])
            self.assertIn('r8-s42', args['name'])
            self.assertEqual(args['group'], 'lr-sweep')
            self.assertEqual(args['config'], c)
            self.assertEqual(args['entity'], 'team')
            tracker.finish('complete')
            resumed = Tracker(directory, {**c['tracking'], 'run_name':'changed'}, c['run_id'], wandb, run_config=c)
            self.assertEqual(resumed.organization, tracker.organization)
            evaluated = Tracker(directory, c['tracking'], c['run_id'], wandb, run_config=c, job_type='evaluation')
            self.assertNotEqual(evaluated.organization['id'], tracker.organization['id'])
            self.assertEqual(evaluated.organization['job_type'], 'evaluation')

    def test_pipeline_uses_stage_settings_and_persists_resolved_config(self):
        class Backend(FakeBackend):
            def update(self, rows, loss, learning_rate, optimizer=None):
                self.received = (loss, learning_rate, optimizer)
                return super().update(rows, loss, learning_rate)
        with tempfile.TemporaryDirectory() as directory:
            c = smoke_config(Path(directory), PRICES, Path(directory)/'ledger.json')
            c.update(run_id='auto', output=str(Path(directory)/'{run_id}'), group_retries=0)
            stage = {'kind':'grpo','max_updates':1,'max_batches':1,'batch_size':2,
                'group_size':3,'temperature':1,'learning_rate':.0002,
                'optimizer':{'beta1':.8,'grad_clip_norm':1.5}}
            c['stages'] = [stage]
            backend = Backend()
            pipeline = Pipeline(c, backend=backend)
            seen = []
            def group(task, size, temperature):
                seen.append((task['id'], size, temperature))
                return [trajectory(i, i) for i in range(size)]
            pipeline.group = group
            checkpoint = pipeline.run()
            manifest = load_checkpoint(checkpoint)
            self.assertEqual(len(seen), 2)
            self.assertEqual([s[1:] for s in seen], [(3, 1), (3, 1)])
            self.assertEqual(backend.received, ('importance_sampling', .0002, stage['optimizer']))
            self.assertNotEqual(manifest['config']['run_id'], 'auto')
            self.assertEqual(read(pipeline.root/'config.json'), manifest['config'])
            resumed = Pipeline(manifest['config'], backend=Backend(backend.registry))
            self.assertEqual(resumed.config['run_id'], pipeline.config['run_id'])
            resumed.run(checkpoint, 'resume')

    def test_configured_group_retry_count(self):
        with tempfile.TemporaryDirectory() as directory:
            c = smoke_config(Path(directory), PRICES, Path(directory)/'ledger.json')
            c['group_retries'] = 2
            pipeline = Pipeline(c, backend=FakeBackend(), tracker=Mock())
            def unresolved(*args):
                t = trajectory()
                from training_pipeline.contracts import VerificationResult
                t.policy_id = pipeline.backend.policy_id
                t.verification = VerificationResult('unresolved', None, 'v1', retryable=True)
                return t
            pipeline.rollout = Mock(side_effect=unresolved)
            pipeline.group({}, 4, 1)
            self.assertEqual(pipeline.rollout.call_count, 12)


@unittest.skipUnless(importlib.util.find_spec('tinker'), 'optional Tinker SDK not installed')
class SDKConfigTests(unittest.TestCase):
    def test_judge_uses_separate_sampling_only_model_and_hf_renderer(self):
        import tinker
        from training_pipeline.judge import TinkerJudge
        c = collection_config()['judge']; c['temperature'] = .2
        c['renderer'] = 'hf-chat-no-thinking-v1'  # Explicitly exercise the HF adapter.
        sampler = Mock()
        sampler.get_tokenizer.return_value = Tokenizer()
        sampler.sample.return_value = NS(result=lambda **kw: NS(sequences=[
            NS(tokens=[65], logprobs=[-.1], stop_reason='stop')]))
        service = Mock()
        service.get_server_capabilities.return_value = NS(supported_models=[
            NS(model_name=c['base_model'], sampleable=True, trainable=False, max_context_length=32768)])
        service.create_sampling_client.return_value = sampler
        ledger = Mock()
        judge = TinkerJudge(c, ledger, service=service, sdk=tinker)
        result = judge.sample([{'role':'user','content':'test'}], c['max_tokens'], .2)
        self.assertEqual(service.create_sampling_client.call_args.kwargs['base_model'], c['base_model'])
        service.create_lora_training_client.assert_not_called()
        self.assertEqual(sampler.sample.call_args.kwargs['sampling_params'].temperature, .2)
        self.assertEqual(ledger.reserve_external.call_args.kwargs['model'], c['base_model'])
        self.assertEqual(result.text, 'A')

    def test_optimizer_reaches_tinker(self):
        import tinker
        from training_pipeline.tinker_backend import TinkerBackend
        from training_pipeline.rendering import shifted
        from tests.test_training_pipeline import Tokenizer
        trainer = Mock()
        trainer.forward_backward.return_value = NS(result=lambda **kw: NS(
            metrics={'loss:sum':.5}, loss_fn_outputs=[{'logprobs':NS(data=[0, -.5])}]))
        trainer.optim_step.return_value = NS(result=lambda **kw: NS(metrics={}))
        service = Mock()
        service.get_server_capabilities.return_value = NS(supported_models=[
            NS(model_name='solver-other', sampleable=True, trainable=True, max_context_length=32768)])
        service.create_sampling_client.return_value.get_tokenizer.return_value = Tokenizer()
        service.create_lora_training_client.return_value = trainer
        c = collection_config()
        backend = TinkerBackend({**c['model'], 'base_model':'solver-other', 'rank':16}, c['limits'], service=service, sdk=tinker)
        backend.create_trainer(43)
        backend.update([shifted([1, 2], [3], [1.0])], 'cross_entropy', .0003,
                       optimizer={'beta1':.85, 'beta2':.98, 'eps':1e-8, 'weight_decay':.02, 'grad_clip_norm':1.2})
        params = trainer.optim_step.call_args.args[0]
        self.assertEqual((params.learning_rate, params.beta1, params.beta2, params.eps,
                          params.weight_decay, params.grad_clip_norm), (.0003, .85, .98, 1e-8, .02, 1.2))
        self.assertEqual(service.create_lora_training_client.call_args.kwargs,
                         {'base_model':'solver-other', 'rank':16, 'seed':43})


if __name__ == '__main__':
    unittest.main()
