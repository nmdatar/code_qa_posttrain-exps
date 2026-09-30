"""End-to-end coordinator tests use only explicit simulated models and tasks."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from posttrain.backends import FakeBackend
from posttrain.checkpoints import load_checkpoint
from posttrain.config import load_config
from posttrain.runner import run, readiness, freeze_data
from posttrain.storage import read


def fixture_task(identifier, split):
    return {'id': identifier, 'split': split, 'system_prompt': 'Inspect source.',
            'user_prompt': 'Describe this repository.', 'environment_id': 'env-'+identifier,
            'repository': {'family_id': 'fixture/'+identifier, 'commit': 'a'*40, 'url': 'https://example.invalid/'+identifier},
            'permitted_tools': ['list_files', 'read_file', 'search_code'],
            'budgets': {'max_output_tokens': 2048, 'max_tool_calls': 8, 'latency_seconds': 300}}


def config(root, identifier='diagnostic'):
    return {'run_id':identifier,'backend':'fake','model':'fake','diagnostic':True,
            'artifacts_root':str(Path(root)/'runs'),'budget_ledger':str(Path(root)/'spending.json'),
            'train_tasks':[fixture_task('training', 'train')],
            'eval_tasks':[fixture_task('evaluation', 'development')],
            'stages':[{'algorithm':'grpo','max_updates':3,'learning_rate':.001}],
            'eval_every':1,'checkpoint_every':1,'group_size':4,'max_concurrency':2,
            'tracking_mode':'disabled'}


def events(directory):
    return [json.loads(line) for line in (Path(directory)/'events.jsonl').read_text().splitlines()]


class PipelineIntegrationTests(unittest.TestCase):
    def test_release_selection_preserves_configured_order(self):
        c=load_config({'run_id':'order','training_release':'release','train_tasks':[{'id':'b'},{'id':'a'}]})
        public=[fixture_task('a','train'),fixture_task('b','train')]
        with patch('posttrain.data.validate_artifacts'), patch('posttrain.data.read_jsonl',side_effect=[public,[]]), patch('posttrain.data.adapt_task',side_effect=lambda task,*_:task):
            frozen=freeze_data(c)
        self.assertEqual([t['id'] for t in frozen['train_tasks']],['b','a'])

    def test_grpo_updates_checkpoint_and_periodic_eval(self):
        with tempfile.TemporaryDirectory() as root:
            result=run(config(root))
            self.assertTrue(result['simulated'])
            self.assertEqual(result['state']['optimizer_step'],3)
            manifest=load_checkpoint(result['checkpoint'])
            self.assertEqual(manifest['state']['optimizer_step'],3)
            self.assertEqual(manifest['references']['training_state']['step'],3)
            reports=[read(p) for p in (Path(result['run_dir'])/'evaluations').glob('*.json')]
            self.assertEqual(sorted(r['optimizer_step'] for r in reports),[0,1,2,3])
            self.assertTrue(all(r['simulated'] and r['expected']==1 for r in reports))
            self.assertEqual(read(Path(root)/'spending.json')['calls'],{})

    def test_sft_then_grpo_resets_optimizer_at_stage_boundary(self):
        with tempfile.TemporaryDirectory() as root:
            c=config(root)
            c['stages']=[{'algorithm':'sft','max_updates':1,'learning_rate':.001},
                         {'algorithm':'grpo','max_updates':2,'learning_rate':.001}]
            c['sft_examples']=[{'id':'training','split':'train','status':'accepted','provenance':{'origin':'synthetic-fixture'},
                'messages':[{'role':'user','content':'Question'}, {'role':'assistant','content':'Answer'}]}]
            result=run(c); manifest=load_checkpoint(result['checkpoint'])
            self.assertEqual(result['state']['optimizer_step'],3)
            self.assertEqual(manifest['references']['training_state']['step'],2)
            updates=[e for e in events(result['run_dir']) if e['kind']=='update']
            self.assertEqual([e['algorithm'] for e in updates],['sft','grpo','grpo'])

    def test_resume_and_fork_have_distinct_optimizer_semantics(self):
        with tempfile.TemporaryDirectory() as root:
            c=config(root); first=run(c)
            candidates=[(p,load_checkpoint(p)) for p in (Path(first['run_dir'])/'checkpoints').glob('*/manifest.json')]
            checkpoint=next(str(p) for p,m in candidates if m['state']['optimizer_step']==1)
            resumed=run(c,checkpoint=checkpoint,resume=True)
            self.assertEqual(load_checkpoint(resumed['checkpoint'])['references']['training_state']['step'],3)
            fork_config=config(root,'forked'); fork_config['stages'][0]['max_updates']=1
            forked=run(fork_config,checkpoint=checkpoint,resume=False)
            self.assertEqual(forked['state']['optimizer_step'],1)
            self.assertEqual(load_checkpoint(forked['checkpoint'])['references']['training_state']['step'],1)
            self.assertEqual(forked['state']['parent_checkpoint'],checkpoint)

    def test_unknown_optimizer_outcome_restarts_from_committed_state(self):
        class Ambiguous(FakeBackend):
            def update(self,*args,**kwargs):
                super().update(*args,**kwargs)
                raise ConnectionError('Response lost after optimizer mutation')
        with tempfile.TemporaryDirectory() as root:
            c=config(root)
            with patch('posttrain.runner.make_backend',lambda c,ledger:Ambiguous()):
                with self.assertRaises(ConnectionError):run(c)
            directory=Path(root)/'runs'/'diagnostic'
            journal=read(directory/'update-journal.json')
            self.assertEqual(journal['status'],'update_pending')
            checkpoint=journal['last_checkpoint']
            self.assertEqual(load_checkpoint(checkpoint)['references']['training_state']['step'],0)
            resumed=run(c,checkpoint=checkpoint,resume=True)
            self.assertEqual(load_checkpoint(resumed['checkpoint'])['references']['training_state']['step'],3)

    def test_diagnostic_grading_does_not_promote_machine_review_to_gold(self):
        from qa_eval.demo import fixture, KEY
        from qa_eval.dataset import freeze
        from qa_eval.security import bindings, digest, seal
        from qa_eval.grading import evaluate
        from qa_eval.source import GitSource
        with tempfile.TemporaryDirectory() as root:
            task,answer,experiment,metrics,semantic=fixture(Path(root))
            task.update(human_reviewed=False,gold_status='draft')
            experiment=freeze(experiment,[task],{'synthetic':True})
            metrics.update(bindings(task,answer));metrics['time_to_first_token_seconds']=None;metrics['cost']=None
            semantic.update(bindings(task,answer));semantic['experiment_hash']=digest(experiment)
            source=GitSource(root,task['repository']['commit'])
            args=(task,answer,seal('EpisodeMetrics',metrics,KEY),experiment,source,KEY)
            normal=evaluate(*args,semantic_envelope=seal('SemanticAssessment',semantic,KEY))[0]
            diagnostic=evaluate(*args,semantic_envelope=seal('SemanticAssessment',semantic,KEY),diagnostic_machine_review=True)[0]
            self.assertEqual(normal['tier'],'unresolved')
            self.assertEqual(diagnostic['tier'],'accepted')
            self.assertFalse(task['human_reviewed']);self.assertEqual(task['gold_status'],'draft')

    def test_live_inline_tasks_cannot_bypass_release_admission(self):
        with tempfile.TemporaryDirectory() as root:
            c=config(root);c['backend']='tinker'
            with self.assertRaisesRegex(ValueError,'immutable training and evaluation releases'):
                freeze_data(load_config(c))

    def test_sft_cannot_relabel_evaluation_task(self):
        with tempfile.TemporaryDirectory() as root:
            c=config(root)
            c['sft_examples']=[{'id':'evaluation','split':'train','status':'accepted','provenance':{'origin':'fixture'},
                              'messages':[{'role':'user','content':'Q'},{'role':'assistant','content':'A'}]}]
            self.assertIn('SFT task overlaps evaluation: evaluation',readiness(load_config(c))['blockers'])

    def test_readiness_does_not_mark_uncalibrated_long_experiments_ready(self):
        with tempfile.TemporaryDirectory() as root:
            c=config(root);c['diagnostic']=False;c['review_policy']='human'
            self.assertEqual(readiness(load_config(c))['status'],'blocked')
            c=config(root);c['train_tasks'][0]['repository']['family_id']='fixture/evaluation'
            self.assertIn('Family split leakage: fixture/evaluation',readiness(load_config(c))['blockers'])

if __name__=='__main__':unittest.main()
