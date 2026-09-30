import copy
from dataclasses import asdict
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from training_pipeline.contracts import Generation, Trajectory, VerificationResult, InfrastructureError, AmbiguousUpdate
from training_pipeline.rendering import ChatRenderer, shifted, sft_batch
from training_pipeline.strategies import group_advantages, grpo_batch
from training_pipeline.storage import atomic_json, read, Tracker, Checkpoints, load_checkpoint, digest
from training_pipeline.budget import SpendLedger, BudgetLimit
from training_pipeline.config import inputs, load_release
from training_pipeline.smoke import smoke_config
from training_pipeline.toy import ToyFactory, toy_inputs
from training_pipeline.orchestrator import Pipeline, evaluate_checkpoint
from agent_harness.runner import run_episode
from agent_harness.repository_tools import command

PRICES = {'prefill': .33, 'sample': 1.005, 'train': .737, 'storage_gb_month': .1, 'params': 4_000_000_000}


class Tokenizer:
    eos_token_id = 0
    def get_chat_template(self): return 'test-chat-v1'
    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False, **kwargs):
        text = ''.join('<'+m['role']+'>'+m['content']+'!' for m in messages)
        if add_generation_prompt: text += '<assistant>'
        return [ord(c) for c in text] if tokenize else text
    def decode(self, tokens, **kwargs): return ''.join(chr(t) for t in tokens)


class FakeBackend:
    def __init__(self, registry=None):
        self.registry = registry if registry is not None else {}
        self.renderer = ChatRenderer(Tokenizer(), 4096)
        self.identity = {'base_model': 'Qwen/Qwen3.5-4B', 'rank': 8, **self.renderer.identity}
        self.policy_id = 'base:test'
        self.value = 0
        self.optimizer = 0
        self.calls = []
        self.samples = 0
        self.trainer_count = 0
        self.fail_update = False
        self.fail_save = False
    def create_trainer(self, seed):
        self.trainer_count += 1
        self.optimizer = 0
        self.calls.append(('create', seed))
    def sample(self, messages, max_tokens, temperature):
        self.samples += 1
        if len(messages) == 2:
            key = messages[1]['content'].split()[-1]
            text = json.dumps({'tool': 'lookup', 'key': key})
        else:
            value = json.loads(messages[-1]['content'].removeprefix('Tool observation: '))['value']
            text = json.dumps({'answer': value, 'tag': 'blue' if self.samples % 4 else 'red'})
        return Generation([1, 2], [3, 4], [-.3, -.4], text, 'stop', self.policy_id)
    def update(self, rows, loss, learning_rate):
        if self.fail_update: raise AmbiguousUpdate('unknown')
        self.value += 1
        self.optimizer += 1
        self.calls.append(('update', loss, self.policy_id))
        return {'acknowledged': True, 'loss': loss, 'metrics': {'loss:sum': .5}}
    def save(self, name):
        if self.fail_save: raise InfrastructureError('save failed')
        artifacts = {'training': 'fake://training/'+name, 'sampler': 'fake://sampler/'+name}
        self.registry[artifacts['training']] = (self.value, self.optimizer)
        self.registry[artifacts['sampler']] = self.value
        self.calls.append(('save', name))
        return artifacts
    def verify_artifacts(self, artifacts):
        if any(v not in self.registry for v in artifacts.values()): raise ValueError('missing artifact')
    def use_sampler(self, artifacts):
        self.policy_id = artifacts['sampler']
        self.calls.append(('sampler', self.policy_id))
    def load(self, artifacts, purpose):
        self.verify_artifacts(artifacts)
        if purpose != 'evaluate':
            self.value, optimizer = self.registry[artifacts['training']]
            self.optimizer = optimizer if purpose == 'resume' else 0
        self.use_sampler(artifacts)
        self.calls.append(('load', purpose))


def trajectory(i=0, reward=0, tokens=2):
    return Trajectory('run', 0, 'task', 'hash', 'group', 'ep'+str(i), 'policy', 'env', 'experiment', 'train',
        generations=[Generation([1, 2], [3]*tokens, [-.5]*tokens, 'answer', 'stop', 'policy')],
        termination='completed', verification=VerificationResult('resolved', reward, 'v1'))


class TrainingMathTests(unittest.TestCase):
    def test_shift_and_multiturn_sft(self):
        r = ChatRenderer(Tokenizer(), 4096)
        example = {'split': 'train', 'status': 'accepted', 'messages': [
            {'role':'user','content':'question'}, {'role':'assistant','content':'action'},
            {'role':'user','content':'observation'}, {'role':'assistant','content':'answer'}]}
        rows = sft_batch(r, [example])
        self.assertEqual(len(rows), 2)
        self.assertAlmostEqual(sum(sum(x.weights) for x in rows), 1)
        for row in rows:
            active = [t for t,w in zip(row.target_tokens,row.weights) if w]
            self.assertNotIn('observation', Tokenizer().decode(active))
        self.assertEqual(Tokenizer().decode([t for t,w in zip(rows[1].target_tokens,rows[1].weights) if w]), 'answer!')
        with self.assertRaises(ValueError): sft_batch(r, [{**example,'split':'development'}])
        with self.assertRaises(ValueError): shifted([1], [2,3], [1])
        with self.assertRaises(ValueError): r.supervised({**example,'assistant_turns':[]})
        with self.assertRaises(ValueError): ChatRenderer(Tokenizer(), 30).supervised(example)

    def test_reward_curve_counts_excluded_and_zero_signal_groups(self):
        good = [trajectory(0, 0), trajectory(1, 1)]
        excluded = [trajectory(2, 1), trajectory(3, 0)]
        excluded[1].verification = VerificationResult('unresolved', None, 'v1')
        _, stats = grpo_batch([good, excluded])
        self.assertAlmostEqual(stats['mean_reward'], 2/3)
        self.assertEqual(stats['eligible_mean_reward'], .5)
        self.assertEqual(stats['scoring_coverage'], .75)
        self.assertEqual(stats['excluded_group_fraction'], .5)
        _, stats = grpo_batch([[trajectory(0, 0), trajectory(1, 0)]])
        self.assertEqual(stats['mean_reward'], 0)
        self.assertEqual(stats['zero_variance_groups'], 1)

    def test_reward_curve_local_and_remote_metrics(self):
        import csv
        from unittest.mock import MagicMock
        with tempfile.TemporaryDirectory() as tmp:
            wandb = MagicMock()
            wandb.init.return_value.url = 'https://example.test/run'
            tracker = Tracker(tmp, {'mode': 'online'}, 'test', wandb_module=wandb)
            tracker.event('training_batch', attempted_batches=1, optimizer_step=0,
                          mean_reward=0, reward_ema=0, scoring_coverage=1,
                          excluded_groups=0, zero_variance_groups=2)
            tracker.event('training_batch', attempted_batches=2, optimizer_step=0,
                          mean_reward=None, reward_ema=None, scoring_coverage=0,
                          excluded_groups=2, zero_variance_groups=0)
            with (Path(tmp)/'reward-curve.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(rows[0]['mean_reward'], '0')
            self.assertEqual(rows[1]['mean_reward'], '')
            self.assertEqual(rows[1]['attempted_batches'], '2')
            logged = wandb.init.return_value.log.call_args.args[0]
            self.assertIsNone(logged['training/mean_reward'])
            self.assertEqual(logged['training/scoring_coverage'], 0)
            wandb.init.return_value.define_metric.assert_any_call('training/*', step_metric='attempted_batches')

    def test_grpo_population_and_equal_trajectory_weight(self):
        a,b = trajectory(0,0,2), trajectory(1,1,6)
        self.assertEqual(group_advantages([a,b]), [-1,1])
        b.generations = [Generation([1,2], [3]*3, [-.5]*3,'x','stop','policy'),
                         Generation([1,2,3,3,3,4], [5]*3, [-.5]*3,'y','stop','policy')]
        rows,stats = grpo_batch([[a,b]])
        self.assertEqual(stats['contributing_trajectories'],2)
        self.assertAlmostEqual(sum(rows[0].weights), -.5)
        self.assertAlmostEqual(sum(sum(r.weights) for r in rows[1:]), .5)
        self.assertEqual(rows[2].weights[:5], [0]*5)
        # Tinker's documented sum reduction yields the mean of per-trajectory means.
        ratios = [2, 3, 3]
        actual = -sum(ratio*sum(row.weights) for ratio,row in zip(ratios,rows))
        self.assertAlmostEqual(actual, -.5*((-1)*2 + 1*3))

    def test_exclusions_and_alignment(self):
        group = [trajectory(0,1), trajectory(1,1)]
        self.assertEqual(grpo_batch([group])[0], [])
        group[1].verification = VerificationResult('unresolved',None,'v1')
        self.assertIsNone(group_advantages(group))
        group[1].policy_id='different'
        with self.assertRaises(ValueError): group_advantages(group)
        group=[trajectory(0,0),trajectory(1,1)]
        group[1].generations[0].logprobs=[float('nan'),0]
        with self.assertRaises(ValueError): grpo_batch([group])
        group[1].generations[0].logprobs=[]
        with self.assertRaises(ValueError): grpo_batch([group])
        group=[trajectory(0,0),trajectory(1,1)]
        group[1].verification.version='v2'
        with self.assertRaises(ValueError): grpo_batch([group])


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.config=smoke_config(self.root,PRICES,self.root/'ledger.json')
        self.config['limits']['context_tokens']=4096
        self.backend=FakeBackend()

    def test_full_pipeline_refresh_evaluate_and_resume(self):
        p=Pipeline(self.config,backend=self.backend)
        path=p.run(stop_after_updates=2)
        m=load_checkpoint(path)
        self.assertEqual(m['state']['optimizer_step'],2)
        self.assertEqual(m['state']['stage'],1)
        self.assertEqual(self.backend.trainer_count,2)
        restored=FakeBackend(self.backend.registry)
        q=Pipeline(self.config,backend=restored)
        final=q.run(path,'resume')
        self.assertEqual(load_checkpoint(final)['state']['optimizer_step'],3)
        self.assertEqual(restored.optimizer,2) # resumes first RL optimizer step
        self.assertIn(('load','resume'), restored.calls)
        self.assertNotEqual(m['artifacts']['sampler'],restored.policy_id)
        eval_backend=FakeBackend(self.backend.registry)
        report=evaluate_checkpoint(final,self.root/'evaluation',backend=eval_backend)
        self.assertEqual(report['resolved'],2)
        self.assertEqual(report['optimizer_step'],3)
        self.assertEqual(eval_backend.trainer_count,0)
        self.assertEqual(report['policy_id'],load_checkpoint(final)['artifacts']['sampler'])
        changed=copy.deepcopy(self.config);changed['seed']+=1
        with self.assertRaises(ValueError): load_checkpoint(final,changed)

    def test_fork_fresh_optimizer_and_lineage(self):
        p=Pipeline(self.config,backend=self.backend)
        path=p.run(stop_after_updates=1)
        fork_config=copy.deepcopy(self.config)
        fork_config['run_id']='fork';fork_config['output']=str(self.root/'fork')
        fork_backend=FakeBackend(self.backend.registry)
        fork=Pipeline(fork_config,backend=fork_backend)
        output=fork.run(path,'fork',stop_after_updates=1)
        self.assertIn(('load','fork'),fork_backend.calls)
        self.assertEqual(fork_backend.optimizer,1)
        self.assertEqual(load_checkpoint(output)['state']['optimizer_step'],1)

    def test_failed_save_never_publishes_manifest(self):
        p=Pipeline(self.config,backend=self.backend)
        p.setup(True);self.backend.create_trainer(42)
        p.commit();previous=read(p.checkpoints.root/'latest.json')
        self.backend.fail_save=True
        with self.assertRaises(InfrastructureError): p.commit()
        self.assertEqual(previous,read(p.checkpoints.root/'latest.json'))
        self.assertEqual(len(list(p.checkpoints.root.glob('*.pending.json'))),1)

    def test_unknown_update_is_not_retried(self):
        self.backend.fail_update=True
        p=Pipeline(self.config,backend=self.backend)
        with self.assertRaises(AmbiguousUpdate): p.run()
        m=load_checkpoint(read(p.checkpoints.root/'latest.json')['path'])
        self.assertEqual(m['state']['optimizer_step'],0)
        self.assertEqual(m['state']['cursor'],0)

    def test_whole_group_retry_and_zero_batch(self):
        p=Pipeline(self.config,backend=self.backend);p.setup(True)
        calls=[]
        def rollout(task,gid,temp):
            t=trajectory(len(calls),0)
            t.group_id=gid
            t.policy_id=self.backend.policy_id
            for g in t.generations:g.policy_id=t.policy_id
            if len(calls)==0:
                t.verification=VerificationResult('unresolved',None,'v1',retryable=True)
            calls.append(t)
            return t
        p.rollout=rollout
        group=p.group({'id':'task'},4,1)
        self.assertEqual(len(calls),8)
        self.assertEqual(len({t.group_id for t in group}),1)
        self.assertNotEqual(calls[0].group_id,group[0].group_id)
        self.assertEqual(grpo_batch([group])[0],[])

    def test_tracking_outage_preserves_events(self):
        class Remote:
            def log(self,*a,**k): raise RuntimeError('offline')
            def finish(self,**k): raise RuntimeError('offline')
        tracker=Tracker(self.root/'tracking',{'mode':'disabled'},'run')
        tracker.remote=Remote()
        tracker.event('update',optimizer_step=1)
        tracker.finish('complete')
        events=[json.loads(x) for x in (tracker.root/'events.jsonl').read_text().splitlines()]
        self.assertEqual(sum(e['event']=='update' for e in events),1)
        self.assertTrue(any(e['event']=='tracking_failure' for e in events))

    def test_spend_is_persisted_before_calls(self):
        ledger=SpendLedger(self.root/'ledger.json',.001,PRICES,86400)
        ledger.reserve('sample',input_tokens=100,output_tokens=10)
        second=SpendLedger(self.root/'ledger.json',.001,PRICES,86400)
        self.assertGreater(second.state['reserved_usd'],0)
        with self.assertRaises(BudgetLimit): second.reserve('checkpoint')
        self.assertEqual(len(second.state['reservations']),1)

    def test_empty_and_ineligible_release_fail_before_backend(self):
        root=self.root/'release';root.mkdir()
        atomic_json(root/'manifest.json',{'training_eligible':False})
        env={'release':str(root),'manifest_sha256':hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest()}
        with self.assertRaisesRegex(ValueError,'not training eligible'):load_release(env)
        bad=copy.deepcopy(self.config);bad['stages'][0]['kind']='ppo'
        with self.assertRaises(ValueError):inputs(bad)

    def test_cleanup_and_budget_terminal(self):
        class Factory(ToyFactory):
            episodes=[]
            def create(self,*args):
                ep=super().create(*args);self.episodes.append(ep);return ep
        factory=Factory();tracker=Tracker(self.root/'runner',{'mode':'disabled'},'run')
        limits={**self.config['limits'],'max_generations':1}
        t=run_episode(self.backend,factory,toy_inputs()['tasks'][0],limits,run_id='r',stage=0,
            group_id='g',episode_id='ep',experiment_hash='x',temperature=1,tracker=tracker)
        self.assertEqual(t.termination,'budget_exhausted')
        self.assertEqual(t.verification.reward,0)
        self.assertTrue(factory.episodes[0].closed)
        self.assertTrue((tracker.root/'trajectories/ep.json').exists())


class RepositoryAdapterTests(unittest.TestCase):
    def test_strict_grading_with_synthetic_judge_and_unknown_cost(self):
        from qa_eval.demo import fixture
        from qa_eval.dataset import freeze
        from qa_eval.security import digest as qa_digest, bindings
        from training_pipeline.repository import RepositoryEpisode
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            task,submission,experiment,metrics,semantic=fixture(root/'source')
            task['split']='train';task['permitted_tools']=['read_file']
            experiment['frozen']=False
            experiment=freeze(experiment,[task],{'synthetic':True})
            semantic.update(bindings(task,submission),experiment_hash=qa_digest(experiment),
                            judge_family=experiment['training_judge_family'])
            class Sandbox:
                closed=False
                def close(self):self.closed=True
            sandbox=Sandbox()
            factory=NS(key=b'x'*32,root=root/'run',judge_call=None)
            episode=RepositoryEpisode({'task':task,'experiment':experiment,'snapshot_root':str(root/'source')},
                                      sandbox,factory,'ep',root/'trajectory.json')
            visible=json.dumps(episode.messages)
            self.assertNotIn('critical_errors',visible)
            self.assertNotIn('claims',json.dumps(json.loads(episode.messages[-1]['content'])))
            episode.usage(10,10)
            t=trajectory();t.submission=submission
            with patch('qa_eval.grading.judge',return_value=semantic):
                result=episode.verify(t)
            self.assertEqual(result.status,'resolved')
            self.assertEqual(result.diagnostics['tier'],'accepted')
            grade=read(factory.root/'private/ep0.grade.json')
            self.assertIsNone(grade['metrics']['cost'])
            episode.close();self.assertTrue(sandbox.closed)

    def test_tools_validate_without_shell_interpolation(self):
        argv=command('search_code',{'query':'$(touch /tmp/bad)'})
        self.assertEqual(argv[:2],['python3','-c'])
        self.assertIn('$(touch /tmp/bad)',argv[3])
        for tool,args in [('read_file',{'path':'a','start_line':True}),('python_probe',{'code':'x','secret':1}),('shell',{})]:
            with self.assertRaises(ValueError):command(tool,args)


@unittest.skipUnless(importlib.util.find_spec('tinker'), 'optional Tinker SDK not installed')
class TinkerAdapterTests(unittest.TestCase):
    def test_real_sdk_datum_and_remote_call_order(self):
        import tinker
        from training_pipeline.tinker_backend import TinkerBackend
        def future(x):return NS(result=lambda **k:x)
        calls=[]
        class Trainer:
            def forward_backward(self,data,loss_fn):
                calls.append(('forward',loss_fn,data))
                scores=[{'logprobs':NS(data=d.loss_fn_inputs['logprobs'].data)} for d in data]
                total=-sum(sum(d.loss_fn_inputs['advantages'].data) for d in data)
                return future(NS(metrics={'loss:sum':total},loss_fn_outputs=scores))
            def optim_step(self,params):calls.append(('optim',params));return future(NS(metrics={}))
        service=NS(get_server_capabilities=lambda:NS(supported_models=[NS(model_name='test',sampleable=True,trainable=True,max_context_length=4096)]),
            create_sampling_client=lambda **kw:NS(get_tokenizer=lambda:Tokenizer()),
            create_lora_training_client=lambda **kw:Trainer())
        config=smoke_config('/tmp/unused',PRICES,'/tmp/unused-ledger')
        model={**config['model'],'base_model':'test'}
        backend=TinkerBackend(model,config['limits'],service=service,sdk=tinker)
        backend.create_trainer(1)
        row=shifted([1,2],[3,4],[.25,-.25],[-.5,-.6])
        backend.update([row],'importance_sampling',1e-5)
        self.assertEqual([c[0] for c in calls],['forward','optim'])
        datum=calls[0][2][0]
        self.assertEqual(datum.loss_fn_inputs['advantages'].data,[0,.25,-.25])
        backend.trainer.optim_step=lambda p:(_ for _ in ()).throw(TimeoutError())
        with self.assertRaises(AmbiguousUpdate):backend.update([row],'importance_sampling',1e-5)
        self.assertTrue(backend.poisoned)
        before=len(calls)
        with self.assertRaises(AmbiguousUpdate):backend.update([row],'importance_sampling',1e-5)
        self.assertEqual(len(calls),before)

class AdditionalContractTests(unittest.TestCase):
    def test_release_hash_and_split_protection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            values={'sft/train.jsonl':[{'id':'s','split':'train','status':'accepted','family_id':'same','lineage_id':'s'}],
                    'rl/train.jsonl':[{'id':'t','split':'train','family_id':'same','lineage_id':'t'}],
                    'evaluation/development/tasks.jsonl':[{'id':'d','split':'development','family_id':'same','lineage_id':'d'}]}
            artifacts={}
            for name,rows in values.items():
                p=root/name;p.parent.mkdir(parents=True,exist_ok=True)
                p.write_text(''.join(json.dumps(r)+'\n' for r in rows));artifacts[name]=hashlib.sha256(p.read_bytes()).hexdigest()
            atomic_json(root/'manifest.json',{'training_eligible':True,'artifacts':artifacts})
            env={'release':str(root),'manifest_sha256':hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest()}
            with self.assertRaisesRegex(ValueError,'crosses splits'):load_release(env)
            (root/'rl/train.jsonl').write_text('')
            with self.assertRaisesRegex(ValueError,'hash mismatch'):load_release(env)

    def test_pristine_tool_read_hash_and_symlink_rejection(self):
        import subprocess
        from agent_harness.repository_tools import READ_SCRIPT
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'a.py').write_text('one\ntwo\nthree\n')
            (root/'link').symlink_to(root/'a.py')
            script=READ_SCRIPT.replace("pathlib.Path('/repo')",'pathlib.Path('+repr(str(root))+')')
            def invoke(tool,args):
                return subprocess.run(['python3','-c',script,json.dumps({'tool':tool,'arguments':args}),json.dumps(['a.py','link'])],capture_output=True,text=True)
            result=invoke('read_file',{'path':'a.py','start_line':1,'end_line':2})
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertEqual(json.loads(result.stdout)['file_sha256'],hashlib.sha256((root/'a.py').read_bytes()).hexdigest())
            for path in ('link','../a.py','/etc/passwd'):
                self.assertNotEqual(invoke('read_file',{'path':path}).returncode,0)

    def test_repository_factory_uses_fresh_modal_instances(self):
        from tests.test_modal_backend import FakeModal,manifest
        from agent_harness.modal_backend import ModalSandboxBackend
        from training_pipeline.repository import RepositoryFactory
        from qa_eval.demo import fixture
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);task,submission,experiment,_,_=fixture(root/'source')
            task['permitted_tools']=['read_file']
            config=smoke_config(root,PRICES,root/'ledger')
            config['environment']={'kind':'repository'}
            modal=FakeModal()
            def backend(*a,**kw):return ModalSandboxBackend(*a,modal_module=modal,**kw)
            factory=RepositoryFactory(config,root/'run',backend_factory=backend)
            row={'task':task,'experiment':experiment,'environment':manifest(),'snapshot_root':str(root/'source')}
            for name in ('ep-a','ep-b'):
                ep=factory.create(row,name,root/(name+'.json'))
                ep.step({'tool':'read_file','arguments':{'path':'executor.py','start_line':1,'end_line':2}})
                ep.close()
            self.assertEqual(len(modal.instances),2)
            for call in modal.Sandbox.create.call_args_list:
                self.assertTrue(call.kwargs['block_network'])
                self.assertNotIn('volumes',call.kwargs)
                self.assertNotIn('secrets',call.kwargs)
            for instance in modal.instances:instance.terminate.assert_called_once()

    def test_unknown_cost_does_not_become_zero_in_report(self):
        from qa_eval.demo import fixture,KEY
        from qa_eval.security import seal
        from qa_eval.grading import evaluate
        from qa_eval.reporting import scorecard
        with tempfile.TemporaryDirectory() as tmp:
            task,submission,config,metrics,semantic=fixture(tmp)
            metrics['cost']=None
            report,_=evaluate(task,submission,seal('EpisodeMetrics',metrics,KEY),config,tmp,KEY,
                              semantic_envelope=seal('SemanticAssessment',semantic,KEY))
            result=scorecard([report])
            self.assertIsNone(result['total_cost_all_attempts'])
            self.assertIsNone(result['cost_per_accepted_answer'])

    def test_checkpoint_manifest_does_not_alias_live_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            backend=FakeBackend();backend.create_trainer(42)
            config=smoke_config(tmp,PRICES,Path(tmp)/'ledger')
            state={'cursor':0,'order':[1,2]}
            _,manifest=Checkpoints(tmp).commit(backend,config,state)
            state['order'].reverse();state['cursor']=2
            self.assertEqual(manifest['state']['cursor'],0)
            self.assertEqual(manifest['state']['order'],[1,2])

class SchedulingAndAccountingTests(unittest.TestCase):
    def test_evaluation_cadence_independent_of_sft_checkpoint_cadence(self):
        with tempfile.TemporaryDirectory() as tmp:
            config=smoke_config(tmp,PRICES,Path(tmp)/'ledger')
            config['limits']['context_tokens']=4096
            config['checkpoint_every']=10
            config['stages']=[{**config['stages'][0],'max_updates':2,'max_batches':2}]
            p=Pipeline(config,backend=FakeBackend());p.run()
            events=[json.loads(l) for l in (Path(config['output'])/'events.jsonl').read_text().splitlines()]
            self.assertEqual(sum(e['event']=='evaluation' for e in events),2)

    def test_shared_ledger_cannot_overwrite_prior_reservations(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'ledger'
            a=SpendLedger(path,5,PRICES,86400)
            b=SpendLedger(path,5,PRICES,86400)
            a.reserve('sample',input_tokens=100,output_tokens=1)
            b.reserve('sample',input_tokens=100,output_tokens=1)
            self.assertEqual(len(read(path)['reservations']),2)
            with self.assertRaises(ValueError):b.reserve('sample',input_tokens=-1)

    def test_direct_grpo_and_repeated_stages(self):
        with tempfile.TemporaryDirectory() as tmp:
            config=smoke_config(tmp,PRICES,Path(tmp)/'ledger')
            config['limits']['context_tokens']=4096
            stage={**config['stages'][1],'max_updates':1,'max_batches':1}
            config['stages']=[stage,stage]
            backend=FakeBackend();p=Pipeline(config,backend=backend);path=p.run()
            self.assertEqual(load_checkpoint(path)['state']['optimizer_step'],2)
            self.assertEqual(backend.trainer_count,2)

    def test_zero_variance_advances_cursor_without_update(self):
        with tempfile.TemporaryDirectory() as tmp:
            config=smoke_config(tmp,PRICES,Path(tmp)/'ledger')
            config['limits']['context_tokens']=4096
            config['stages']=[{**config['stages'][1],'max_updates':1,'max_batches':1}]
            backend=FakeBackend();p=Pipeline(config,backend=backend)
            def group(task,size,temp):return [trajectory(i,1) for i in range(size)]
            p.group=group
            path=p.run();state=load_checkpoint(path)['state']
            self.assertEqual(state['optimizer_step'],0)
            self.assertEqual(state['attempted_batches'],1)
            self.assertFalse(any(c[0]=='update' for c in backend.calls))

    def test_provider_reduction_rejects_wrong_scaling(self):
        from training_pipeline.tinker_backend import check_reduction
        row=shifted([1,2],[3,4],[.25,.25])
        output=NS(loss_fn_outputs=[{'logprobs':NS(data=[0,-2,-2])}],metrics={'loss:sum':1})
        self.assertEqual(check_reduction([row],'cross_entropy',output),1)
        output.metrics['loss:sum']=.5
        with self.assertRaisesRegex(ValueError,'reduction'):check_reduction([row],'cross_entropy',output)
