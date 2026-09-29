import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from training_pipeline.collection import CollectionFactory, CollectionEpisode, sandbox_manifest, modal_reservation
from training_pipeline.contracts import Generation, Trajectory
from training_pipeline.storage import read
from training_pipeline.budget import SpendLedger, BudgetLimit
from training_pipeline.launch import estimate
from tests.test_modal_backend import FakeModal
from agent_harness.modal_backend import ModalSandboxBackend
from training_pipeline.admission import sha


class CollectionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.config = read('examples/training-repository-qwen-v3.json')
        self.repo = {'commit':'a'*40,'family_id':'org/repo','url':'https://github.com/org/repo'}
        self.row = {'id':'task','split':'train','public':{'id':'task','split':'train','repository':self.repo,
            'permitted_tools':['read_file'],'user_prompt':'Question',
            'budgets':{'max_tool_calls':5,'max_output_tokens':3000,'latency_seconds':600,'max_submission_bytes':64000}},
            'environment':{'image_id':'im-test','snapshot_sha256':'b'*64,'repository':self.repo},
            'image_result':{'readiness_command':['test','-d','/workspace'],'readiness':{'exit_code':0},'snapshot_files':{'a.py':sha(b'pass\n')}},
            'snapshot_root':'/unused','reference':{'reference_answer':'SECRET REFERENCE'},'excerpts':[]}
        self.modal = FakeModal()
        self.modal.Sandbox.create.side_effect = lambda *args, **kwargs: self.modal.create(**kwargs)
        self.stream = patch('agent_harness.modal_backend.StreamProcess')
        mocked = self.stream.start(); self.addCleanup(self.stream.stop)
        from agent_harness.process import CommandResult
        mocked.return_value.run.return_value = CommandResult('ok','',0,.01,2,0,False,False)
        self.ledger = SpendLedger(self.root/'spend.json',5,self.config['spend']['prices'],86400)
        self.factory = CollectionFactory(self.config,self.root,self.ledger,backend_factory=lambda m,p,**kw:ModalSandboxBackend(m,p,modal_module=self.modal,**kw))

    def episode(self, name='ep'):
        return self.factory.create(self.row,name,self.root/(name+'.json'))

    def test_concurrent_modal_episodes_are_isolated_and_cleaned(self):
        import threading
        from training_pipeline.concurrency import ordered_map
        barrier = threading.Barrier(4)
        def episode(i):
            e = self.episode('parallel-'+str(i))
            try:
                barrier.wait(timeout=3)
                self.assertNotIn('SECRET', json.dumps(e.messages))
            finally:
                e.close()
        ordered_map(episode, range(4), 4)
        self.assertEqual(len(self.modal.instances),4)
        self.assertEqual(len({id(s) for s in self.modal.instances}),4)
        for sandbox in self.modal.instances:sandbox.terminate.assert_called_once()

    def test_workspace_private_separation_freshness_and_reservation(self):
        a,b=self.episode('a'),self.episode('b')
        self.assertNotIn('SECRET',json.dumps(a.messages))
        self.assertEqual(self.modal.Sandbox.create.call_args.kwargs['workdir'],'/workspace')
        self.assertEqual(self.modal.Sandbox.create.call_args.args[:4], ('python3', '-u', '-I', '-c'))
        self.assertEqual(len(self.ledger.state['reservations']),2)
        a.close();b.close()
        for sandbox in self.modal.instances:sandbox.terminate.assert_called_once()

    def test_real_verifier_and_telemetry_with_injected_judge(self):
        episode=self.episode()
        self.factory.judge=type('Judge',(),{'sample':lambda s,m,n,t:Generation([1],[2],[-.1],json.dumps({'status':'resolved','score':.75,'reason':'supported'}),'stop','base:judge')})()
        answer={'schema_version':'1.0','task_id':'task','text':'It passes.', 'diagram':None,
                'citations':[{'id':'c','path':'a.py','start_line':1,'end_line':1,'file_sha256':sha(b'pass\n')}]}
        self.assertTrue(episode.step({'answer':answer})[0])
        trajectory=Trajectory('r',0,'task','h','g','ep','p','e','x','train',submission=answer,termination='completed')
        with patch('training_pipeline.collection.blob',return_value=b'pass\n'):
            result=episode.verify(trajectory)
        self.assertEqual(result.reward,.75)
        self.assertFalse(result.diagnostics['calibrated'])
        self.assertTrue((self.root/'private/ep.telemetry.json').exists())
        episode.close();self.modal.instances[0].terminate.assert_called_once()

    def test_invalid_judge_remains_unresolved(self):
        episode=self.episode()
        self.factory.grade=lambda request:(_ for _ in ()).throw(ValueError('bad json'))
        answer={'schema_version':'1.0','task_id':'task','text':'answer','diagram':None,'citations':[{'id':'c','path':'a.py','start_line':1,'end_line':1,'file_sha256':sha(b'pass\n')}]}
        trajectory=Trajectory('r',0,'task','h','g','ep','p','e','x','train',submission=answer,termination='completed')
        with patch('training_pipeline.collection.blob',return_value=b'pass\n'):
            result=episode.verify(trajectory)
        self.assertEqual(result.status,'unresolved');self.assertIsNone(result.reward)
        self.modal.instances[0].terminate.assert_called_once()

    def test_budget_exhaustion_before_modal_allocation(self):
        self.ledger.reserve_external('used',5)
        with self.assertRaises(BudgetLimit):self.episode()
        self.modal.Sandbox.create.assert_not_called()

    def test_estimate_covers_all_services_and_retries(self):
        plan=estimate(self.config,self.config['spend']['prices'])
        self.assertEqual(plan['status'],'within_ceiling')
        self.assertGreaterEqual(plan['episodes_including_retries'],10)
        self.assertGreater(plan['components_usd']['modal_sandboxes'],0)
        self.config['stages'][0]['max_batches']=100
        self.assertEqual(estimate(self.config,self.config['spend']['prices'])['status'],'over_ceiling')

class StreamTransportTests(unittest.TestCase):
    def test_real_server_multiple_commands_and_bounded_output(self):
        import subprocess,sys
        from types import SimpleNamespace
        from agent_harness.stream_process import SERVER,StreamProcess
        process=subprocess.Popen([sys.executable,'-u','-I','-c',SERVER],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        class Writer:
            def write(self,value):process.stdin.write(value.encode())
            def drain(self):process.stdin.flush()
        transport=StreamProcess(SimpleNamespace(stdin=Writer(),stdout=process.stdout))
        try:
            a=transport.run([sys.executable,'-c','print("a"*1000)'],5,20)
            b=transport.run([sys.executable,'-c','print("second")'],5,20)
            self.assertEqual(len(a.stdout),20);self.assertTrue(a.stdout_truncated)
            self.assertEqual(b.stdout,'second\n')
            self.assertEqual(b.exit_code,0)
        finally:
            process.terminate();process.wait(timeout=5)
            for stream in (process.stdin,process.stdout,process.stderr):stream.close()

class ActionRepairTests(unittest.TestCase):
    def test_fences_and_envelope_repair_preserve_answer(self):
        episode=object.__new__(CollectionEpisode)
        episode.row={'id':'real-task'}
        answer={'task_id':'TASK_ID','text':'unchanged answer','citations':[]}
        raw='```json\n'+json.dumps({'answer':answer})+'\n```'
        action=episode.parse_action(raw)
        self.assertEqual(action['answer']['text'], 'unchanged answer')
        self.assertEqual(action['answer']['task_id'], 'real-task')
        self.assertIsNone(action['answer']['diagram'])
        self.assertEqual(answer['task_id'],'TASK_ID')
        bad=episode.parse_action(json.dumps({'answer':{**answer,'task_id':'wrong-task'}}))
        self.assertEqual(bad['answer']['task_id'],'wrong-task')

    def test_compact_citation_metadata_is_bound_to_observed_source(self):
        episode=object.__new__(CollectionEpisode)
        episode.row={'id':'real-task'}
        episode.observed_files={'a.py':'a'*64}
        result=episode.parse_action(json.dumps({'answer':'Unchanged words','citations':[{'path':'a.py','start_line':2,'end_line':3}]}))
        self.assertEqual(result['answer']['text'],'Unchanged words')
        self.assertEqual(result['answer']['citations'][0]['file_sha256'],'a'*64)
        with self.assertRaises(ValueError):
            episode.parse_action(json.dumps({'answer':'text','citations':[{'path':'unseen','start_line':1,'end_line':2}]}))

class RetentionBudgetTests(unittest.TestCase):
    def test_checkpoint_retention_is_reserved_per_save(self):
        with tempfile.TemporaryDirectory() as tmp:
            c=read('examples/training-repository-qwen-v3.json')
            ledger=SpendLedger(Path(tmp)/'spend.json',5,c['spend']['prices'],86400)
            day=ledger.estimate('checkpoint')
            hour=ledger.reserve('checkpoint',ttl_seconds=3600)
            self.assertAlmostEqual(hour,day/24)
            self.assertEqual(ledger.state['reservations'][0]['ttl_seconds'],3600)
            self.assertEqual(ledger.state['ttl_seconds'],86400)

class JudgeSyntaxTests(unittest.TestCase):
    def test_only_unambiguous_numeric_quote_is_repaired(self):
        from training_pipeline.collection import parse_judge
        value,repaired=parse_judge('{"status":"resolved","score":0.75","reason":"supported"}')
        self.assertTrue(repaired);self.assertEqual(value['score'],.75)
        with self.assertRaises(ValueError):parse_judge('{"status":"resolved","score":2.0","reason":"unsupported"}')
        with self.assertRaises(ValueError):parse_judge('{"status":"resolved","score":,"reason":"missing"}')
        value,repaired=parse_judge('{"status":"unresolved","score":null,"reason":"uncertain"}')
        self.assertFalse(repaired);self.assertIsNone(value['score'])

class GradeRecoveryGuards(unittest.TestCase):
    def test_prior_optimizer_update_blocks_recovery(self):
        from training_pipeline.recover_grades import prepare
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'config.json').write_text('{}')
            (root/'events.jsonl').write_text(json.dumps({'event':'update'})+'\n')
            with self.assertRaisesRegex(ValueError,'no optimizer updates'):
                prepare(root,root/'new')
