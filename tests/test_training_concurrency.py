from tests.test_claim_grading import request_fixture, judgments
import json
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock
from training_pipeline.concurrency import ordered_map, SamplingClient
from training_pipeline.config import validate_config
from training_pipeline.collection import CollectionFactory
from training_pipeline.contracts import Generation, VerificationResult, ConfigurationError
from training_pipeline.budget import SpendLedger, BudgetLimit
from training_pipeline.storage import Tracker, read
from training_pipeline.orchestrator import Pipeline
from training_pipeline.smoke import smoke_config
from tests.test_training_pipeline import FakeBackend, PRICES, trajectory


class ConcurrencyTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.c=smoke_config(self.root,PRICES,self.root/'ledger.json')
        self.c['concurrency']={'rollouts':4,'judges':2}

    def test_limits_and_stable_order_at_all_benchmark_sizes(self):
        for workers in (1,8,16,32):
            lock=threading.Lock();active=peak=0
            barrier=threading.Barrier(workers)
            def work(i):
                nonlocal active,peak
                with lock:active+=1;peak=max(peak,active)
                try:
                    if i<workers:barrier.wait(timeout=3)
                    time.sleep(.002*(3-i%3));return i
                finally:
                    with lock:active-=1
            self.assertEqual(ordered_map(work,range(workers*2),workers),list(range(workers*2)))
            self.assertEqual(peak,workers);self.assertEqual(active,0)

    def test_failure_drains_active_work_and_stops_dispatch(self):
        barrier=threading.Barrier(4);finished=[];started=[];lock=threading.Lock()
        def work(i):
            with lock:started.append(i)
            try:
                barrier.wait(timeout=3)
                if i==0:raise BudgetLimit('cap')
                time.sleep(.03)
            finally:
                with lock:finished.append(i)
        with self.assertRaises(BudgetLimit):ordered_map(work,range(20),4)
        self.assertEqual(sorted(started),list(range(4)))
        self.assertEqual(sorted(finished),list(range(4)))

    def test_group_retry_keeps_whole_groups_and_one_policy(self):
        p=Pipeline(self.c,backend=FakeBackend(),tracker=Mock())
        lock=threading.Lock();seen={};calls=[]
        def rollout(task,gid,temp):
            with lock:
                seen.setdefault(task['id'],[])
                if gid not in seen[task['id']]:seen[task['id']].append(gid)
                first=task['id']=='bad' and len(seen[task['id']])==1
                calls.append((task['id'],gid))
                t=trajectory(len(calls),.5)
            t.policy_id=p.backend.policy_id;t.group_id=gid
            if first:t.verification=VerificationResult('unresolved',None,'v',retryable=True)
            return t
        p.rollout=rollout
        groups=p.groups([{'id':'bad'},{'id':'good'}],4,1)
        self.assertEqual(len(calls),12)
        self.assertEqual([len(seen[k]) for k in ('bad','good')],[2,1])
        self.assertTrue(all(t.group_id==seen['bad'][1] for t in groups[0]))
        p.rollout=lambda *a:trajectory(0,.5)
        with self.assertRaises(ConfigurationError):p.groups([{'id':'mixed'}],4,1)

    def test_judge_gate_and_local_records(self):
        c=read('examples/training-repository-qwen-nemotron.json');c['concurrency']={'rollouts':8,'judges':2}
        lock=threading.Lock();active=peak=0;barrier=threading.Barrier(2);count=0
        class Judge:
            identity={'base_model':'fake-judge'}
            def sample(self,*args):
                nonlocal active,peak,count
                with lock:active+=1;peak=max(active,peak);count+=1;n=count
                try:
                    if n<=2:barrier.wait(timeout=3)
                    time.sleep(.003)
                    return Generation([1],[2],[-.1],json.dumps(judgments()),'stop','frozen')
                finally:
                    with lock:active-=1
        f=CollectionFactory(c,self.root,None,judge=Judge())
        results=ordered_map(lambda i:f.grade({**request_fixture(), 'episode_id':str(i)}),range(12),8)
        self.assertEqual(len(results),12);self.assertEqual(peak,2);self.assertEqual(active,0)
        self.assertEqual(len(list((self.root/'private').glob('*.judge.json'))),12)

    def test_shared_budget_never_overshoots_and_logs_survive_wandb_failure(self):
        ledger=SpendLedger(self.root/'spend.json',.05,PRICES,172800)
        def reserve(i):
            try:ledger.reserve_external('test',.01);return True
            except BudgetLimit:return False
        results=ordered_map(reserve,range(32),32)
        self.assertEqual(sum(results),5)
        self.assertLessEqual(read(ledger.path)['reserved_usd'],.05)
        tracker=Tracker(self.root,{'mode':'disabled','project':'test'},'test')
        tracker.remote=Mock();tracker.remote.log.side_effect=RuntimeError('outage')
        ordered_map(lambda i:tracker.event('training_step_timing',index=i),range(100),16)
        rows=[json.loads(l) for l in (self.root/'events.jsonl').read_text().splitlines()]
        self.assertEqual(sum(r['event']=='training_step_timing' for r in rows),100)
        self.assertEqual(sum(r['event']=='tracking_failure' for r in rows),100)

    def test_sdk_dispatch_is_serialized_but_futures_overlap(self):
        barrier=threading.Barrier(8);lock=threading.Lock();dispatch=0
        class Client:
            def sample(self):
                nonlocal dispatch
                with lock:
                    dispatch+=1
                    if dispatch!=1:raise AssertionError('concurrent SDK enqueue')
                time.sleep(.001)
                with lock:dispatch-=1
                return NS(result=lambda:barrier.wait(timeout=3))
        client=SamplingClient(Client())
        self.assertEqual(len(ordered_map(lambda _:client.sample().result(),range(8),8)),8)

    def test_invalid_configuration_rejected(self):
        for value in (0,True,1.2,33):
            self.c['concurrency']['rollouts']=value
            with self.assertRaises(ConfigurationError):validate_config(self.c)


    def test_training_update_waits_for_every_sampler(self):
        class Backend(FakeBackend):
            active=0
            peak=0
            lock=threading.Lock()
            def sample(self,*args):
                with self.lock:self.active+=1;self.peak=max(self.peak,self.active)
                try:
                    time.sleep(.003)
                    return super().sample(*args)
                finally:
                    with self.lock:self.active-=1
            def update(self,*args):
                assert self.active==0
                return super().update(*args)
            def save(self,*args):
                assert self.active==0
                return super().save(*args)
            def use_sampler(self,*args):
                assert self.active==0
                return super().use_sampler(*args)
        self.c['stages']=[{'kind':'grpo','max_updates':1,'max_batches':1,'batch_size':2,
                          'group_size':4,'temperature':1,'learning_rate':1e-5}]
        backend=Backend();p=Pipeline(self.c,backend=backend)
        p.run()
        self.assertEqual(backend.active,0)
        self.assertGreater(backend.peak,1)
        self.assertEqual(sum(c[0]=='update' for c in backend.calls),1)

    def test_benchmark_never_allocates_training_and_pins_tasks(self):
        from training_pipeline.benchmark import task_manifest, selected_tasks, benchmark
        from training_pipeline.toy import toy_inputs, ToyFactory
        from training_pipeline.storage import atomic_json
        from unittest.mock import patch
        data=toy_inputs();manifest=task_manifest(data,2)
        path=self.root/'cohort.json';atomic_json(path,manifest)
        self.c['benchmark']={'task_manifest':str(path),'manifest_hash':manifest['manifest_hash'],'attempts':4}
        tasks=selected_tasks(self.c,data)
        self.assertEqual([t['id'] for t in tasks],manifest['task_ids'])
        # Inject synthetic data/environment while exercising the real benchmark and runner.
        self.c['environment']={'kind':'collection'}
        backend=FakeBackend()
        with patch('training_pipeline.benchmark.inputs',return_value=data), patch('training_pipeline.orchestrator.resolve_run_config',side_effect=lambda c:c):
            report=benchmark(self.c,backend=backend,factory=ToyFactory())
        self.assertEqual(report['attempted_episodes'],8)
        self.assertEqual(backend.trainer_count,0)
        self.assertFalse(any(c[0] in {'save','update'} for c in backend.calls))
        self.assertEqual(report['optimizer_updates'],0)
        changed=read(path);changed['task_ids'].reverse();atomic_json(path,changed)
        with self.assertRaises(ConfigurationError):selected_tasks(self.c,data)


    def test_parallel_equal_reward_batch_skips_optimizer(self):
        class ConstantBackend(FakeBackend):
            def sample(self,*args):
                g=super().sample(*args)
                action=json.loads(g.text)
                if 'answer' in action:
                    action['tag']='blue';g.text=json.dumps(action)
                return g
        self.c['stages']=[{'kind':'grpo','max_updates':1,'max_batches':1,'batch_size':2,
                          'group_size':4,'temperature':1,'learning_rate':1e-5}]
        backend=ConstantBackend();p=Pipeline(self.c,backend=backend)
        p.run()
        self.assertFalse(any(c[0]=='update' for c in backend.calls))
