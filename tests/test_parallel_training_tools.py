import copy
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from agent_harness.modal_backend import ModalSandboxBackend, SandboxLimits, BudgetExceeded, SandboxInfrastructureError
from agent_harness.process import CommandResult
from tests.test_modal_backend import FakeModal, manifest
from training_pipeline.collection import CollectionEpisode
from training_pipeline.config import validate_config as validate


def result(text='ok', code=0):
    return CommandResult(text, '', code, .01, len(text), 0, False, False)


class ParallelModalTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.modal = FakeModal()
        self.backend = ModalSandboxBackend(manifest(), self.tmp.name, modal_module=self.modal,
            limits=SandboxLimits(max_tool_calls=3))
        self.episode = self.backend.create('batch')
        self.addCleanup(self.episode.close)

    def test_overlap_order_and_atomic_budget(self):
        barrier = threading.Barrier(2)
        def run(sandbox, argv, *args, **kwargs):
            barrier.wait(timeout=2)
            if argv[0] == 'first': time.sleep(.02)
            return result(argv[0])
        with patch('agent_harness.modal_backend.run_process', side_effect=run):
            values = self.episode.execute_many([['first'], ['second']], max_parallel=2)
        self.assertEqual([v.stdout for v in values], ['first', 'second'])
        self.assertEqual(self.episode._calls, 2)
        with patch('agent_harness.modal_backend.run_process') as run:
            with self.assertRaises(BudgetExceeded):
                self.episode.execute_many([['a'], ['b']], max_parallel=2)
            run.assert_not_called()
        self.assertEqual(self.episode._calls, 2)

    def test_failure_drains_workers_then_closes_without_replay(self):
        barrier = threading.Barrier(2); drained = threading.Event()
        def run(sandbox, argv, *args, **kwargs):
            barrier.wait(timeout=2)
            if argv[0] == 'fail': raise TimeoutError('lost')
            time.sleep(.02); drained.set(); return result()
        with patch('agent_harness.modal_backend.run_process', side_effect=run) as run:
            with self.assertRaises(TimeoutError):
                self.episode.execute_many([['fail'], ['ok']], max_parallel=2)
            self.assertEqual(run.call_count, 2)
        self.assertTrue(drained.is_set()); self.assertTrue(self.episode._closed)
        self.modal.instances[0].terminate.assert_called_once()

    def test_invalid_batch_is_not_dispatched(self):
        for commands in ([], [['valid'], ['bad\0']]):
            with patch('agent_harness.modal_backend.run_process') as run:
                with self.assertRaises(ValueError): self.episode.execute_many(commands)
                run.assert_not_called()
        self.assertEqual(self.episode._calls, 0)


class ParallelCollectionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.row = {'id':'task','public':{'id':'task','permitted_tools':['read_file','search_code','list_files','python_probe'],
            'budgets':{'max_tool_calls':5,'max_output_tokens':3000,'latency_seconds':600,'max_submission_bytes':64000}},
            'image_result':{'snapshot_files':{'a.py':'a'*64}}}
        self.factory = SimpleNamespace(root=Path(self.tmp.name), config={'run_id':'test',
            'environment':{'tool_parallelism':4},'limits':{'max_tool_calls':5}})
        self.sandbox = SimpleNamespace(execute_many=Mock(return_value=[result('first'),result('second')]))
        self.ep = CollectionEpisode(self.row, self.sandbox, self.factory, 'ep', Path(self.tmp.name)/'ep.json')
        self.calls = [{'tool':'read_file','arguments':{'path':'a.py'}}, {'tool':'search_code','arguments':{'query':'x'}}]

    def test_order_budget_evidence_and_prompt(self):
        action={'tool_calls':self.calls}
        self.assertEqual(self.ep.action_tool_count(action), 2)
        done, obs=self.ep.step(action)
        self.assertFalse(done); self.assertEqual(self.ep.tool_calls,2)
        self.assertEqual(self.ep.recorder.record['tool_calls'], ['read_file','search_code'])
        self.assertEqual(self.ep.observed_files,{'a.py':'a'*64})
        self.assertEqual([r['call_index'] for r in obs['tool_results']],[0,1])
        self.assertIn('tool_calls',self.ep.messages[0]['content'])

    def test_prevalidation_no_partial_dispatch(self):
        bads = [[], self.calls*3, [self.calls[0],{'tool':'python_probe','arguments':{'code':'pass'}}],
                [self.calls[0],{'tool':'read_file','arguments':{'path':'missing'}}],
                [self.calls[0],{'answer':'bad'}]]
        for calls in bads:
            with self.assertRaises(ValueError): self.ep.step({'tool_calls':calls})
        self.sandbox.execute_many.assert_not_called(); self.assertEqual(self.ep.tool_calls,0)
        self.assertEqual(self.ep.recorder.record['tool_calls'],[])
        with self.assertRaises(ValueError): self.ep.step({'tool_calls':self.calls,'answer':'bad'})

    def test_missing_result_is_infrastructure_failure(self):
        from training_pipeline.contracts import InfrastructureError
        self.sandbox.execute_many.return_value=[result()]
        with self.assertRaises(InfrastructureError): self.ep.step({'tool_calls':self.calls})
        self.assertEqual(self.ep.observed_files,{})

    def test_remaining_budget_and_disabled(self):
        self.ep.remaining_tool_calls=1
        with self.assertRaises(ValueError): self.ep.step({'tool_calls':self.calls})
        self.ep.remaining_tool_calls=5; self.factory.config['environment']['tool_parallelism']=1
        with self.assertRaises(ValueError): self.ep.step({'tool_calls':self.calls})
        self.sandbox.execute_many.assert_not_called()

    def test_config_width_validation(self):
        config=json.loads(Path('examples/training-repository-qwen-v3.json').read_text())
        for width in (True,0,9,1.5):
            config['environment']['tool_parallelism']=width
            with self.assertRaises(ValueError): validate(config)
        for width in (1,2,4,8):
            config['environment']['tool_parallelism']=width
            validate(config)

class ParallelRunnerTests(ParallelCollectionTests):
    def test_batch_is_one_generation_and_two_accounted_calls(self):
        from agent_harness.training_runner import run_episode
        from training_pipeline.contracts import Generation, VerificationResult
        self.factory.identity='test-env'; self.factory.reward_version='test-reward'
        self.factory.create=Mock(return_value=self.ep)
        self.ep.verify=Mock(return_value=VerificationResult('resolved',1,'test-reward'))
        self.ep.close=Mock()
        backend=SimpleNamespace(policy_id='policy',identity={},sample=Mock(side_effect=[
            Generation([1],[2],[-.1],json.dumps({'tool_calls':self.calls}),'stop','policy'),
            Generation([1],[2],[-.1],json.dumps({'answer':'done','citations':[]}),'stop','policy')]))
        tracker=Mock(root=Path(self.tmp.name))
        limits={'max_tool_calls':5,'max_generations':6,'max_output_tokens':3000,
                'max_tokens_per_call':512,'latency_seconds':600,'max_tool_output_bytes':3500}
        t=run_episode(backend,self.factory,{'id':'task','split':'train'},limits,run_id='run',
            stage=0,group_id='g',episode_id='ep',experiment_hash='h',temperature=1,tracker=tracker)
        self.assertEqual(t.termination,'completed')
        self.assertEqual(t.usage['tool_calls'],2)
        self.assertEqual(t.usage['tool_turns'],1)
        self.assertEqual(t.usage['mean_calls_per_tool_turn'],2)
        self.assertEqual(len(t.generations),2)
        self.assertEqual(self.sandbox.execute_many.call_count,1)
        self.ep.close.assert_called_once()

    def test_shared_observation_cap_preserves_every_call(self):
        from agent_harness.training_runner import observation_text
        from dataclasses import asdict
        obs={'tool_results':[{'call_index':i,'tool':'read_file','arguments':{'path':'a.py'},
             'observation':asdict(result('x'*3500))} for i in range(8)]}
        text=observation_text(obs,3500)
        self.assertLessEqual(len(text.encode()),3500)
        parsed=json.loads(text)
        self.assertEqual([r['call_index'] for r in parsed['tool_results']],list(range(8)))
        self.assertTrue(all(r['observation']['stdout_truncated'] for r in parsed['tool_results']))
        self.assertEqual(len(obs['tool_results'][0]['observation']['stdout']),3500)

if __name__ == '__main__': unittest.main()
