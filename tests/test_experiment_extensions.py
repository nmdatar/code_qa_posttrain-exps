import copy
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from agent_harness.research_tools import run, command
from training_pipeline.efficiency_reward import feedback
from training_pipeline.harness_variants import ObservationHistory, limit_context
from training_pipeline.remote import validate_execution, operation_estimate


class TierRewardTests(unittest.TestCase):
    def result(self, tier='accepted'):
        return {'status':'resolved','tier':tier,'reward_claim_weights':{'c1':1.,'c2':1.},
                'claims':[{'id':'c1','verdict':'supported','coverage':'complete'},
                          {'id':'c2','verdict':'supported','coverage':'partial'}]}

    def test_ordering_and_only_accepted_get_bonus(self):
        metrics={'input_tokens':100,'output_tokens':50,'tool_seconds':1.}
        scores={}
        for tier in ['failed','partial','accepted']:
            q=feedback(self.result(tier),metrics,1000,'tier-quality-v1')
            e=feedback(self.result(tier),metrics,1000,'tier-efficiency-v1')
            scores[tier]=e['reward']
            if tier != 'accepted':self.assertEqual(q['reward'],e['reward'])
        self.assertEqual(scores['failed'],0.)
        self.assertAlmostEqual(scores['partial'],.15)
        self.assertAlmostEqual(scores['accepted'],.97)
        self.assertEqual(feedback(self.result(),metrics,100,'tier-efficiency-v1')['reward'],.9)

    def test_unresolved_is_not_zero_and_invalid_citations_fail(self):
        unresolved=self.result();unresolved['status']='unresolved'
        self.assertIsNone(feedback(unresolved,{},100,'tier-quality-v1')['reward'])
        metrics={'input_tokens':0,'output_tokens':0,'tool_seconds':0}
        self.assertEqual(feedback(self.result(),metrics,100,'tier-efficiency-v1',['invalid_citation'])['reward'],0.)
        with self.assertRaises(ValueError):feedback(self.result(),{},100,'tier-efficiency-v1')
        with self.assertRaises(ValueError):feedback(self.result('invented'),metrics,100,'tier-quality-v1')


class SourceToolTests(unittest.TestCase):
    def test_source_only_search_hashes_and_isolated_subprocess(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);raw=b'def example(value):\n    return value + 1\n\nanswer = example(3)\n'
            (root/'a.py').write_bytes(raw);(root/'private.json').write_text('SECRET')
            inv={'a.py':hashlib.sha256(raw).hexdigest()}
            defs=run(root,inv,'find_definition',{'symbol':'example'})
            self.assertEqual(defs['matches'][0]['start_line'],1)
            refs=run(root,inv,'find_references',{'symbol':'example'})
            self.assertEqual(refs['matches'][0]['start_line'],4)
            for method in ['lexical-v1','hybrid-subword-v1']:
                result=run(root,inv,'get_context',{'query':'example'},method)
                self.assertEqual(result['snippets'][0]['file_sha256'],inv['a.py'])
                self.assertNotIn('SECRET',str(result))
                self.assertEqual(result,run(root,inv,'get_context',{'query':'example'},method))
            argv=command('find_definition',{'symbol':'example'},inv,'/repo')
            argv[3]=tmp
            actual=json.loads(subprocess.run(argv,check=True,capture_output=True,text=True).stdout)
            self.assertEqual(actual,defs)
            (root/'a.py').write_text('changed')
            with self.assertRaisesRegex(ValueError,'hash'):run(root,inv,'get_context',{'query':'example'})

    def test_arguments_and_symlink_confinement(self):
        for args in [{'symbol':'../secret'},{'symbol':'x','code':'evil'}]:
            with self.assertRaises(ValueError):command('find_definition',args,{},'/repo')
        with self.assertRaises(ValueError):command('get_context',{'query':'x','budget':513},{},'/repo')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'a.py').symlink_to('/etc/passwd')
            with self.assertRaises(ValueError):run(root,{'a.py':'x'},'get_context',{'query':'root'})

    def test_context_budget_uses_tokenizer_and_preserves_source_lines(self):
        tok=SimpleNamespace(encode=lambda s,**kw:list(s))
        obs={'exit_code':0,'stdout':json.dumps({'snippets':[{'path':'a.py','start_line':1,'end_line':30,'file_sha256':'a'*64,
                'lines':[str(i)+': x'*20 for i in range(1,31)]}]})}
        limited=limit_context(obs,tok,512,3500)
        self.assertLessEqual(len(json.dumps(limited)),512)
        r=json.loads(limited['stdout'])['snippets'][0]
        self.assertEqual(r['end_line'],len(r['lines']))
        self.assertEqual(json.loads(obs['stdout'])['snippets'][0]['end_line'],30)

    def test_history_roundtrip_tamper_and_episode_isolation(self):
        with tempfile.TemporaryDirectory() as tmp:
            h=ObservationHistory(tmp,'one')
            first={'stdout':json.dumps({'path':'a.py','start_line':1,'end_line':1,'file_sha256':'hash','lines':['1: value']})}
            messages=[{'role':'assistant','content':'original action'},{'role':'user','content':json.dumps(first)},
                      {'role':'assistant','content':'second action'},{'role':'user','content':'second observation'}]
            h.remember(first,1);h.remember({'stdout':'second'},3);h.compact(messages)
            self.assertEqual(messages[0]['content'],'original action')
            self.assertIn('hash',messages[1]['content'])
            self.assertEqual(json.loads(h.read({'id':'o1'})['text']),first)
            with self.assertRaises(ValueError):ObservationHistory(tmp,'two').read({'id':'o1'})
            (Path(tmp)/'observations/one/o1.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'changed'):h.read({'id':'o1'})


class ForkExecutionTests(unittest.TestCase):
    def test_fork_requires_checkpoint_and_training_estimate(self):
        c=json.loads(Path('configs/experiments/grpo-long-v6/run.json').read_text())
        base=operation_estimate(c)
        c['execution']['operation']='fork'
        with self.assertRaises(ValueError):validate_execution(c['execution'])
        c['execution']['checkpoint']='artifacts/experiments/sft/checkpoints/best.json'
        validate_execution(c['execution'])
        self.assertEqual(operation_estimate(c)['upper_estimate_usd'],base['upper_estimate_usd'])
        c['execution']['checkpoint']='../escape.json'
        with self.assertRaises(ValueError):validate_execution(c['execution'])

class CollectionVariantIntegrationTests(unittest.TestCase):
    def test_tools_are_enabled_in_actual_episode_and_token_bounded(self):
        from dataclasses import dataclass
        from training_pipeline.collection import CollectionEpisode
        @dataclass
        class Result:
            exit_code:int
            stdout:str
            stderr:str=''
            stdout_truncated:bool=False
            stderr_truncated:bool=False
        class Recorder:
            def __init__(self,*args):pass
            def tool(self,name,fn):return fn()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);raw=b'def example(value):\n    return value\n'
            (root/'a.py').write_bytes(raw);h=hashlib.sha256(raw).hexdigest()
            class Sandbox:
                def execute(self,argv):
                    argv[3]=tmp
                    p=subprocess.run(argv,check=True,capture_output=True,text=True)
                    return Result(0,p.stdout)
            spec={'symbols':True,'retrieval':'hybrid-subword-v1','history':'evidence-ledger-v1'}
            factory=SimpleNamespace(root=root,config={'run_id':'test','harness':spec,'environment':{},
                'limits':{'max_tool_output_bytes':3500,'max_tool_calls':5}},
                solver_tokenizer=SimpleNamespace(encode=lambda text,**kw:list(text)))
            row={'id':'task','public':{'id':'task','permitted_tools':['read_file'],'budgets':{'max_tool_calls':5}},
                 'image_result':{'snapshot_files':{'a.py':h}}}
            with patch('training_pipeline.collection.EpisodeRecorder',Recorder):
                episode=CollectionEpisode(row,Sandbox(),factory,'one',root/'trace.json')
                self.assertNotIn('get_context',row['public']['permitted_tools'])
                self.assertIn('get_context',episode.row['public']['permitted_tools'])
                done,obs=episode.step({'tool':'get_context','arguments':{'query':'example','budget':512}})
                self.assertFalse(done);self.assertLessEqual(len(json.dumps(obs)),512)
                self.assertEqual(episode.observed_files,{'a.py':h})
                episode.messages.append({'role':'user','content':json.dumps(obs)})
                episode.remember_observation(obs)
                _,restored=episode.step({'tool':'read_observation','arguments':{'id':'o1'}})
                self.assertEqual(json.loads(restored['text']),obs)
                with self.assertRaises(ValueError):episode.step({'tool':'python_probe','arguments':{'code':'pass'}})

    def test_tier_training_does_not_use_independent_positive_reward_pass(self):
        from training_pipeline.collection import CollectionFactory
        c=json.loads(Path('configs/experiments/grpo-long-v6/run.json').read_text())
        c['training_reward']['version']='tier-quality-v1'
        factory=CollectionFactory(c,Path('/tmp'),None,judge=object())
        strict={'status':'resolved','score':1.,'tier':'accepted'}
        request={'episode_id':'test','source_row':{'split':'train'},'rubric':{'claims':[]}}
        with patch('training_pipeline.strict_grading.assess',return_value=strict), \
             patch('training_pipeline.coverage_judge.assess_coverage',side_effect=AssertionError('Wrong reward path')), \
             patch('training_pipeline.collection.atomic_json'):
            self.assertEqual(factory._grade(request),strict)


class InvestigationAdmissionTests(unittest.TestCase):
    def test_rejects_stale_judge_and_failed_answer_before_tool_admission(self):
        from training_pipeline.investigation_sft import candidate
        row={'rubric':{'rubric_hash':'current'}}
        trace={'termination':'completed','verification':{'status':'resolved','version':'old',
            'diagnostics':{'strict_status':'resolved','strict_score':1.,'rubric_hash':'current'}}}
        with self.assertRaisesRegex(ValueError,'matched current'):candidate(trace,row,'new')
        trace['verification']['version']='new';trace['verification']['diagnostics']['strict_score']=0.
        with self.assertRaisesRegex(ValueError,'strict-passing'):candidate(trace,row,'new')
