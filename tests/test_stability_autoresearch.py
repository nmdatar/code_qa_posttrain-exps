import copy
import fcntl
import json
from pathlib import Path
import tempfile
import unittest
from training_pipeline.autoresearch import worker
from training_pipeline.budget import BudgetLimit, process_lock_path
from training_pipeline.contracts import ConfigurationError
from training_pipeline.storage import atomic_json, digest, semantic_hash, read


class StabilityAutoresearchTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup)
        self.root=Path(temp.name)
        self.calls=[]

    def configs(self):
        configs=[]
        for name, stability in [('control',{}),('width',{'scale_tool_width':True}),('overlong',{'mask_overlong':True})]:
            configs.append({'run_id':name,'output':str(self.root/name),'seed':42,'model':{'rank':8},
                'environment':{},'evaluation':{},'stages':[{'kind':'reinforce','max_batches':8,'max_updates':8,
                    'learning_rate':1e-5,'stability':stability}]})
        return configs

    def factory(self, qualities=None, coverage=None, crash=None, terminations=None):
        owner=self
        qualities=qualities or {'control':.2,'width':.3,'overlong':.25}
        coverage=coverage or {}
        class Offline:
            def __init__(self,config): self.config=config
            def run(self,checkpoint=None,purpose='run'):
                c=self.config; name=c['run_id']; root=Path(c['output'])
                owner.calls.append((name,purpose))
                root.mkdir(parents=True,exist_ok=True)
                if crash=='budget': raise BudgetLimit('bounded')
                if crash=='unsafe': raise KeyboardInterrupt()
                manifest={'id':'ckpt-1','config':c,'config_hash':semantic_hash(c),
                    'identity':{},'artifacts':{},'state':{'optimizer_step':1,'stage':1}}
                manifest['manifest_hash']=digest(manifest)
                path=root/'checkpoints/ckpt-1.json'; atomic_json(path,manifest)
                atomic_json(root/'checkpoints/latest.json',{'path':str(path)})
                events=[]
                for step,q in [(0,.2),(1,qualities[name])]:
                    report={'checkpoint_id':'ckpt-1','optimizer_step':step,'demonstrated_quality':q,
                        'scoring_coverage':coverage.get(name,1),'completion_rate':1,'resolved':32,'expected':32,
                        'cohort':{'name':'selection','manifest_hash':'frozen'},'data_identity':'same','reward_version':'same'}
                    target=root/f'evaluations/{step}.json';atomic_json(target,report)
                    events.append({'event':'evaluation','artifact':str(target)})
                for termination in terminations or []:
                    events.append({'event':'trajectory','phase':'training','termination':termination})
                events.append({'event':'run_finish','status':'complete'})
                with (root/'events.jsonl').open('a') as stream:
                    for event in events: stream.write(json.dumps(event)+'\n')
                if crash=='safe': raise KeyboardInterrupt()
                return path
        return Offline

    def test_promotes_only_matched_eligible_improvement_and_dynamic_order(self):
        result=worker(self.configs(),self.root/'controller',pipeline_factory=self.factory(terminations=['budget_exhausted']))
        self.assertEqual([n for n,_ in self.calls],['control','overlong','width'])
        self.assertEqual(result['incumbent']['run_id'],'width')
        self.assertTrue(result['incumbent']['provisional'])
        self.assertEqual(result['status'],'complete')
        again=worker(self.configs(),self.root/'controller',pipeline_factory=self.factory())
        self.assertEqual(len(self.calls),3)
        self.assertEqual(again['incumbent'],result['incumbent'])

    def test_low_coverage_rejects_promotion_but_continues(self):
        result=worker(self.configs(),self.root/'controller',pipeline_factory=self.factory(coverage={'width':.9}))
        self.assertEqual(len(self.calls),3)
        self.assertEqual(result['incumbent']['run_id'],'overlong')
        self.assertEqual(result['runs'][1]['summary']['eligible'],False)

    def test_missing_control_coverage_never_claims_winner(self):
        result=worker(self.configs(),self.root/'controller',pipeline_factory=self.factory(coverage={'control':.9}))
        self.assertIsNone(result['incumbent'])
        self.assertEqual(result['status'],'complete_inconclusive')

    def test_budget_failure_stops_without_trying_other_candidates(self):
        result=worker(self.configs(),self.root/'controller',pipeline_factory=self.factory(crash='budget'))
        self.assertEqual(result['status'],'stopped_budget')
        self.assertEqual(len(self.calls),1)

    def test_unsafe_crash_never_replays_optimizer(self):
        with self.assertRaises(KeyboardInterrupt):
            worker(self.configs(),self.root/'controller',pipeline_factory=self.factory(crash='unsafe'))
        result=worker(self.configs(),self.root/'controller',pipeline_factory=self.factory())
        self.assertEqual(result['status'],'stopped_ambiguous')
        self.assertEqual(len(self.calls),1)

    def test_durable_finish_recovers_without_rerunning_finished_arm(self):
        with self.assertRaises(KeyboardInterrupt):
            worker(self.configs(),self.root/'controller',pipeline_factory=self.factory(crash='safe'))
        result=worker(self.configs(),self.root/'controller',pipeline_factory=self.factory())
        self.assertEqual(result['status'],'complete')
        self.assertEqual([n for n,_ in self.calls].count('control'),1)

    def test_changed_config_fails_closed(self):
        configs=self.configs(); worker(configs,self.root/'controller',pipeline_factory=self.factory())
        configs[0]['stages'][0]['learning_rate']=5e-6
        with self.assertRaises(ConfigurationError): worker(configs,self.root/'controller',pipeline_factory=self.factory())

    def test_duplicate_controller_flock(self):
        root=self.root/'controller';root.mkdir()
        with process_lock_path(root/'.autoresearch.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            with self.assertRaises(ConfigurationError): worker(self.configs(),root,pipeline_factory=self.factory())
        self.assertEqual(self.calls,[])


    def test_committed_interruption_resumes_same_config(self):
        from training_pipeline.autoresearch import _events
        configs=self.configs()
        base=self.factory()
        owner=self
        class InterruptedOnce(base):
            def run(self,checkpoint=None,purpose='run'):
                path=super().run(checkpoint,purpose)
                if self.config['run_id']=='control' and purpose=='run':
                    event_path=Path(self.config['output'])/'events.jsonl'
                    rows=_events(self.config['output']); rows[-1]['status']='interrupted_at_committed_boundary'
                    event_path.write_text(''.join(json.dumps(e)+'\n' for e in rows))
                    raise KeyboardInterrupt()
                return path
        with self.assertRaises(KeyboardInterrupt):
            worker(configs,self.root/'controller',pipeline_factory=InterruptedOnce)
        result=worker(configs,self.root/'controller',pipeline_factory=base)
        self.assertEqual(result['status'],'complete')
        self.assertEqual(self.calls[:2],[('control','run'),('control','resume')])
        self.assertTrue(result['runs'][0]['receipts'][0]['recovered'])

    def test_infrastructure_trajectory_stops_search(self):
        result=worker(self.configs(),self.root/'controller',pipeline_factory=self.factory(terminations=['infrastructure_error']))
        self.assertEqual(result['status'],'stopped_infrastructure')
        self.assertEqual(len(self.calls),1)
        self.assertIsNone(result['incumbent'])


    def test_runtime_paths_are_matched_by_required_content_hashes(self):
        configs=self.configs()
        for i,c in enumerate(configs):
            c['environment'].update(release=f'/bundle/inputs/{i}/release',manifest_sha256='a'*64)
            c['evaluation'].update(cohort_manifest=f'/bundle/inputs/{i}/cohort_manifest.json',cohort_sha256='b'*64)
            c['harness']={'source_manifest':f'/bundle/inputs/{i}/sources.json','source_manifest_sha256':'c'*64}
        result=worker(configs,self.root/'controller',pipeline_factory=self.factory())
        self.assertEqual(result['status'],'complete')

    def test_runtime_path_normalization_does_not_ignore_hash_mismatch(self):
        for field,key,path_key in [('environment','manifest_sha256','release'),
                                   ('evaluation','cohort_sha256','cohort_manifest'),
                                   ('harness','source_manifest_sha256','source_manifest')]:
            configs=self.configs()
            for i,c in enumerate(configs): c.setdefault(field,{}).update({path_key:f'/bundle/{i}',key:'a'*64})
            configs[1][field][key]='b'*64
            with self.assertRaises(ConfigurationError):
                worker(configs,self.root/'controller',pipeline_factory=self.factory())
        self.assertEqual(self.calls,[])

    def test_runtime_path_requires_hash(self):
        configs=self.configs()
        for i,c in enumerate(configs): c['environment']['release']=f'/bundle/{i}/release'
        with self.assertRaises(ConfigurationError): worker(configs,self.root/'controller',pipeline_factory=self.factory())
        self.assertEqual(self.calls,[])

    def test_training_diagnostics_use_weighted_rewards_and_actual_format_events(self):
        from training_pipeline.autoresearch import _summary, _pick
        configs=self.configs()
        result=worker(configs,self.root/'controller',pipeline_factory=self.factory())
        root=Path(configs[0]['output'])
        trajectory=root/'trajectories/example.json'
        atomic_json(trajectory,{'events':[{'kind':'format_failure'},{'kind':'format_failure'}]})
        with (root/'events.jsonl').open('a') as stream:
            for event in [
                {'event':'training_batch','mean_reward':.25,'resolved_trajectories':4,
                 'masked_overlong_trajectories':2,'width_scaled_trajectories':1},
                {'event':'training_batch','mean_reward':.75,'resolved_trajectories':8,
                 'masked_overlong_trajectories':1,'width_scaled_trajectories':3},
                {'event':'trajectory','phase':'training','termination':'completed','artifact':str(trajectory)}]:
                stream.write(json.dumps(event)+'\n')
        summary=_summary(configs[0],result['runs'][0])
        self.assertAlmostEqual(summary['mean_training_reward'],7/12)
        self.assertEqual(summary['masked_overlong_trajectories'],3)
        self.assertEqual(summary['width_scaled_trajectories'],4)
        self.assertEqual(summary['format_failure_count'],2)
        configs[2]['environment']['invalid_action_policy']='zero-v1'
        result['runs'][0]['summary']=summary
        result['runs'][1]['status']=result['runs'][2]['status']='pending'
        self.assertEqual(_pick(configs,result)[0],2)
