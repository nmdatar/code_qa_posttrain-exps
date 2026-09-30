import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, Mock

from training_pipeline.remote import (validate_execution, require_remote_worker,
    reconcile_ledger, verify_bundle, relative, controller_reservation, operation_estimate)
from training_pipeline.storage import digest, read
from training_pipeline.contracts import ConfigurationError

ROOT = Path(__file__).resolve().parents[1]


class RemoteExperimentsTests(unittest.TestCase):
    def config(self):
        return read(ROOT/'configs/experiments/02-direct-grpo-screen.json')

    def test_local_production_execution_is_rejected_before_remote_allocation(self):
        from training_pipeline.orchestrator import Pipeline
        c=self.config()
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ,{},clear=True):
            c['output']=str(Path(tmp)/'run')
            p=Pipeline(c,data={'identity':'fixture','tasks':[],'development':[],'sft':[]})
            with self.assertRaisesRegex(ConfigurationError,'only on Modal'):
                p.setup(True)
            self.assertFalse(Path(c['output']).exists())

    def test_no_local_fallback(self):
        c=self.config()
        with patch.dict(os.environ,{},clear=True), self.assertRaises(ConfigurationError):
            require_remote_worker(c)
        with patch.dict(os.environ,{'QA_MODAL_WORKER':'1'}):
            require_remote_worker(c)
            del c['execution']
            with self.assertRaises(ConfigurationError):require_remote_worker(c)

    def test_resource_and_path_validation(self):
        c=self.config();validate_execution(c['execution'])
        for path in ('/tmp/ledger','../ledger','artifacts/../../ledger'):
            with self.assertRaises(ConfigurationError):relative(path)
        for key,value in [('kind','local'),('cpu',True),('timeout_seconds',86401)]:
            bad=copy.deepcopy(c['execution']);bad[key]=value
            with self.assertRaises(ConfigurationError):validate_execution(bad)

    def test_cloud_ledger_cannot_reset_or_diverge_from_local(self):
        seed={'cap':60,'prices':{},'ttl_seconds':172800,'reserved_usd':3,
              'reservations':[{'kind':'sample','upper_estimate_usd':3}]}
        remote=copy.deepcopy(seed);remote['reserved_usd']=4
        remote['reservations'].append({'kind':'sample','upper_estimate_usd':1})
        reconcile_ledger(remote,seed)
        with self.assertRaises(ConfigurationError):reconcile_ledger(seed,remote)
        remote['reservations'][0]['kind']='other'
        with self.assertRaises(ConfigurationError):reconcile_ledger(remote,seed)

    def test_bundle_integrity_rejects_extra_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);m={'files':{},'configs':[]};m['bundle_id']=digest(m)
            (root/'bundle.json').write_text(json.dumps(m));verify_bundle(root)
            (root/'secret.txt').write_text('must not upload')
            with self.assertRaises(ConfigurationError):verify_bundle(root)

    def test_all_experiment_configs_pin_remote_and_strict_qwen_judge(self):
        for path in (ROOT/'configs/experiments').glob('*.json'):
            c=read(path)
            if 'environment' not in c:continue
            validate_execution(c['execution'])
            self.assertEqual(c['environment']['grading_version'],'all-claims-v3')
            self.assertEqual(c['judge']['base_model'],'Qwen/Qwen3.5-397B-A17B')
            self.assertEqual(c['judge']['max_tokens'],4096)
            self.assertIn('all-claims-v3-modal',c['run_id'])
            self.assertGreater(controller_reservation(c),0)
            if c['execution']['operation']=='run':
                self.assertLessEqual(operation_estimate(c)['upper_estimate_usd']+controller_reservation(c),c['spend']['cap_usd'])

    def test_strict_estimate_includes_both_judge_calls(self):
        from training_pipeline.launch import estimate
        c=self.config();strict=estimate(c,c['spend']['prices'])
        c['environment']['grading_version']='source-claims-v2'
        legacy=estimate(c,c['spend']['prices'])
        self.assertEqual(strict['components_usd']['reference_grading'],2*legacy['components_usd']['reference_grading'])

    def test_prepare_packages_exact_git_objects_and_remaps_only_runtime_paths(self):
        import contextlib
        import subprocess
        from qa_eval.demo import fixture
        from training_pipeline.remote import prepare
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);repo=root/'repo';task,*_=fixture(repo)
            release=root/'release';release.mkdir();(release/'manifest.json').write_text('{}')
            cohort=root/'cohort.json';cohort.write_text('{}')
            c=self.config();c['environment']['release']=str(release)
            c['evaluation']['cohort_manifest']=str(cohort)
            c['output']='artifacts/new-run';c['spend']['ledger']='artifacts/budget.json'
            config=root/'config.json';config.write_text(json.dumps(c))
            data={'tasks':[{'snapshot_root':str(repo),'public':{'repository':task['repository']}}],
                  'development':[],'sft':[],'identity':'fixture'}
            with contextlib.chdir(root), patch('training_pipeline.config.inputs',return_value=data), patch('training_pipeline.remote.operation_estimate',return_value={'upper_estimate_usd':1}):
                result=prepare([config],root/'bundle')
            m=verify_bundle(root/'bundle')
            self.assertEqual(result['bundle_id'],m['bundle_id'])
            self.assertEqual(m['configs'][0]['environment']['manifest_sha256'],c['environment']['manifest_sha256'])
            self.assertEqual(m['configs'][0]['spend']['ledger'],'/state/artifacts/budget.json')
            archive=root/'bundle/snapshots'/m['snapshots'][str(repo)]
            clone=root/'clone'
            subprocess.run(['git','clone','--bare',str(archive),str(clone)],check=True,capture_output=True)
            content=subprocess.run(['git','-C',str(clone),'show',task['repository']['commit']+':executor.py'],check=True,capture_output=True).stdout
            self.assertEqual(content,(repo/'executor.py').read_bytes())
            (root/'bundle/code/training_pipeline/remote.py').write_text('changed')
            with self.assertRaises(ConfigurationError):verify_bundle(root/'bundle')

    def test_submit_detaches_named_sandbox_and_refuses_repeat(self):
        import modal
        from modal.exception import NotFoundError
        from training_pipeline.remote import submit, APP, CONTROLLER
        with tempfile.TemporaryDirectory() as tmp:
            bundle=Path(tmp)/'bundle';bundle.mkdir()
            c=self.config();manifest={'bundle_id':'a'*64,'configs':[c]}
            image=Mock()
            for name in ('apt_install','pip_install','add_local_dir','env'):
                getattr(image,name).return_value=image
            with patch('training_pipeline.remote.verify_bundle',return_value=manifest), \
                 patch.object(modal.Image,'debian_slim',return_value=image), \
                 patch.object(modal.Sandbox,'from_name',side_effect=NotFoundError('missing')), \
                 patch.object(modal.Sandbox,'create',return_value=Mock(object_id='sb-fixture')) as create, \
                 patch.object(modal.App,'lookup'),patch.object(modal.Secret,'from_name'), \
                 patch.object(modal.Volume,'from_name'):
                result=submit(bundle)
                self.assertEqual(result['sandbox_id'],'sb-fixture')
                self.assertEqual(create.call_args.kwargs['name'],CONTROLLER)
                self.assertEqual(create.call_args.args,('python','-m','training_pipeline.remote','worker'))
                self.assertIn('/state',create.call_args.kwargs['volumes'])
                with self.assertRaises(ConfigurationError):submit(bundle)
                self.assertEqual(create.call_count,1)

    def test_worker_orders_benchmarks_before_parallel_training_and_persists_results(self):
        import modal
        from training_pipeline.remote import worker
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);configs=[]
            for i,operation in enumerate(('benchmark','run','benchmark','run')):
                c=self.config();c['run_id']='job'+str(i)
                c['execution']['operation']=operation
                c['output']=str(root/'artifacts'/('output'+str(i)))
                c['spend']['ledger']=str(root/'artifacts'/('ledger'+str(i)+'.json'))
                configs.append(c)
            manifest={'bundle_id':'b'*64,'configs':configs,'snapshots':{},'ledger_seeds':{},'parallel_training':True}
            calls=[]
            def run(command,**kwargs):
                
                if command[0] != 'sync':calls.append(command[3])
                return Mock(returncode=0)
            with patch.dict(os.environ,{'QA_MODAL_WORKER':'1'}), \
                 patch('training_pipeline.remote.verify_bundle',return_value=manifest), \
                 patch('training_pipeline.remote.operation_estimate',return_value={'upper_estimate_usd':1}), \
                 patch('training_pipeline.remote.subprocess.run',side_effect=run), \
                 patch.object(modal.Volume,'from_name'):
                worker(state_root=tmp)
            self.assertEqual(calls[:2],['benchmark','benchmark'])
            self.assertEqual(calls[2:],['run','run'])
            report=read(root/'campaigns'/('b'*64)/'status.json')
            self.assertEqual(report['status'],'complete')
            self.assertEqual(len(report['results']),4)
            self.assertGreater(read(configs[0]['spend']['ledger'])['reserved_usd'],0)

    def test_remote_locks_are_local_to_singleton_controller(self):
        from training_pipeline.budget import process_lock_path
        with patch.dict(os.environ,{'QA_MODAL_WORKER':'1'}):
            first=process_lock_path('/state/artifacts/budget.json.lock')
            self.assertFalse(str(first).startswith('/state/'))
            self.assertEqual(first,process_lock_path('/state/artifacts/budget.json.lock'))
            self.assertNotEqual(first,process_lock_path('/state/artifacts/other.json.lock'))
