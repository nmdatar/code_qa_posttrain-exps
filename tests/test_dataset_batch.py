import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from dataset_builder.batch import package, audit_batch
from dataset_builder.build import write_json
from dataset_builder.contracts import canonical_hash
from tests import test_dataset_build as build_tests


class BatchReleaseTests(unittest.TestCase):
    def setUp(self):
        fixture=build_tests.DatasetBuildTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.root=fixture.base
        bundle, manifest=fixture.prepare_spec()
        self.bundle=bundle
        env=json.loads((bundle/'public/environment.json').read_text())
        built={'status':'ready','commit':fixture.commit,'backend':'modal','image_digest':'im-test',
               'source_environment_sha256':canonical_hash(env),'recipe_sha256':'1'*64,
               'snapshot_sha256':'2'*64,'tool_version':'test','capability':'execution'}
        self.built=built
        write_json(bundle/'environment-build/result.json',built)
        write_json(bundle/'private/verification-report.json',{'status':'passed',
                   'bundle_hash':canonical_hash(manifest),'environment_hash':canonical_hash(built),
                   'assertions':[{'status':'passed','task_id':'timeout','assertion_id':'timeout-value'}]})
        write_json(bundle/'private/isolation-report.json',{'passed':True,'environment_hash':canonical_hash(built)})
        write_json(bundle/'private/image-attestation.json',{'status':'passed','environment_hash':canonical_hash(built)})
        self.config={'id':'test-dev','families':[{'family_id':'example/repository',
                     'repository':'https://github.com/example/repository','commit':fixture.commit,
                     'split':'development','bundle':'output'}],'release':'release'}

    def test_only_development_exports_and_deterministic_artifacts(self):
        a=package(self.config,self.root,self.root/'a')
        b=package(self.config,self.root,self.root/'b')
        self.assertEqual(a,b)
        self.assertFalse(a['training_eligible'])
        self.assertEqual(a['counts']['development'],1)
        self.assertEqual((self.root/'a/sft/train.jsonl').read_text(),'')
        self.assertNotIn('PRIVATE_', (self.root/'a/evaluation/development/tasks.jsonl').read_text())
        for path,digest in a['artifacts'].items():
            self.assertEqual(hashlib.sha256((self.root/'a'/path).read_bytes()).hexdigest(),digest)

    def test_training_assignment_cannot_enter_development_export(self):
        self.config['families'][0]['split']='train'
        with self.assertRaisesRegex(ValueError,'development-only'):audit_batch(self.config,self.root)

    def test_stale_and_empty_probe_reports_block_release(self):
        path=self.bundle/'private/verification-report.json'
        report=json.loads(path.read_text());report['environment_hash']='0'*64;write_json(path,report)
        with self.assertRaisesRegex(ValueError,'Stale assertion'):audit_batch(self.config,self.root)
        report['environment_hash']=canonical_hash(self.built);report['assertions']=[];write_json(path,report)
        with self.assertRaisesRegex(ValueError,'Assertions must pass'):audit_batch(self.config,self.root)

    def test_changed_private_references_and_unbound_isolation_block_release(self):
        path=self.bundle/'private/isolation-report.json';write_json(path,{'passed':True})
        with self.assertRaisesRegex(ValueError,'Isolation'):audit_batch(self.config,self.root)
        write_json(path,{'passed':True,'environment_hash':canonical_hash(self.built)})
        (self.bundle/'private/tasks.jsonl').write_text('{}\n')
        with self.assertRaisesRegex(ValueError,'artifact changed'):audit_batch(self.config,self.root)


if __name__=='__main__':unittest.main()
