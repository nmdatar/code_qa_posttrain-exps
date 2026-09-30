import copy
import hashlib
import json
import unittest

from dataset_builder.build import write_json
from dataset_builder.contracts import canonical_hash
from dataset_builder.environment import _export_snapshot
from dataset_builder.reuse import reuse_environment
from tests import test_dataset_build as fixtures


class EnvironmentReuseTests(unittest.TestCase):
    def setUp(self):
        fixture = fixtures.DatasetBuildTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.fixture = fixture
        self.source, _ = fixture.prepare_spec(name='source')
        spec = copy.deepcopy(fixture.spec)
        spec['tasks'][0]['id'] = 'another-task'
        self.target, _ = fixture.prepare_spec(spec, name='target')
        env = json.loads((self.source/'public/environment.json').read_text())
        self.built = {'status':'ready','backend':'modal','image_id':'im-test',
                      'commit':fixture.commit, 'source_environment_sha256':canonical_hash(env),
                      'readiness':{'exit_code':0,'timed_out':False,'truncated':False},
                      'snapshot_files':_export_snapshot(fixture.repo, fixture.commit)[0]}
        self.save_built()
        for name in ('environment.json','recipe.json'):
            write_json(self.source/'environment-build'/name,{})
        write_json(self.source/'private/verification-report.json',{'DO_NOT_COPY':True})

    def save_built(self):
        write_json(self.source/'environment-build/result.json',self.built)
        digest=canonical_hash(self.built)
        write_json(self.source/'private/isolation-report.json',{'passed':True,'environment_hash':digest})
        write_json(self.source/'private/image-attestation.json',{'status':'passed','environment_hash':digest})

    def test_reuse_requires_new_task_verification(self):
        result=reuse_environment(self.source,self.target)
        self.assertFalse(result['task_verification_reused'])
        self.assertFalse((self.target/'private/verification-report.json').exists())
        self.assertEqual(json.loads((self.target/'environment-build/result.json').read_text()),self.built)
        with self.assertRaisesRegex(ValueError,'already has'):reuse_environment(self.source,self.target)

    def test_stale_attestation_rejected(self):
        write_json(self.source/'private/image-attestation.json',{'status':'passed','environment_hash':'wrong'})
        with self.assertRaisesRegex(ValueError,'stale'):reuse_environment(self.source,self.target)
        self.assertFalse((self.target/'environment-build').exists())

    def test_source_mismatch_rejected(self):
        self.built['snapshot_files']={}
        self.save_built()
        with self.assertRaisesRegex(ValueError,'Snapshot differs'):reuse_environment(self.source,self.target)

    def test_modified_prompt_rejected(self):
        (self.target/'public/tasks.jsonl').write_text('{}\n')
        with self.assertRaisesRegex(ValueError,'Artifact changed'):reuse_environment(self.source,self.target)

    def test_recipe_change_even_with_updated_manifest_rejected(self):
        path=self.target/'public/environment.json'
        env=json.loads(path.read_text());env['recipe']['base_image']='different:version';write_json(path,env)
        p=self.target/'manifest.json';m=json.loads(p.read_text())
        m['artifacts']['public/environment.json']=hashlib.sha256(path.read_bytes()).hexdigest();write_json(p,m)
        with self.assertRaisesRegex(ValueError,'must match exactly'):reuse_environment(self.source,self.target)


if __name__=='__main__':unittest.main()
