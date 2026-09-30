import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
import modal
from modal.exception import NotFoundError
from training_pipeline.remote import isolated_resources, isolated_names, submit, APP, CONTROLLER, VOLUME
from training_pipeline.contracts import ConfigurationError

class ParallelCampaignTests(unittest.TestCase):
    def manifest(self, ident='a'):
        return {'bundle_id':ident*64,'configs':[{'execution':{'operation':'run','timeout_seconds':100,'cpu':2,'memory_mib':4096},'spend':{'ledger':'/state/artifacts/new.json','cap_usd':320}}], 'ledger_seeds':{'artifacts/new.json':None},'budget_allocation':{'ledger_caps':{'artifacts/new.json':320},'reserves_usd':{},'project_ceiling_usd':320}}

    def test_private_resources_differ_for_every_bundle(self):
        a=isolated_resources(self.manifest());b=isolated_resources(self.manifest('b'))
        self.assertNotEqual(a[0],b[0]);self.assertNotEqual(a[1],b[1])
        self.assertNotEqual(a[0],CONTROLLER);self.assertNotEqual(a[1],VOLUME)
        from modal._utils.name_utils import is_valid_object_name
        for name in a+b:self.assertTrue(is_valid_object_name(name))

    def test_rejects_history_external_dependencies_and_bad_budgets(self):
        variants=[]
        m=self.manifest();m['ledger_seeds']['artifacts/new.json']={'reserved_usd':0};variants.append(m)
        for operation in ('fork','evaluate'):
            m=self.manifest();m['configs'][0]['execution']['operation']=operation;variants.append(m)
        m=self.manifest();m['configs'][0]['execution']['checkpoint']='old/best.json';variants.append(m)
        m=self.manifest();del m['budget_allocation'];variants.append(m)
        m=self.manifest();m['budget_allocation']['ledger_caps']['unrelated.json']=1;variants.append(m)
        m=self.manifest();m['budget_allocation']['project_ceiling_usd']=319;variants.append(m)
        m=self.manifest();m['budget_allocation']['project_ceiling_usd']=float('nan');variants.append(m)
        m=self.manifest();m['validation']='archive-restore-v1';variants.append(m)
        for m in variants:
            with self.subTest(manifest=m),self.assertRaises(ConfigurationError):isolated_resources(m)

    def test_two_campaigns_submit_while_shared_controller_active(self):
        image=Mock()
        for name in ('apt_install','pip_install','add_local_dir','env'):getattr(image,name).return_value=image
        queried=[]
        def lookup(app,name):
            queried.append(name)
            if name==CONTROLLER:return Mock(poll=Mock(return_value=None))
            raise NotFoundError('absent')
        with tempfile.TemporaryDirectory() as tmp,patch.object(modal.Image,'debian_slim',return_value=image),patch.object(modal.Sandbox,'from_name',side_effect=lookup),patch.object(modal.Sandbox,'create',return_value=Mock(object_id='sb-test')) as create,patch.object(modal.App,'lookup'),patch.object(modal.Secret,'from_name'),patch.object(modal.Volume,'from_name') as volume:
            for ident in ('a','b'):
                path=Path(tmp)/ident;path.mkdir()
                with patch('training_pipeline.remote.verify_bundle',return_value=self.manifest(ident)):
                    receipt=submit(path,isolated=True)
                    self.assertEqual(receipt['volume'],isolated_names(ident*64)[1])
                    self.assertTrue(receipt['isolated'])
                    with self.assertRaises(ConfigurationError):submit(path,isolated=True)
            self.assertEqual(create.call_count,2)
            self.assertNotIn(CONTROLLER,queried)
            self.assertEqual(len(set(c.kwargs['name'] for c in create.call_args_list)),2)
            self.assertNotIn(VOLUME,[c.args[0] for c in volume.call_args_list])
            for c in create.call_args_list:self.assertEqual(set(c.kwargs['volumes']),{'/state'})

    def test_same_isolated_campaign_is_guarded(self):
        with tempfile.TemporaryDirectory() as tmp,patch('training_pipeline.remote.verify_bundle',return_value=self.manifest()),patch.object(modal.Sandbox,'from_name',return_value=Mock(poll=Mock(return_value=None))),patch.object(modal.Sandbox,'create') as create:
            with self.assertRaisesRegex(ConfigurationError,'active'):submit(tmp,isolated=True)
            create.assert_not_called()

if __name__=='__main__':unittest.main()
