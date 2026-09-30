import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from dataset_builder.git_source_environment import inventory, source_objects, read_git_file
from dataset_builder.investigate import validated_snapshot_files
from dataset_builder.source_evidence import inspect_reference


class GitSourceEnvironmentTests(unittest.TestCase):
    def test_case_variants_links_submodule_metadata_and_host_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            def git(*args,input=None):return subprocess.run(['git','-C',tmp,*args],input=input,capture_output=True,check=True).stdout
            git('init','--quiet');git('config','user.email','fixture@example.invalid');git('config','user.name','Fixture')
            def blob(data):return git('hash-object','-w','--stdin',input=data).decode().strip()
            a=blob(b'UPPER\n');b=blob(b'lower\n');link=blob(b'.')
            for mode,oid,path in [('100644',a,'a.F90'),('100644',b,'a.f90'),('120000',link,'cycle'),('160000','a'*40,'dependency')]:
                git('update-index','--add','--cacheinfo',mode+','+oid+','+path)
            git('commit','--quiet','-m','source fixture');commit=git('rev-parse','HEAD').decode().strip()
            # No host checkout is needed; two case-distinct objects are retained.
            info=inventory(root,commit)
            self.assertEqual(len(info['snapshot_files']),2)
            self.assertNotEqual(info['snapshot_files']['a.F90'],info['snapshot_files']['a.f90'])
            self.assertEqual(info['symbolic_links']['cycle']['target'],'.')
            self.assertFalse(info['submodules']['dependency']['available'])
            self.assertEqual(read_git_file(root,commit,'a.F90'),b'UPPER\n')
            built={**info,'commit':commit,'capability':'source_reading','source_inventory_mode':'git_objects_primary'}
            self.assertEqual(validated_snapshot_files(root,built),info['snapshot_files'])
            with self.assertRaises(ValueError):validated_snapshot_files(root,{**built,'capability':'execution'})
            with self.assertRaises(ValueError):validated_snapshot_files(root,{**built,'symbolic_links':{}})

    def test_git_reader_checks_hash_without_trusting_host_case_resolution(self):
        with tempfile.TemporaryDirectory() as tmp:
            data=b'one\ntwo\n';hashes={'code.py':hashlib.sha256(data).hexdigest()}
            result=inspect_reference('code.py:1-2',Path(tmp),hashes,blob_reader=lambda p:data)
            self.assertEqual(len(result['evidence']),1)
            bad=inspect_reference('code.py:1-2',Path(tmp),hashes,blob_reader=lambda p:b'changed\n')
            self.assertEqual(bad['unresolved'][0]['reason'],'file_hash_mismatch')


if __name__=='__main__':unittest.main()
