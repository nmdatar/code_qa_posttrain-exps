import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from dataset_builder.environment import _git_blobs, EnvironmentError


class GitBatchTests(unittest.TestCase):
    def test_real_binary_empty_and_duplicate_objects(self):
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(['git','init','--quiet',tmp],check=True)
            values=[b'',b'hello\n',bytes(range(256))+b'\nabc blob 99\n']
            ids=[subprocess.run(['git','-C',tmp,'hash-object','-w','--stdin'],input=v,capture_output=True,check=True).stdout.decode().strip() for v in values]
            blobs=_git_blobs(Path(tmp),ids+ids)
            self.assertEqual(blobs,dict(zip(ids,values)))

    def test_missing_or_malformed_objects_rejected(self):
        for data in (b'abc missing\n',b'abc tree 0\n\n',b'abc blob 5\nx\n',b'abc blob 0\n\nextra'):
            with self.subTest(data=data),patch('dataset_builder.environment.subprocess.run',return_value=subprocess.CompletedProcess([],0,data,b'')):
                with self.assertRaises(EnvironmentError):_git_blobs(Path('.'),['abc'])

    def test_empty_inventory_needs_no_process(self):
        with patch('dataset_builder.environment.subprocess.run') as call:
            self.assertEqual(_git_blobs(Path('.'),[]),{})
            call.assert_not_called()


if __name__=='__main__':unittest.main()
