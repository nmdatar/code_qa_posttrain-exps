"""Synthetic Git-object evidence fixtures; never execute repository source."""
import hashlib
from pathlib import Path
import subprocess
import tempfile
import unittest
from qa_eval.source import GitSource
from qa_eval.deterministic import read_evidence, snapshot


class GitSourceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.git('init', '-q')
        for name, raw, mode in [('Name.py', b'def Upper():\n    return 1\n', '100644'),
                                ('name.py', b'def lower():\n    return 2\n', '100644'),
                                ('link.py', b'Name.py', '120000')]:
            sha = self.git('hash-object', '-w', '--stdin', data=raw).decode().strip()
            self.git('update-index', '--add', '--cacheinfo', mode, sha, name)
        self.git('-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid',
                 'commit', '-q', '-m', 'Synthetic evidence fixture')
        self.commit = self.git('rev-parse', 'HEAD').decode().strip()
        self.source = GitSource(self.root, self.commit)

    def tearDown(self):
        self.tmp.cleanup()

    def git(self, *args, data=None):
        return subprocess.run(['git','-C',str(self.root),*args],input=data,
                              capture_output=True,check=True).stdout

    def test_case_sensitive_paths_ignore_dirty_checkout(self):
        self.assertNotEqual(self.source.bytes('Name.py'), self.source.bytes('name.py'))
        (self.root/'Name.py').write_text('dirty working tree')
        self.assertEqual(self.source.bytes('Name.py'), b'def Upper():\n    return 1\n')
        self.assertTrue(snapshot(self.source,self.commit))
        self.assertEqual([x['path'] for x in self.source.catalog()],['Name.py','name.py'])

    def test_reject_links_escapes_nonexact_paths(self):
        for path in ('link.py','../Name.py','/Name.py','./Name.py','NAME.py','dir/../Name.py','dir\\Name.py'):
            with self.subTest(path=path), self.assertRaises(ValueError):
                self.source.bytes(path)

    def test_bound_evidence_hash_lines_and_symbols(self):
        ref={'path':'Name.py','file_sha256':hashlib.sha256(self.source.bytes('Name.py')).hexdigest(),
             'start_line':1,'end_line':2,'symbol':'Upper'}
        _, result=read_evidence(self.source,ref)
        self.assertIn('return 1',result['text'])
        for replacement in ({'file_sha256':'0'*64},{'end_line':3},{'start_line':0},{'symbol':'lower'}):
            with self.subTest(replacement=replacement),self.assertRaises(ValueError):
                read_evidence(self.source,{**ref,**replacement})
        with self.assertRaises(ValueError): snapshot(self.source,'0'*40)
        with self.assertRaises(ValueError): GitSource(self.root,'HEAD')


if __name__=='__main__':unittest.main()
