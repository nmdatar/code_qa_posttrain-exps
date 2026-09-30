"""Read-only, case-sensitive Git object evidence, independent of working tree."""
import hashlib
import os
from pathlib import PurePosixPath
import re
import subprocess
from .security import digest

class GitSource:
    def __init__(self, repository, commit, catalog_paths=None):
        if not re.fullmatch(r'[a-f0-9]{40,64}',commit): raise ValueError('Full immutable commit required')
        self.repository, self.commit = str(repository), commit
        self.catalog_paths = None if catalog_paths is None else frozenset(catalog_paths)
        self.tree = self._git('rev-parse',commit+'^{tree}').decode().strip()
        entries = self._git('ls-tree','-rz','--full-tree',commit).split(b'\0')
        self.blobs = {}
        for entry in entries:
            if not entry: continue
            meta,name=entry.split(b'\t',1);mode,kind,sha=meta.decode().split()
            if mode in ('100644','100755') and kind == 'blob': self.blobs[name.decode()] = sha
        self._hashes = {}

    def _git(self,*args):
        return subprocess.run(['git','-c','core.fsmonitor=false','-C',self.repository,*args],capture_output=True,check=True,timeout=30,env={**os.environ,"GIT_NO_REPLACE_OBJECTS":"1"}).stdout

    def bytes(self,path):
        p=PurePosixPath(path)
        if p.is_absolute() or '..' in p.parts or '\\' in path or path not in self.blobs:
            raise ValueError('Evidence must be an exact regular-file path in the pinned Git tree')
        return self._git('cat-file','blob',self.blobs[path])

    def sha(self,path):
        if path not in self._hashes:self._hashes[path]=hashlib.sha256(self.bytes(path)).hexdigest()
        return self._hashes[path]

    def catalog(self): return [{'path':p,'file_sha256':self.sha(p)} for p in sorted(self.blobs if self.catalog_paths is None else self.catalog_paths) if p in self.blobs]

    def fingerprint(self,commit):
        if commit != self.commit: raise ValueError('Source commit mismatch')
        if self._git('rev-parse',commit+'^{tree}').decode().strip() != self.tree: raise ValueError('Source tree changed')
        return digest({'commit':commit,'tree':self.tree})
