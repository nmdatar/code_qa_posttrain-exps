"""Import public GitHub source without checkout hooks or repository execution."""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time


def import_repository(url, ref, root):
    match = re.fullmatch(r'https://github\.com/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+?)(?:\.git)?/?', url.strip())
    if not match or any(p in ('.', '..') for p in match.groups()):
        raise ValueError('Use a public https://github.com/owner/repository URL')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_./-]{0,199}', ref) or '..' in ref or ref.endswith('/'):
        raise ValueError('Enter a branch, tag, or commit (default: HEAD)')
    name = '/'.join(match.groups())
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, 'GIT_TERMINAL_PROMPT':'0', 'GIT_CONFIG_NOSYSTEM':'1', 'GIT_CONFIG_GLOBAL':os.devnull}
    with tempfile.TemporaryDirectory(dir=root) as temp:
        deadline = time.monotonic() + 120
        def git(*args):
            try:
                remaining = deadline - time.monotonic()
                if remaining <= 0: raise ValueError('Repository import exceeded the two-minute limit')
                return subprocess.run(['git','--no-replace-objects','-C',temp,*args],env=env,
                    stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=True,timeout=min(90,remaining)).stdout
            except (subprocess.CalledProcessError,subprocess.TimeoutExpired):
                raise ValueError('Could not import repository/ref. Check that it is public and the ref exists.') from None
        git('init','--bare','-q')
        git('-c','http.followRedirects=false','fetch','--depth=1','--no-tags','https://github.com/'+name+'.git',ref)
        commit=git('rev-parse','FETCH_HEAD').decode().strip()
        identity='github-'+hashlib.sha256((name+commit).encode()).hexdigest()[:24]
        destination=root/identity
        if (destination/'repository.json').exists():
            return json.loads((destination/'repository.json').read_text())
        files={}; omitted=0; total_bytes=0
        for entry in git('ls-tree','-rlz','--full-tree',commit).split(b'\0'):
            if not entry: continue
            meta,path=entry.split(b'\t',1); mode,kind,oid,size=meta.decode().split()
            if mode not in ('100644','100755') or kind!='blob' or int(size)>2_000_000:
                omitted+=1; continue
            if len(files)>=10000: raise ValueError('Repository exceeds the 10,000 source-file import limit')
            total_bytes += int(size)
            if total_bytes > 100_000_000: raise ValueError('Repository exceeds the 100 MB source import limit')
            files[path.decode()]=hashlib.sha256(git('cat-file','blob',oid)).hexdigest()
        config=dict(id=identity,name=name,commit=commit,execution=False,ready=True,reason=None,
            example=f'How is {name} organized?',source_path=str(destination),
            manifest_path=str(destination/'manifest.json'),imported=True,omitted_files=omitted)
        Path(temp,'manifest.json').write_text(json.dumps({'commit':commit,'snapshot_files':files}))
        Path(temp,'repository.json').write_text(json.dumps(config))
        Path(temp).rename(destination)
        return config
