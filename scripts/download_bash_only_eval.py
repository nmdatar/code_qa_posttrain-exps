"""Retrieve the approved completed campaign, atomically, with bounded concurrency."""
import os
import json
import hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import certifi
os.environ.setdefault('SSL_CERT_FILE',certifi.where())
import modal
from modal.volume import FileEntryType
r=json.loads(Path('artifacts/bash-only-eval-v1-funded-bundle.submission.json').read_text())
volume=modal.Volume.from_name(r['volume'])
out=Path('artifacts/bash-only-eval-v1-results')
files=[]
for prefix in ('campaigns/'+r['bundle_id'],'artifacts'):
 files.extend(e.path for e in volume.listdir(prefix,recursive=True) if e.type==FileEntryType.FILE)
def fetch(path):
 rel=Path(path.lstrip('/'))
 if '..' in rel.parts:raise ValueError('Unsafe remote path')
 raw=b''.join(volume.read_file(path))
 if rel.suffix=='.json':json.loads(raw)
 target=out/rel;target.parent.mkdir(parents=True,exist_ok=True)
 temp=target.with_name(target.name+'.download');temp.write_bytes(raw);temp.replace(target)
 return {'path':str(rel),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
with ThreadPoolExecutor(max_workers=12) as pool:
 records=list(pool.map(fetch,files))
Path('reports/bash-only-eval-v1/archive-checksums.json').write_text(json.dumps(records,indent=2)+'\n')
print(json.dumps({'files':len(records),'bytes':sum(x['bytes'] for x in records)}))
