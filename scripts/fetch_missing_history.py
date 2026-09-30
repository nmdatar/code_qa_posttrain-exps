"""Read-only snapshot of the missing run's trajectory records from its Modal volume."""
import json,concurrent.futures
from pathlib import Path
import modal
from modal.volume import FileEntryType
run='expanded-direct-seed43-v1'
volume=modal.Volume.from_name('qa-state-bbxsv6dcorr5n2rpar2qd242kq7mswr4q25pcg4mqhjvivf4oybq')
prefix='artifacts/experiments/'+run
out=Path('artifacts/trajectory-history-snapshots')/run
paths=[e.path for e in volume.listdir(prefix,recursive=True) if e.type==FileEntryType.FILE and ('/trajectories/' in e.path or Path(e.path).name=='answers.json' or Path(e.path).name.startswith('tracking'))]
def fetch(path):
 try:
  raw=b''.join(volume.read_file(path));json.loads(raw)
  target=out/Path(path).relative_to(prefix);target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
  return {'path':path,'status':'saved'}
 except Exception as e:return {'path':path,'status':type(e).__name__,'error':str(e)}
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:results=list(pool.map(fetch,paths))
Path('reports/trajectory-history/snapshot-download.json').write_text(json.dumps(results,indent=2))
sources=Path('reports/trajectory-history/sources.json');data=json.loads(sources.read_text())
if list((out/'trajectories').glob('*.json')):
 data.append({'root':str(out.resolve()),'run_id':run,'count':len(list((out/'trajectories').glob('*.json')))})
 sources.write_text(json.dumps(data,indent=2))
print('Saved',sum(x['status']=='saved' for x in results),'of',len(paths),'files')
