"""Read-only remote progress; cache only finalized, readable trajectory summaries."""
import concurrent.futures,json,datetime
from pathlib import Path
import os,certifi
os.environ.setdefault('SSL_CERT_FILE',certifi.where())
import modal
ROOT=Path('reports/expanded-studies'); CACHE=ROOT/'teacher-progress-cache.json'
v=modal.Volume.from_name('repository-qa-training-state-v2')
run='artifacts/experiments/expanded-teacher-collection-v1-seed42'
def get(path):
 try:return json.loads(b''.join(v.read_file(path)))
 except (ValueError,UnicodeError):return None
cache=json.loads(CACHE.read_text()) if CACHE.exists() else {}
files=v.listdir(run+'/trajectories')
pending=[e.path for e in files if e.path not in cache]
def summarize(path):
 r=get(path)
 if not r or not isinstance(r.get('verification'),dict):return path,None
 q=r['verification'];d=q.get('diagnostics') or {}
 return path,{'status':q.get('status'),'reward':q.get('reward'),'strict_score':d.get('strict_score'),'termination':r.get('termination')}
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
 for path,row in pool.map(summarize,pending):
  if row is not None:cache[path]=row
 ledgers={k:pool.submit(get,'artifacts/project-budget/expanded-studies-v1/'+k+'.json') for k in ['teacher','student']}
 ledgers={k:f.result() for k,f in ledgers.items()}
if any(x is None for x in ledgers.values()):
 # Read an atomic file inside the existing controller if the volume snapshot
 # overlaps a rewrite. Only fixed ledger fields are emitted; no run mutation.
 sb=modal.Sandbox.from_id('sb-OuAPccGFF0sGuxUGm4ty96')
 try:
  for key,x in ledgers.items():
   if x is None:
    command="import json; x=json.load(open('/state/artifacts/project-budget/expanded-studies-v1/"+key+".json')); print(json.dumps({k:x[k] for k in ['cap','reserved_usd']}))"
    proc=sb.exec('python','-c',command,timeout=30)
    output=proc.stdout.read()
    if proc.wait()==0:ledgers[key]=json.loads(output)
 except Exception:
  pass
 finally:sb.detach()
CACHE.write_text(json.dumps(cache,indent=2))
rows=list(cache.values());resolved=[r for r in rows if r['status']=='resolved']
result={'observed_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'completed_readable':len(rows),'planned':1200,'resolved':len(resolved),'strict_passes':sum(r['strict_score']==1 for r in rows),'mean_training_reward':sum(r['reward'] or 0 for r in resolved)/len(resolved) if resolved else None,'ledgers':{k:{z:x.get(z) for z in ['cap','reserved_usd']} if x else {'snapshot':'temporarily_unreadable'} for k,x in ledgers.items()}}
(ROOT/'latest-teacher-progress.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
