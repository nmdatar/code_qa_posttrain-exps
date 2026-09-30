"""Preserve stopped phase1 only, including imperfect files and integrity audit."""
import concurrent.futures,hashlib,json,argparse
from pathlib import Path
import modal
from modal.volume import FileEntryType
parser=argparse.ArgumentParser();parser.add_argument('--receipt',default='artifacts/expanded-studies-phase1-bundle.submission.json');parser.add_argument('--run',default='expanded-teacher-collection-v1-seed42');parser.add_argument('--output',default='artifacts/expanded-studies-phase1-billing-stop-results');parser.add_argument('--report',default='reports/expanded-studies/billing-stop-archive.json');parser.add_argument('--ledger-prefix',default='artifacts/project-budget/expanded-studies-v1');args=parser.parse_args()
r=json.loads(Path(args.receipt).read_text())
s=modal.Sandbox.from_id(r['sandbox_id']);status=s.poll();s.detach()
assert status is not None,'Controller must finish stopping before archive'
v=modal.Volume.from_name(r['volume']);root=Path(args.output);root.mkdir(exist_ok=False)
prefixes=['control-campaign','artifacts/controls/aligned-live-controls-v1','artifacts/project-budget']
paths=[e.path for prefix in prefixes for e in v.listdir(prefix,recursive=True) if e.type==FileEntryType.FILE]
def get(p):
 target=root/p.lstrip('/');target.parent.mkdir(parents=True,exist_ok=True);raw=b''.join(v.read_file(p));target.write_bytes(raw);err=None
 try:
  if target.suffix=='.json':json.loads(raw)
  if target.suffix=='.jsonl':
   for line in raw.splitlines():
    if line.strip():json.loads(line)
 except (ValueError,UnicodeError) as e:err=type(e).__name__
 return {'path':p,'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw),'parse_error':err}
records=[]
with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
 for record in pool.map(get,paths):
  records.append(record)
  if len(records)%500==0:print('archived',len(records),'of',len(paths),flush=True)
report={'controller_exit':status,'root':str(root.resolve()),'files':records,'parse_errors':sum(bool(x['parse_error']) for x in records)}
Path(args.report).write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='files'}),flush=True)
