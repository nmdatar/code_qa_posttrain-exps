import argparse,json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import modal
p=argparse.ArgumentParser();p.add_argument('bundle');p.add_argument('--all',action='store_true');a=p.parse_args()
m=json.loads((Path(a.bundle)/'bundle.json').read_text());v=modal.Volume.from_name('repository-qa-training-state-v2');out=Path('artifacts/reward-shaping-v2-results')
entries=[]
for prefix in ['campaigns/'+m['bundle_id']]+[c['output'].removeprefix('/state/') for c in m['configs']]:
 try:entries.extend(e for e in v.listdir(prefix,recursive=True) if e.type.name=='FILE' and '/wandb/' not in e.path)
 except modal.exception.NotFoundError:pass
if not a.all:entries=[e for e in entries if '/trajectories/' not in e.path and '/private/' not in e.path]
def fetch(e):
 target=out/e.path.lstrip('/');target.parent.mkdir(parents=True,exist_ok=True)
 raw=b''.join(v.read_file(e.path));target.write_bytes(raw)
 return str(target)
with ThreadPoolExecutor(max_workers=12) as pool:
 for _ in pool.map(fetch,entries):pass
print('Downloaded',len(entries),'files to',out)
for c in m['configs']:
 for f in (out/c['output'].removeprefix('/state/')/'evaluations').glob('*.json'):
  d=json.loads(f.read_text());print(c['run_id'],{k:d[k] for k in ['expected','resolved','completion_rate','scoring_coverage','demonstrated_quality','wall_seconds']});print('unresolved',[(r['task_id'],r['episode_id']) for r in d['results'] if r['status']!='resolved'])
