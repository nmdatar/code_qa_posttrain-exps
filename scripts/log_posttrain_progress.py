"""Refresh honest implementation/data progress without marking pending gates ready."""
import argparse,datetime,json,time
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from posttrain.storage import append,read,atomic

def main():
 p=argparse.ArgumentParser();p.add_argument('--milestone',required=True);a=p.parse_args()
 root=Path('reports/posttrain');root.mkdir(exist_ok=True,parents=True)
 start=root/'progress-start.json'
 if not start.exists():atomic(start,{'started_at':Path('posttrain/config.py').stat().st_birthtime if hasattr(Path('posttrain/config.py').stat(),'st_birthtime') else time.time(),'basis':'implementation file creation time; excludes prior planning'})
 candidates=[json.loads(l) for l in (root/'fresh-task-candidates.jsonl').read_text().splitlines() if l.strip()] if (root/'fresh-task-candidates.jsonl').exists() else []
 review=read(root/'fresh-task-independent-review.json') if (root/'fresh-task-independent-review.json').exists() else {}
 entry={'time_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'elapsed_minutes':round((time.time()-read(start)['started_at'])/60,2),
        'milestone':a.milestone,'fresh_candidates':len(candidates),'fresh_admitted_tasks':0,
        'source_reviewed':review.get('counts',{}).get('supported',0),
        'runtime_ready_tasks':10*sum(json.loads(p.read_text()).get('status')=='passed' for p in Path('data/prepared/posttrain-fresh-v1').glob('*/private/source-runtime-check.json')),
        'budget':read('artifacts/posttrain/spending.json') if Path('artifacts/posttrain/spending.json').exists() else None}
 append(root/'progress.jsonl',entry)
 rows=[json.loads(l) for l in (root/'progress.jsonl').read_text().splitlines()]
 text='# Post-training implementation progress\n\nCounts distinguish candidates, independently source-reviewed tasks, runnable tasks, and human-admitted training records. Elapsed time starts at implementation-file creation, excluding planning.\n\n| UTC | Minutes | Candidates | Human admitted | Milestone |\n|---|---:|---:|---:|---|\n'
 text+=''.join('| '+r['time_utc']+' | '+str(r['elapsed_minutes'])+' | '+str(r['fresh_candidates'])+' | '+str(r['fresh_admitted_tasks'])+' | '+r['milestone'].replace('|','/')+' |\n' for r in rows)
 Path('requirements/POSTTRAIN_PROGRESS.md').write_text(text)
 print(json.dumps({k:v for k,v in entry.items() if k!='budget'}))
if __name__=='__main__':main()
