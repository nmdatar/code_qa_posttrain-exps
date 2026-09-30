"""Append timestamped collection counts derived from actual on-disk evidence."""
import argparse
import datetime as dt
import json
from pathlib import Path

START=dt.datetime.fromtimestamp(1790633152,dt.timezone.utc)


def record(root, note):
    root=Path(root);folder=root/'reports/task-generation-1000';folder.mkdir(parents=True,exist_ok=True)
    acquired=[]
    for p in (folder/'acquisition').glob('*.json'):
        if p.name!='summary.json':acquired.append(json.loads(p.read_text()))
    prep_path=folder/'preparation.json';prep=json.loads(prep_path.read_text()) if prep_path.exists() else {}
    runtime=[]
    for p in (root/'data/generated/collection-1000-v1').glob('*/private/source-runtime-check.json'):
        r=json.loads(p.read_text());runtime.append((p.parents[1],r))
    ready=0;env_ready=0
    for bundle,r in runtime:
        if r.get('status')=='passed':
            ready+=json.loads((bundle/'manifest.json').read_text())['task_count'];env_ready+=1
    reviews={}
    for p in [*folder.glob('semantic-review-[0-9]*.json'),*(folder/'full-review').glob('batch-*.json')]:
        for row in json.loads(p.read_text())['tasks']:reviews[row['task_id']]=row
    corrections={}
    for p in (folder/'correction-review').glob('worker-*/*.json'):
        for row in json.loads(p.read_text())['tasks']:corrections[row['task_id']]=row
    from collections import Counter
    now=dt.datetime.now(dt.timezone.utc)
    event={'timestamp_utc':now.isoformat(),'elapsed_seconds':round((now-START).total_seconds(),1),
           'existing_runnable_tasks':12,'downloaded_benchmark_candidates':980,
           'prepared_imported_tasks':prep.get('prepared_tasks',0),'snapshot_ready':sum(r['status']=='ready' for r in acquired),
           'snapshot_total':len(acquired),'runtime_checked_imported_tasks':ready,'runtime_checked_imported_environments':env_ready,
           'runnable_total_including_existing':12+ready,'source_reviewed_imported_tasks':len(reviews),
           'review_verdict_counts':dict(Counter(r['status'] for r in reviews.values())),
           'proposed_reference_corrections':sum(bool(r.get('corrected_reference_answer')) for r in reviews.values()),
           'corrections_independently_reviewed':len(corrections),'correction_review_verdict_counts':dict(Counter(r['status'] for r in corrections.values())),
           'note':note,
           'counting_rule':'Runtime checked is not independent semantic review or human approval; preparation and imports are separate.'}
    path=folder/'progress.jsonl'
    with path.open('a') as f:f.write(json.dumps(event)+'\n')
    rows=[json.loads(l) for l in path.read_text().splitlines() if l]
    lines=['# Collection progress: approximately 1,000 tasks','',
           'Elapsed time starts at the recorded start of this collection goal (2026-09-28). Imported candidates, runnable tasks, semantic review and human approval are distinct. Source-reading readiness does not establish answer correctness.','',
           '| UTC | Elapsed minutes | Downloaded candidates | Prepared imports | Runtime-checked imports | Runnable total (includes original 12) | Ready snapshots | Source-reviewed imports | Corrections reviewed | Update |',
           '|---|---:|---:|---:|---:|---:|---:|---:|---:|---|']
    for x in rows:
        lines.append(f"| {x['timestamp_utc'][:19]} | {x['elapsed_seconds']/60:.1f} | {x['downloaded_benchmark_candidates']} | {x['prepared_imported_tasks']} | {x['runtime_checked_imported_tasks']} | {x['runnable_total_including_existing']} | {x['snapshot_ready']}/{x['snapshot_total']} | {x.get('source_reviewed_imported_tasks','—')} | {x.get('corrections_independently_reviewed',0)} | {x['note'].replace('|','/')} |")
    (root/'requirements/TASK_COLLECTION_PROGRESS.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(event))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--note',required=True);p.add_argument('--root',type=Path,default=Path.cwd());a=p.parse_args();record(a.root,a.note)
