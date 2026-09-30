"""Fetch the source snapshots in a preassigned collection plan; no repo code runs."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import time
from scripts.prepare_dataset import checkout
from dataset_builder.build import write_json
from qa_eval.deterministic import snapshot
from dataset_builder.environment import _export_snapshot


def worker(row, root):
    started=time.monotonic()
    try:
        result=checkout({'repo':row['repo'],'commit_id':row['commit']},root/'artifacts/repos')
        path=Path(result['path']);tree=snapshot(path,row['commit'])
        files,links=_export_snapshot(path,row['commit'])
        return {**row,'status':'ready','snapshot_path':str(path),'tree_fingerprint':tree,
                'source_files':len(files),'materialized_links':len(links),'seconds':time.monotonic()-started}
    except Exception as exc:
        return {**row,'status':'quarantined','error':str(exc),'seconds':time.monotonic()-started}


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--workers',type=int,default=4);a=p.parse_args()
    root=Path.cwd();plan=json.loads(a.plan.read_text());out=root/'reports/task-generation-1000/acquisition';out.mkdir(parents=True,exist_ok=True)
    results=[]
    with ThreadPoolExecutor(max_workers=a.workers) as executor:
        futures={executor.submit(worker,row,root):row for row in plan['repositories']}
        for future in as_completed(futures):
            r=future.result();results.append(r);write_json(out/(r['repo'].replace('/','--')+'-'+r['commit'][:12]+'.json'),r)
            print(r['repo'],r['status'],round(r['seconds'],1),flush=True)
    write_json(out/'summary.json',results)


if __name__=='__main__':main()
