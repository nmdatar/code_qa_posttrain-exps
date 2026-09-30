"""Read progress and efficiently archive the selected-data GRPO pilot."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path

import certifi
os.environ.setdefault('SSL_CERT_FILE', certifi.where())
import modal
from modal.exception import NotFoundError
from modal.volume import FileEntryType

from training_pipeline.storage import atomic_json, read

REPORT = Path('reports/bash-correctness-grpo-v1')
BASE = 'artifacts/experiments/bash-correctness-grpo-v1'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=['poll','archive'])
    args = parser.parse_args()
    receipt = read('artifacts/bash-correctness-grpo-v1-bundle.submission.json')
    sb = modal.Sandbox.from_id(receipt['sandbox_id'])
    try: code = sb.poll()
    finally: sb.detach()
    volume = modal.Volume.from_name(receipt['volume'])
    def get(path): return b''.join(volume.read_file(path))
    if args.operation == 'poll':
        events = []
        try:
            for line in get(BASE+'/events.jsonl').splitlines():
                try: events.append(json.loads(line))
                except ValueError: pass
        except (NotFoundError, ValueError): pass
        evaluations = [e for e in events if e.get('event') == 'evaluation']
        batches = [e for e in events if e.get('event') == 'training_batch']
        updates = [e for e in events if e.get('event') == 'update']
        result = {'controller_exit':code, 'last_event':events[-1].get('event') if events else None,
                  'trajectories':dict(Counter(e.get('phase') for e in events if e.get('event') == 'trajectory')),
                  'attempted_batches':len(batches), 'optimizer_updates':len(updates),
                  'evaluations':[{'step':e['optimizer_step'],'resolved':e['resolved'],'expected':e['expected'],
                                  'mean_reward':e['mean_reward']} for e in evaluations],
                  'tracking_failures':sum(e.get('event')=='tracking_failure' for e in events)}
        try:
            ledger = json.loads(get('artifacts/project-budget/bash-correctness-grpo-v1/run.json'))
            result['reserved_usd'] = ledger['reserved_usd']
        except (NotFoundError, ValueError):
            result['ledger_snapshot']='temporarily_unreadable_while_running'
        atomic_json(REPORT/'live-status.json',result)
        print(json.dumps(result,indent=2));return
    if code is None: raise RuntimeError('Wait for controller completion before final archive')
    paths=[]
    for prefix in (BASE, 'artifacts/project-budget/bash-correctness-grpo-v1', 'campaigns/'+receipt['bundle_id']):
        for entry in volume.listdir(prefix, recursive=True):
            name=entry.path.lstrip('/')
            if entry.type == FileEntryType.FILE and '/source/' not in name and '/wandb/' not in name:
                paths.append(name)
    output=Path('artifacts/bash-correctness-grpo-v1-results')
    def fetch(path):
        raw=get(path);dest=output/path;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(raw)
        if path.endswith('.json'): json.loads(raw)
        return str(dest.resolve()),hashlib.sha256(raw).hexdigest()
    with ThreadPoolExecutor(max_workers=16) as pool: checksums=dict(pool.map(fetch,paths))
    atomic_json(REPORT/'archive-checksums.json',checksums)
    atomic_json(REPORT/'archive-status.json',{'controller_exit':code,'files':len(checksums)})
    print(json.dumps({'controller_exit':code,'files':len(checksums),'output':str(output)}))


if __name__=='__main__':main()
