"""Read progress or archive the approved full-pool difficulty campaign."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path

import certifi
os.environ.setdefault('SSL_CERT_FILE', certifi.where())
import modal

from training_pipeline.storage import atomic_json, read

REPORT = Path('reports/task-difficulty-full-v1')
RUN = 'task-difficulty-full-v1-remaining'
RUN_PATH = 'artifacts/experiments/' + RUN


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=['poll', 'prefetch', 'archive'])
    args = parser.parse_args()
    receipt = read('artifacts/task-difficulty-full-v1-bundle.submission.json')
    sb = modal.Sandbox.from_id(receipt['sandbox_id'])
    try:
        code = sb.poll()
        if args.operation == 'poll':
            result = {'exit_code': code, 'sandbox_id': receipt['sandbox_id']}
            if code is None:
                source = """
import json
from pathlib import Path
from collections import Counter
r=Path('/state/artifacts/experiments/task-difficulty-full-v1-remaining')
recorded=len(list((r/'trajectories').glob('*.json')))
events=[]
if (r/'events.jsonl').exists():
 for line in (r/'events.jsonl').read_text().splitlines():
  try: events.append(json.loads(line))
  except ValueError: pass
tracking=r/'tracking-url.json'
ledger=Path('/state/artifacts/project-budget/task-difficulty-full-v1/remaining.json')
v=[e for e in events if e.get('event')=='trajectory']
print(json.dumps({'recorded_traces':recorded,'graded_traces':len(v),'resolved':sum(x.get('reward') is not None for x in v),'tracking':json.loads(tracking.read_text()) if tracking.exists() else None,'tracking_failures':sum(e.get('event')=='tracking_failure' for e in events),'reserved_usd':json.loads(ledger.read_text())['reserved_usd'] if ledger.exists() else None,'benchmark_complete':(r/'benchmark.json').exists()}))
"""
                process = sb.exec('python', '-c', source, timeout=45)
                stdout = process.stdout.read()
                if process.wait():
                    raise RuntimeError('Progress query failed')
                result.update(json.loads(stdout))
            atomic_json(REPORT / 'live-status.json', result)
            print(json.dumps(result, indent=2))
            return
        if code not in (None, 0) and args.operation != 'prefetch':
            raise RuntimeError('Controller failed; inspect before archiving: ' + str(code))
    finally:
        sb.detach()
    volume = modal.Volume.from_name(receipt['volume'])
    output = Path('artifacts/task-difficulty-full-v1-results')
    benchmark = json.loads(b''.join(volume.read_file(RUN_PATH + '/benchmark.json')))
    if benchmark['attempted_episodes'] != 516 or benchmark['optimizer_updates'] != 0:
        raise RuntimeError('Archive requires all planned sampling-only rollouts finished')
    names = ['config.json', 'benchmark.json', 'tracking-url.json', 'events.jsonl', 'answers.json']
    if args.operation == 'prefetch':
        names = ['config.json', 'benchmark.json']
    paths = [RUN_PATH + '/' + name for name in names]
    paths += [e.path.lstrip('/') for e in volume.listdir(RUN_PATH + '/trajectories') if e.path.endswith('.json')]
    if args.operation == 'archive':
        paths += ['artifacts/project-budget/task-difficulty-full-v1/remaining.json',
                  'campaigns/' + receipt['bundle_id'] + '/status.json']
    cached = read(REPORT / 'prefetch-checksums.json') if (REPORT / 'prefetch-checksums.json').exists() else {}

    def fetch(path):
        target = output / path
        key = str(target.resolve())
        if '/trajectories/' in path and key in cached and target.exists():
            if hashlib.sha256(target.read_bytes()).hexdigest() == cached[key]:
                return key, cached[key]
        raw = b''.join(volume.read_file(path))
        if path.endswith('.json'):
            json.loads(raw)
        else:
            for line in raw.splitlines():
                json.loads(line)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
        return str(target.resolve()), hashlib.sha256(raw).hexdigest()

    with ThreadPoolExecutor(max_workers=16) as pool:
        checksums = dict(pool.map(fetch, paths))
    if args.operation == 'archive':
        records = [read(p) for p in (output/RUN_PATH/'trajectories').glob('*.json')]
        if len(records) != 516 or any(not isinstance(t.get('verification'), dict) for t in records):
            raise RuntimeError('Incomplete final grading records')
        if sum(t['verification']['status'] == 'resolved' for t in records) != benchmark['resolved_episodes']:
            raise RuntimeError('Archived grading differs from finished benchmark')
        events = [json.loads(line) for line in (output/RUN_PATH/'events.jsonl').read_text().splitlines()]
        if {e['episode_id'] for e in events if e.get('event') == 'trajectory'} != {t['episode_id'] for t in records}:
            raise RuntimeError('Archived trace events do not cover finished episodes')
        atomic_json(REPORT/'archive-status.json', {'controller_exit': code, 'rollouts_complete': True,
            'verified_trajectories': len(records), 'tracking_finalization_pending': code is None,
            'tracking_failures': sum(e.get('event') == 'tracking_failure' for e in events)})
    atomic_json(REPORT / ('archive-checksums.json' if args.operation == 'archive' else 'prefetch-checksums.json'), checksums)
    print(json.dumps({'files': len(checksums), 'root': str(output), 'exit_code': code}))


if __name__ == '__main__':
    main()
