"""Private subprocess entrypoint: puts the synchronous loop on the main thread."""
import argparse
import json
import os
from pathlib import Path
from .remote_contracts import atomic_json, identity
from .remote_workers import run_grade, run_rollout


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('kind', choices=['rollout', 'grade'])
    parser.add_argument('--job', required=True)
    parser.add_argument('--result', required=True)
    args = parser.parse_args()
    job = json.loads(Path(args.job).read_text())
    try:
        telemetry_key = bytes.fromhex(os.environ['HARNESS_TELEMETRY_KEY'])
        if args.kind == 'rollout':
            result = run_rollout(job, public_root='/public', rollout_root='/rollouts', signing_key=telemetry_key)
        else:
            result = run_grade(job, public_root='/public', private_root='/private', rollout_root='/rollouts',
                               grade_root='/grades', rollout_key=telemetry_key,
                               grader_key=bytes.fromhex(os.environ['HARNESS_GRADER_KEY']))
    except Exception as exc:
        result = {**identity(job), 'status': 'unresolved', 'error_type': type(exc).__name__,
                  'reason': 'worker_failed'}
    atomic_json(args.result, result)


if __name__ == '__main__':
    main()
