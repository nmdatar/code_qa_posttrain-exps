"""Synthetic-only remote harness check; no policy or judge API calls.

Run from the repository: python -m scripts.smoke_remote_harness --deploy --output /tmp/harness-smoke
Modal CPU/storage usage applies. Existing smoke results remain on Volumes.
"""
import argparse
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys
import time
import uuid

from qa_eval.demo import fixture
from qa_eval.dataset import freeze
from qa_eval.security import bindings, digest
from agent_harness.remote_batch import prepare_batch
from agent_harness.remote_cli import submit_bundle, _journal


def prepare(root):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=False)
    task, answer, experiment, _, semantic = fixture(root / 'repository')
    task['permitted_tools'] = ['read_file', 'search_code']
    experiment['frozen'] = False
    experiment = freeze(experiment, [task], {'synthetic': True})
    semantic.update(**bindings(task, answer), experiment_hash=digest(experiment))
    for name, value in [('task', task), ('experiment', experiment), ('semantic', semantic)]:
        (root / (name + '.json')).write_text(json.dumps(value))
    config = {'run_id': 'smoke-' + uuid.uuid4().hex, 'synthetic': True,
              'model': {'kind': 'scripted', 'actions': [
                  {'type': 'tool_call', 'name': 'read_file',
                   'arguments': {'path': 'executor.py', 'start_line': 1, 'end_line': 2}},
                  {'type': 'final_answer', 'value': answer}]},
              'tasks': [{'task': str(root / 'task.json'), 'experiment': str(root / 'experiment.json'),
                         'semantic': str(root / 'semantic.json'), 'repository': str(root / 'repository')}],
              'episodes_per_task': 4, 'max_rollouts': 2, 'max_graders': 2, 'coordinator_seconds': 600}
    (root / 'config.json').write_text(json.dumps(config, indent=2))
    return prepare_batch(config, root / 'bundle')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--app', default='agent-harness-smoke')
    parser.add_argument('--deploy', action='store_true')
    args = parser.parse_args()
    if not args.app.endswith('-smoke'):
        parser.error('Smoke app name must end with -smoke')
    root = Path(args.output).resolve()
    manifest = prepare(root)
    if args.deploy:
        env = {**os.environ, 'HARNESS_APP': args.app, 'HARNESS_SMOKE': '1', 'HARNESS_INCLUDE_TINKER': '0',
               'HARNESS_TELEMETRY_KEY': secrets.token_hex(32), 'HARNESS_GRADER_KEY': secrets.token_hex(32)}
        subprocess.run([sys.executable, '-m', 'modal', 'deploy', '-m', 'agent_harness.modal_app'], env=env, check=True)
    submission = submit_bundle(root / 'bundle', args.app)
    (root / 'submitted.json').write_text(json.dumps(submission, indent=2))
    print(json.dumps(submission), flush=True)
    import modal
    call = modal.FunctionCall.from_id(submission['call_id'])
    deadline = time.monotonic() + 720
    while time.monotonic() < deadline:
        try:
            result = call.get(timeout=30)
            break
        except TimeoutError:
            print('Remote synthetic cohort is running.', flush=True)
    else:
        raise TimeoutError('Smoke coordinator still pending; inspect the saved call ID')
    (root / 'result.json').write_text(json.dumps(result, indent=2))
    if result['completed_episodes'] != 4 or result['unresolved_episodes'] != 0:
        raise RuntimeError('Synthetic cohort did not fully resolve; inspect the durable journal')
    before = _journal(args.app, manifest['run_id'])
    repeated = modal.Function.from_name(args.app, 'coordinate_run').remote(manifest['run_id'])
    after = _journal(args.app, manifest['run_id'])
    if repeated != result or before['episodes'] != after['episodes']:
        raise RuntimeError('Completed resume unexpectedly changed the cohort')
    print(json.dumps({'verification': 'passed', 'synthetic': True, 'real_model_run': False,
                      'completed_resume_reused_calls': True, **result}, indent=2))


if __name__ == '__main__':
    main()
