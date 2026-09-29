"""Deploy with `python -m modal deploy -m agent_harness.modal_app`.

Only trusted harness code executes in Function workers. Repository commands
execute in credential-free Modal Sandboxes through the generic sandbox adapter.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import modal
from .remote_contracts import identifier, safe_join, validate_job

NAME = os.environ.get('HARNESS_APP', 'agent-harness')
identifier(NAME)
SMOKE = os.environ.get('HARNESS_SMOKE') == '1'
WITH_TINKER = os.environ.get('HARNESS_INCLUDE_TINKER') == '1'
app = modal.App(NAME)
public = modal.Volume.from_name(NAME + '-public', create_if_missing=True)
private = modal.Volume.from_name(NAME + '-private', create_if_missing=True)
rollouts = modal.Volume.from_name(NAME + '-rollouts', create_if_missing=True)
grades = modal.Volume.from_name(NAME + '-grades', create_if_missing=True)
state = modal.Volume.from_name(NAME + '-state', create_if_missing=True)

# Smoke keys are short-lived deployment secrets supplied by the local operator.
# They never enter the public job or the image filesystem.
if SMOKE:
    if modal.is_local():
        for key_name in ('HARNESS_TELEMETRY_KEY', 'HARNESS_GRADER_KEY'):
            if len(bytes.fromhex(os.environ.get(key_name, ''))) < 32:
                raise ValueError('Smoke signing keys must be supplied as hex with at least 32 bytes')
    telemetry_secret = modal.Secret.from_dict({'HARNESS_TELEMETRY_KEY': os.environ.get('HARNESS_TELEMETRY_KEY', 'remote-placeholder')})
    grader_secret = modal.Secret.from_dict({'HARNESS_GRADER_KEY': os.environ.get('HARNESS_GRADER_KEY', 'remote-placeholder')})
    policy_secrets = []
else:
    telemetry_secret = modal.Secret.from_name(NAME + '-telemetry', required_keys=['HARNESS_TELEMETRY_KEY'])
    grader_secret = modal.Secret.from_name(NAME + '-grader', required_keys=['HARNESS_GRADER_KEY'])
    policy_secrets = [modal.Secret.from_name(NAME + '-policy')]

base_image = (modal.Image.debian_slim(python_version='3.12').apt_install('git')
              .env({'PYTHONPATH': '/opt/harness', 'HARNESS_APP': NAME,
                    'HARNESS_SMOKE': '1' if SMOKE else '0',
                    'HARNESS_INCLUDE_TINKER': '1' if WITH_TINKER else '0'}))
if WITH_TINKER:
    base_image = base_image.pip_install('tinker==0.30.4', 'tinker-cookbook==0.5.7')
for package in ('agent_harness', 'qa_eval', 'dataset_builder'):
    base_image = base_image.add_local_dir(Path(__file__).resolve().parents[1] / package,
                                          '/opt/harness/' + package, ignore=['__pycache__', '*.pyc'])


def _check(job):
    validate_job(job)
    if SMOKE and job['model']['kind'] != 'scripted':
        raise ValueError('Smoke deployments accept only scripted jobs')
    if job['model']['kind'] == 'tinker' and not WITH_TINKER:
        raise ValueError('Deploy with HARNESS_INCLUDE_TINKER=1 for Tinker')


def _worker(kind, job):
    _check(job)
    with tempfile.TemporaryDirectory(prefix='harness-worker-') as tmp:
        source, target = Path(tmp) / 'job.json', Path(tmp) / 'result.json'
        source.write_text(json.dumps(job, allow_nan=False))
        # Modal Functions run on a service thread; POSIX deadline handling and
        # the full agent loop belong on the subprocess main thread.
        completed = subprocess.run([sys.executable, '-m', 'agent_harness.remote_worker_cli', kind,
            '--job', str(source), '--result', str(target)], timeout=5300,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if completed.returncode:
            raise RuntimeError('Harness ' + kind + ' worker failed; inspect durable artifacts')
        return json.loads(target.read_text())


@app.function(image=base_image, cpu=1, memory=2048, timeout=5400, retries=0,
              max_containers=32, secrets=[telemetry_secret, *policy_secrets],
              volumes={'/public': public.with_mount_options(read_only=True), '/rollouts': rollouts})
def rollout_episode(job):
    public.reload(); rollouts.reload()
    try:
        return _worker('rollout', job)
    finally:
        rollouts.commit()


@app.function(image=base_image, cpu=1, memory=2048, timeout=5400, retries=0,
              max_containers=16, secrets=[telemetry_secret, grader_secret],
              volumes={'/public': public.with_mount_options(read_only=True),
                       '/private': private.with_mount_options(read_only=True),
                       '/rollouts': rollouts.with_mount_options(read_only=True), '/grades': grades})
def grade_episode(job):
    public.reload(); private.reload(); rollouts.reload(); grades.reload()
    try:
        return _worker('grade', job)
    finally:
        grades.commit()


class ModalExecutor:
    def submit(self, kind, job):
        worker = rollout_episode if kind == 'rollout' else grade_episode
        return worker.spawn(job).object_id

    def cancel(self, call_id):
        modal.FunctionCall.from_id(call_id).cancel(terminate_containers=True)

    def poll(self, call_id):
        try:
            return modal.FunctionCall.from_id(call_id).get(timeout=0)
        except TimeoutError:
            return None


# One input/container and one coordinator container enforce the journal's
# single-writer rule, including across duplicate submissions of the same run.
@app.function(image=base_image, cpu=0.25, memory=1024, timeout=86400, retries=0,
              max_containers=1, volumes={'/public': public.with_mount_options(read_only=True), '/state': state})
def coordinate_run(run_id):
    from .coordinator import coordinate
    identifier(run_id)
    public.reload(); state.reload()
    manifest = json.loads(safe_join('/public', run_id + '/manifest.json').read_text())
    if manifest['run_id'] != run_id or not 0 < manifest['max_rollouts'] <= 32 or not 0 < manifest['max_graders'] <= 16:
        raise ValueError('Manifest identity or deployment limits mismatch')
    if not 0 < manifest['coordinator_seconds'] <= 86000:
        raise ValueError('Coordinator deadline exceeds deployment limit')
    if SMOKE and manifest.get('synthetic') is not True:
        raise ValueError('Smoke deployment requires an explicitly synthetic batch')
    if not 0 < len(manifest['jobs']) <= 1000:
        raise ValueError('Initial deployment supports up to 1000 episodes per cohort')
    for job in manifest['jobs']:
        _check(job)
    summary = coordinate(manifest, ModalExecutor(), safe_join('/state', run_id + '/journal.json'),
                         persist=state.commit, poll_interval=1)
    # Full journal remains in the Volume; large cohorts do not become RPC results.
    return {key: summary[key] for key in ('run_id', 'status', 'expected_episodes',
                                        'completed_episodes', 'unresolved_episodes', 'groups')}
