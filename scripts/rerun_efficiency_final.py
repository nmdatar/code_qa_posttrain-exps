"""Repeat final frozen-checkpoint evaluations; never train or overwrite evidence."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys

NAME = 'efficiency-rl-v2-final-reeval-v1'
CHECKPOINTS = {
    'efficiency': 'ckpt-c4d4e24f2a7548f18a453af7f7cce9a6',
    'quality-only': 'ckpt-95e8c9ba4f264902a74c6fdf36cc6a0a',
}


def worker():
    from training_pipeline.remote import verify_bundle, operation_estimate
    from training_pipeline.storage import atomic_json, read, load_checkpoint
    from training_pipeline.orchestrator import evaluate_checkpoint
    root = Path('/state/campaigns') / NAME
    root.mkdir(parents=True, exist_ok=False)
    state = {'status': 'starting', 'results': [], 'training': False}
    atomic_json(root / 'status.json', state)
    bundle = verify_bundle('/bundle')
    mapping = {}
    for source, name in bundle['snapshots'].items():
        target = Path('/tmp/reeval-snapshots') / Path(name).stem
        target.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(['git', 'clone', '--bare', '/bundle/snapshots/' + name, str(target)],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        mapping[source] = str(target)
    atomic_json(root / 'snapshot-map.json', mapping)
    os.environ['QA_SNAPSHOT_MAP'] = str(root / 'snapshot-map.json')
    try:
        for arm, ident in CHECKPOINTS.items():
            original = 'efficiency-rl-v2-' + arm + '-seed43'
            checkpoint = Path('/state/artifacts/experiments') / original / 'checkpoints' / (ident + '.json')
            manifest = load_checkpoint(checkpoint)
            c = copy.deepcopy(manifest['config'])
            c['run_id'] = NAME + '-' + arm
            c['output'] = '/state/artifacts/experiments/' + c['run_id']
            c['concurrency'] = {'rollouts': 2, 'judges': 2}
            c['execution']['operation'] = 'evaluate'
            c['execution']['checkpoint'] = str(checkpoint.relative_to('/state'))
            c['tracking'].update(run_name='Final evaluation retry: ' + arm,
                                 aggregate_only=True,
                                 notes='Full 32-question reevaluation after provisioning timeouts. Exact final saved weights and frozen science. Two rollouts at a time, arms sequential. Original failed evaluation retained; runtime differs in service conditions.')
            c['tracking']['tags'] += ['final-evaluation-retry', 'no-training']
            required = operation_estimate(c)['upper_estimate_usd']
            ledger = read(c['spend']['ledger'])
            if ledger['reserved_usd'] + required > c['spend']['cap_usd']:
                raise RuntimeError('Insufficient remaining original experiment allocation')
            atomic_json(root / (arm + '-config.json'), c)
            state.update(status='evaluating', active_arm=arm)
            atomic_json(root / 'status.json', state)
            report = evaluate_checkpoint(checkpoint, output=c['output'], cohort='selection', config=c)
            atomic_json(root / (arm + '-report.json'), report)
            state['results'].append({'arm': arm, 'optimizer_step': report['optimizer_step'],
                                     'completed': report['completion_rate'], 'resolved': report['resolved'],
                                     'infrastructure_errors': sum(r['termination'] == 'infrastructure_error' for r in report['results'])})
            atomic_json(root / 'status.json', state)
        state.update(status='complete', active_arm=None)
        atomic_json(root / 'status.json', state)
    except BaseException as exc:
        state.update(status='failed', error_type=type(exc).__name__)
        atomic_json(root / 'status.json', state)
        raise


def launch():
    import certifi
    os.environ.setdefault('SSL_CERT_FILE', certifi.where())
    import modal
    receipt = Path('reports/efficiency-rl-v2/final-reeval-submission.json')
    if receipt.exists():
        raise RuntimeError('Submission already attempted; inspect receipt instead of resubmitting')
    old = json.loads(Path('artifacts/efficiency-rl-v2-bundle.submission.json').read_text())
    original = modal.Sandbox.from_id(old['sandbox_id'])
    if original.poll() != 0:
        raise RuntimeError('Original campaign must have completed successfully')
    original.detach()
    image = (modal.Image.debian_slim(python_version='3.11').apt_install('git')
             .pip_install('modal==1.5.5','tinker==0.30.4','transformers==5.17.0','jinja2==3.1.6','wandb==0.30.0')
             .add_local_dir('artifacts/efficiency-rl-v2-bundle', '/bundle', copy=True)
             .add_local_file(__file__, '/rerun.py', copy=True)
             .env({'PYTHONPATH': '/bundle/code', 'QA_MODAL_WORKER': '1', 'PYTHONUNBUFFERED': '1', 'PYTHONDONTWRITEBYTECODE': '1'}))
    receipt.write_text(json.dumps({'status': 'submission_pending', 'volume': old['volume']}))
    s = modal.Sandbox.create('python', '/rerun.py', 'worker', image=image,
        app=modal.App.lookup('repository-qa-training', create_if_missing=True), name=NAME,
        timeout=3600, cpu=(2,2), memory=(4096,4096), workdir='/state',
        volumes={'/state': modal.Volume.from_name(old['volume'])},
        secrets=[modal.Secret.from_name('repository-qa-training-credentials', required_keys=['TINKER_API_KEY','MODAL_TOKEN_ID','MODAL_TOKEN_SECRET','WANDB_API_KEY'])])
    r = {'status': 'submitted', 'sandbox_id': s.object_id, 'volume': old['volume'], 'campaign': NAME}
    receipt.write_text(json.dumps(r, indent=2) + '\n')
    s.detach()
    print(json.dumps(r))


if __name__ == '__main__':
    worker() if sys.argv[1:] == ['worker'] else launch()
