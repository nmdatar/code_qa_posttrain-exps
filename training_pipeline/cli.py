"""qa-train: validated training, immutable checkpoint operations and toy smoke."""
import argparse
import json
from pathlib import Path
import sys
from .config import inputs
from .storage import read, load_checkpoint
from .orchestrator import Pipeline, make_backend, evaluate_checkpoint


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    subs = p.add_subparsers(dest='command', required=True)
    comparison = subs.add_parser('compare', help='Compare matched evaluation reports (offline)')
    comparison.add_argument('--base', type=Path, required=True)
    comparison.add_argument('--candidates', type=Path, nargs='+', required=True)
    comparison.add_argument('--output', type=Path, required=True)
    cohorts = subs.add_parser('cohorts', help='Write deterministic development cohort identities (offline)')
    cohorts.add_argument('--config', type=Path, required=True)
    cohorts.add_argument('--output', type=Path, required=True)
    cohorts.add_argument('--selection-size', type=int, default=32)
    baseline = subs.add_parser('baseline', help='Evaluate unchanged base weights without training (paid)')
    baseline.add_argument('--config', type=Path, required=True)
    baseline.add_argument('--cohort', choices=['selection', 'confirmation'])
    export = subs.add_parser('export-rollouts', help='Automatically admit experimental collection rollout tasks')
    export.add_argument('--source', type=Path, required=True)
    export.add_argument('--output', type=Path, required=True)
    for name in ('validate', 'run', 'fork'):
        cmd = subs.add_parser(name)
        cmd.add_argument('--config', type=Path, required=True)
        if name == 'validate':
            cmd.add_argument('--remote', action='store_true')
        if name == 'fork':
            cmd.add_argument('--checkpoint', type=Path, required=True)
    resume = subs.add_parser('resume')
    resume.add_argument('--checkpoint', type=Path, required=True)
    resume.add_argument('--config', type=Path)
    evaluate = subs.add_parser('evaluate')
    evaluate.add_argument('--checkpoint', type=Path, required=True)
    evaluate.add_argument('--output', type=Path)
    evaluate.add_argument('--cohort', choices=['selection', 'confirmation'])
    smoke = subs.add_parser('smoke')
    smoke.add_argument('--output', type=Path, default=Path('artifacts/training-smoke'))
    a = p.parse_args(argv)
    if a.command == 'compare':
        from .comparison import compare
        from .storage import atomic_json
        result = compare(read(a.base), [read(path) for path in a.candidates])
        atomic_json(a.output, result)
        print(json.dumps(result, indent=2))
        return 0
    if a.command == 'cohorts':
        from .cohorts import build_cohorts
        from .storage import atomic_json
        if a.output.exists():
            p.error('Cohort output already exists; use a new versioned path')
        manifest = build_cohorts(inputs(read(a.config)), a.selection_size)
        atomic_json(a.output, manifest)
        print(json.dumps({'path': str(a.output.resolve()), 'cohort_sha256': manifest['manifest_hash']}))
        return 0
    if a.command == 'baseline':
        from .orchestrator import evaluate_base
        print(json.dumps(evaluate_base(read(a.config), cohort=a.cohort), indent=2))
        return 0
    if a.command == 'export-rollouts':
        from .admission import export_rollouts
        print(json.dumps(export_rollouts(a.source, a.output), indent=2))
        return 0
    if a.command == 'smoke':
        from .smoke import run_smoke
        path, result = run_smoke(a.output)
        print(json.dumps({'report': str(path), 'status': result['status']}))
        return 0 if result['status'] == 'passed' else 1
    if a.command == 'evaluate':
        print(json.dumps(evaluate_checkpoint(a.checkpoint, a.output, cohort=a.cohort), indent=2))
        return 0
    config = read(a.config) if a.config else load_checkpoint(a.checkpoint)['config']
    if a.command == 'resume' and config['run_id'] == 'auto':
        p.error('Resume needs the resolved checkpoint config; omit --config or use the saved run/config.json')
    data = inputs(config)
    if a.command == 'validate':
        result = {'status': 'valid', 'data_identity': data['identity'],
                  'counts': {k: len(data[k]) for k in ('sft', 'tasks', 'development')}}
        from .storage import run_label
        result['run_name'] = config['tracking'].get('run_name') or run_label(config)
        result['solver_model'] = config['model']['base_model']
        result['judge_model'] = config.get('judge', {}).get('base_model',
            config['model']['base_model'] if config['environment']['kind'] == 'collection' else None)
        if a.remote:
            from .tinker_backend import TinkerBackend
            backend = TinkerBackend(config['model'], config['limits'])
            try:
                result['model'] = backend.identity
            finally:
                backend.close()
        if a.remote and 'judge' in config:
            from .judge import TinkerJudge
            judge = TinkerJudge(config['judge'])
            try:
                result['judge'] = judge.identity
            finally:
                judge.close()
        print(json.dumps(result, indent=2))
        return 0
    pipeline = Pipeline(config, data=data)
    path = pipeline.run(checkpoint=getattr(a, 'checkpoint', None), purpose=a.command)
    print(json.dumps({'checkpoint': str(path)}))
    return 0


def entrypoint():
    try:
        return main()
    except Exception as exc:
        # No raw SDK errors (they may contain auth headers or URLs).
        print('Training stopped: ' + type(exc).__name__, file=sys.stderr)
        if isinstance(exc, ValueError):
            print(str(exc), file=sys.stderr)
        return 1
