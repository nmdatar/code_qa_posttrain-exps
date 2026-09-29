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
    smoke = subs.add_parser('smoke')
    smoke.add_argument('--output', type=Path, default=Path('artifacts/training-smoke'))
    a = p.parse_args(argv)
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
        print(json.dumps(evaluate_checkpoint(a.checkpoint, a.output), indent=2))
        return 0
    config = read(a.config) if a.config else load_checkpoint(a.checkpoint)['config']
    data = inputs(config)
    if a.command == 'validate':
        result = {'status': 'valid', 'data_identity': data['identity'],
                  'counts': {k: len(data[k]) for k in ('sft', 'tasks', 'development')}}
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
