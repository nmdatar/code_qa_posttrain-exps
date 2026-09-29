"""Prepare, submit, and retrieve detached Modal research cohorts."""
import argparse
import json
from pathlib import Path

from .remote_batch import prepare_batch
from .remote_contracts import identifier, job_hash, safe_join


def _volume(app, suffix):
    import modal
    identifier(app)
    return modal.Volume.from_name(app + '-' + suffix)


def submit_bundle(bundle, app):
    import modal
    bundle = Path(bundle).resolve()
    metadata = json.loads((bundle / 'bundle.json').read_text())
    run_id = identifier(metadata['run_id'])
    manifest = json.loads((bundle / 'public' / run_id / 'manifest.json').read_text())
    if manifest['run_id'] != run_id or job_hash(manifest) != metadata['manifest_hash']:
        raise ValueError('Prepared manifest changed')
    # Upload private data first, then public inputs. Launch only after both
    # uploads succeed; force=False prevents silent replacement of a run.
    for suffix in ('private', 'public'):
        volume = _volume(app, suffix)
        with volume.batch_upload(force=False) as upload:
            upload.put_directory(bundle / suffix / run_id, '/' + run_id)
    call = modal.Function.from_name(app, 'coordinate_run').spawn(run_id)
    return {'run_id': run_id, 'call_id': call.object_id, 'app': app,
            'status': 'submitted', 'manifest_hash': metadata['manifest_hash']}


def _journal(app, run_id):
    identifier(run_id)
    return json.loads(b''.join(_volume(app, 'state').read_file(run_id + '/journal.json')))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('prepare')
    p.add_argument('--config', required=True); p.add_argument('--output', required=True)
    p = commands.add_parser('submit')
    p.add_argument('--bundle', required=True); p.add_argument('--app', default='agent-harness')
    p = commands.add_parser('resume', help='Resume saved call handles; never blindly retry uncertain submissions')
    p.add_argument('--run-id', required=True); p.add_argument('--app', default='agent-harness')
    p = commands.add_parser('status')
    p.add_argument('--call-id', required=True)
    p = commands.add_parser('inspect')
    p.add_argument('--run-id', required=True); p.add_argument('--app', default='agent-harness')
    p = commands.add_parser('download')
    p.add_argument('--run-id', required=True); p.add_argument('--app', default='agent-harness')
    p.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        manifest = prepare_batch(json.loads(Path(args.config).read_text()), args.output)
        result = {'run_id': manifest['run_id'], 'episodes': len(manifest['jobs']), 'output': str(Path(args.output).resolve())}
    elif args.command == 'submit':
        result = submit_bundle(args.bundle, args.app)
    elif args.command == 'resume':
        import modal
        identifier(args.run_id); identifier(args.app)
        call = modal.Function.from_name(args.app, 'coordinate_run').spawn(args.run_id)
        result = {'run_id': args.run_id, 'call_id': call.object_id, 'status': 'submitted'}
    elif args.command == 'status':
        import modal
        try:
            result = modal.FunctionCall.from_id(args.call_id).get(timeout=0)
        except TimeoutError:
            result = {'call_id': args.call_id, 'status': 'pending'}
    elif args.command == 'inspect':
        journal = _journal(args.app, args.run_id)
        result = {'run_id': args.run_id, 'status': journal['status'], 'states': {}}
        for record in journal['episodes'].values():
            state = record['state']
            result['states'][state] = result['states'].get(state, 0) + 1
        if journal.get('summary'):
            result['groups'] = journal['summary']['groups']
    else:
        identifier(args.run_id)
        output = Path(args.output).resolve()
        output.mkdir(parents=True, exist_ok=False)
        total = 0
        for suffix in ('rollouts', 'grades', 'state'):
            volume = _volume(args.app, suffix)
            for entry in volume.listdir(args.run_id, recursive=True):
                # Modal FileEntryType.FILE is the public enum value 1.
                from modal.volume import FileEntryType
                if entry.type != FileEntryType.FILE:
                    continue
                relative = entry.path.lstrip('/')
                if not relative.startswith(args.run_id + '/'):
                    raise ValueError('Remote artifact escaped requested run')
                target = safe_join(output / suffix, relative)
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open('xb') as stream:
                    for chunk in volume.read_file(entry.path):
                        stream.write(chunk)
                total += 1
        result = {'run_id': args.run_id, 'files': total, 'output': str(output)}
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
