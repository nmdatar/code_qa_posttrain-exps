"""Prepare/retire one completed private allocation, preserving every reservation.

Default execution performs remote read-only validation; --apply additionally
requires --exclusive-volume-control. This file was prepared without execution.
Never run concurrently with any writer to the target private volume. Modal
Volume upload is not transactional compare-and-swap; polling known controllers
cannot detect an unregistered writer. Does not edit frozen project budgets or
the separate control-evaluation allocation.
"""
import argparse
import copy
import hashlib
import io
import json
from pathlib import Path

PROFILES = {
    43: {
        'run': 'expanded-direct-seed43-v1-continued',
        'ledger_name': 'expanded-direct-seed43-v1-isolated.json',
        'volume': 'qa-state-bbxsv6dcorr5n2rpar2qd242kq7mswr4q25pcg4mqhjvivf4oybq',
        'reserved': 364.8522665304345,
        'receipts': ['expanded-direct-seed43-v1-isolated-bundle.submission.json',
                     'direct-seed43-continuation.submission.json'],
    },
    44: {
        'run': 'expanded-direct-seed44-v1',
        'ledger_name': 'expanded-direct-seed44-v1-isolated.json',
        'volume': 'qa-state-s36zcbozujtknbwd37a4smqobqu5xvazizszcc3kyn3eeu2p34oa',
        'reserved': 343.0981421804351,
        'receipts': ['expanded-direct-seed44-v1-isolated-bundle.submission.json'],
    },
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def retirement_state(original, seed):
    expected = PROFILES[seed]['reserved']
    if original['cap'] != 1800 or original['reserved_usd'] != expected:
        raise ValueError('Unexpected original accounting totals; do not overwrite')
    result = copy.deepcopy(original)
    result['cap'] = expected
    result.setdefault('allocation_history', []).append({
        'previous_cap': 1800, 'cap': expected,
        'reason': f'Completed direct seed{seed}; retire unused private allocation; '
                  'frozen project ceiling and separate control allocation unchanged',
    })
    # Explicit preservation invariant covers reservations, prices, TTL and any
    # additional historical fields, including existing allocation history.
    restored = copy.deepcopy(result)
    restored['cap'] = original['cap']
    if 'allocation_history' in original:
        restored['allocation_history'] = copy.deepcopy(original['allocation_history'])
    else:
        restored.pop('allocation_history')
    if restored != original:
        raise RuntimeError('Retirement modified non-allocation state')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, choices=sorted(PROFILES), required=True)
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--exclusive-volume-control', action='store_true',
                        help='Operator confirms no other process may write this volume')
    args = parser.parse_args()
    if args.apply and not args.exclusive_volume_control:
        parser.error('--apply requires --exclusive-volume-control')
    seed, profile = args.seed, PROFILES[args.seed]
    ledger = 'artifacts/project-budget/' + profile['ledger_name']
    archive_root = Path(f'artifacts/expanded-direct-seed{seed}-completed-results')
    archived = (archive_root / ledger).read_bytes()
    archive_report = Path(f'reports/expanded-studies/direct-seed{seed}-completed-archive.json')
    report = json.loads(archive_report.read_text())
    if report.get('parse_errors') != 0 or report.get('local_hashes_reverified') is not True:
        raise RuntimeError('Completed archive audit missing/failed')
    if Path(report['root']).resolve() != archive_root.resolve():
        raise RuntimeError('Archive audit targets a different root')
    matches = [r for r in report['files'] if r['path'] == ledger]
    if len(matches) != 1 or matches[0]['sha256'] != sha(archived):
        raise RuntimeError('Archived authoritative ledger hash mismatch')
    run_root = archive_root / 'artifacts/experiments' / profile['run']
    event_path = run_root / 'events.jsonl'
    event_bytes = event_path.read_bytes()
    event_matches = [r for r in report['files'] if r['path'] == str(event_path.relative_to(archive_root))]
    if len(event_matches) != 1 or event_matches[0]['sha256'] != sha(event_bytes):
        raise RuntimeError('Completed run events differ from audited archive')
    events = [json.loads(line) for line in event_bytes.decode().splitlines()]
    if not events or events[-1].get('event') != 'run_finish' or events[-1].get('status') != 'complete':
        raise RuntimeError('Completed-run terminal event absent')
    original = json.loads(archived)
    retired = retirement_state(original, seed)
    audit_path = Path(f'reports/expanded-studies/seed{seed}-allocation-retirement.json')
    remote_audit = f'artifacts/project-budget/expanded-direct-seed{seed}-v1-retirement.json'
    if args.apply and audit_path.exists():
        raise RuntimeError('Retirement audit already exists; inspect rather than replay')
    receipts = []
    for path in sorted(Path('artifacts').glob('*.submission.json')):
        item = json.loads(path.read_text())
        if item.get('volume') == profile['volume']:
            if not item.get('sandbox_id'):
                raise RuntimeError('Known volume receipt lacks controller identity: ' + str(path))
            receipts.append((path, item['sandbox_id']))
    if not set(profile['receipts']) <= {p.name for p, _ in receipts}:
        raise RuntimeError('Mandatory original/continuation receipt missing')
    import modal

    def terminal_controllers():
        result = []
        for path, ident in receipts:
            sandbox = modal.Sandbox.from_id(ident)
            try:
                code = sandbox.poll()
                if code is None:
                    raise RuntimeError('Controller is active: ' + ident)
                result.append({'receipt': str(path), 'sandbox_id': ident, 'exit_code': code})
            finally:
                sandbox.detach()
        return result

    controllers = terminal_controllers()
    volume = modal.Volume.from_name(profile['volume'])
    live = b''.join(volume.read_file(ledger))
    if live != archived:
        raise RuntimeError('Live ledger bytes differ from authoritative archive; reconcile first')
    audit = {'status': 'dry_run_validated', 'seed': seed, 'volume': profile['volume'],
             'ledger': ledger, 'archive_report': str(archive_report),
             'archive_sha256': sha(archived), 'before_sha256': sha(live),
             'prior_cap_usd': 1800, 'retired_cap_usd': profile['reserved'],
             'reserved_usd': profile['reserved'],
             'released_allocation_usd': 1800 - profile['reserved'],
             'reservations_preserved': len(original['reservations']),
             'controllers': controllers, 'project_ceiling_usd': 6000,
             'separate_control_allocation_usd': 100, 'frozen_budget_edited': False,
             'exclusive_volume_control_confirmed': args.exclusive_volume_control}
    if not args.apply:
        print(json.dumps(audit, indent=2))
        return
    # Fail closed on changes during preflight. This is still not a remote CAS.
    if terminal_controllers() != controllers or b''.join(volume.read_file(ledger)) != live:
        raise RuntimeError('Controller or ledger changed during preflight')
    before = audit_path.with_name(f'seed{seed}-ledger-before-retirement.json')
    with before.open('xb') as stream:
        stream.write(live)
    new_bytes = (json.dumps(retired, indent=2, sort_keys=True) + '\n').encode()
    audit.update(status='retired', after_sha256=sha(new_bytes))
    audit_bytes = (json.dumps(audit, indent=2, sort_keys=True) + '\n').encode()
    with volume.batch_upload(force=True) as upload:
        upload.put_file(io.BytesIO(new_bytes), ledger)
        upload.put_file(io.BytesIO(audit_bytes), remote_audit)
    after = b''.join(volume.read_file(ledger))
    remote_audit_bytes = b''.join(volume.read_file(remote_audit))
    if after != new_bytes or remote_audit_bytes != audit_bytes:
        raise RuntimeError('Retirement readback mismatch; stop, do not launch or replay')
    audit_path.write_bytes(audit_bytes)
    audit_path.with_name(f'seed{seed}-ledger-after-retirement.json').write_bytes(after)
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()
