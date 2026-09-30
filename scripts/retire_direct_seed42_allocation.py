"""Compare-and-retire finished seed42 allocation; dry-run unless --apply.

All known controllers on this private volume must be terminal. The complete
live ledger must equal the verified archive before any overwrite. Reservation
history, prices and reserved_usd are preserved; only cap and retirement audit
metadata change. Coordinate this script exclusively: Volume upload is not a
transactional compare-and-swap against a newly launched concurrent controller.
"""
import argparse
import copy
import hashlib
import io
import json
from pathlib import Path

EXPECTED_VOLUME = 'qa-state-gfdcqtujymxc7f7txuqzo2myfxaaq75gwimkvcvdx75one3of4cq'
LEDGER = 'artifacts/project-budget/expanded-direct-seed42-v2-isolated.json'
ARCHIVE = Path('artifacts/expanded-direct-seed42-completed-results') / LEDGER
ARCHIVE_REPORT = Path('reports/expanded-studies/direct-seed42-completed-archive.json')
AUDIT = Path('reports/expanded-studies/seed42-allocation-retirement.json')
REMOTE_AUDIT = 'artifacts/project-budget/expanded-direct-seed42-v2-retirement.json'
EXPECTED_RESERVED = 376.22296373843363


def retirement_state(original):
    if original['cap'] != 1800 or original['reserved_usd'] != EXPECTED_RESERVED:
        raise ValueError('Unexpected original seed42 accounting totals')
    result = copy.deepcopy(original)
    result['cap'] = original['reserved_usd']
    result.setdefault('allocation_history', []).append({
        'previous_cap': 1800, 'cap': original['reserved_usd'],
        'reason': 'Completed seed42; retire unused ceiling for seed44 under unchanged6000project cap'})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    import modal
    archived_bytes = ARCHIVE.read_bytes()
    report = json.loads(ARCHIVE_REPORT.read_text())
    matches = [item for item in report['files'] if item['path'] == LEDGER]
    if len(matches) != 1 or hashlib.sha256(archived_bytes).hexdigest() != matches[0]['sha256']:
        raise RuntimeError('Archived authoritative ledger hash does not match verified archive')
    original = json.loads(archived_bytes)
    retired = retirement_state(original)
    receipts = []
    for path in Path('artifacts').glob('*.submission.json'):
        item = json.loads(path.read_text())
        if item.get('volume') == EXPECTED_VOLUME and item.get('sandbox_id'):
            receipts.append((str(path), item['sandbox_id']))
    mandatory = {'artifacts/expanded-direct-seed42-v2-isolated-bundle.submission.json',
                 'artifacts/direct-v2-continuation.submission.json'}
    if not mandatory <= {path for path, _ in receipts}:
        raise RuntimeError('Missing original or continuation controller receipt')
    observed = []
    for path, sandbox_id in receipts:
        sandbox = modal.Sandbox.from_id(sandbox_id)
        try:
            code = sandbox.poll()
            if code is None:
                raise RuntimeError('Known controller still active: ' + sandbox_id)
            observed.append({'receipt': path, 'sandbox_id': sandbox_id, 'exit_code': code})
        finally:
            sandbox.detach()
    volume = modal.Volume.from_name(EXPECTED_VOLUME)
    live_bytes = b''.join(volume.read_file(LEDGER))
    if json.loads(live_bytes) != original:
        raise RuntimeError('Live ledger differs from authoritative completed archive; reconcile instead of overwriting')
    audit = {'status': 'dry_run_validated', 'volume': EXPECTED_VOLUME, 'ledger': LEDGER,
             'before_sha256': hashlib.sha256(live_bytes).hexdigest(),
             'archive_sha256': hashlib.sha256(archived_bytes).hexdigest(),
             'prior_cap_usd': 1800, 'retired_cap_usd': EXPECTED_RESERVED,
             'reserved_usd': EXPECTED_RESERVED, 'released_allocation_usd': 1800-EXPECTED_RESERVED,
             'reservations_preserved': len(original['reservations']), 'controllers': observed,
             'project_ceiling_usd': 6000, 'seed43_allocation_still_reserved_usd': 1800}
    if not args.apply:
        print(json.dumps(audit, indent=2))
        return
    if AUDIT.exists():
        raise RuntimeError('Local retirement audit already exists; inspect instead of repeating')
    # Final authoritative read immediately before replacement. Parent workflow
    # must not start another writer on this retired private volume.
    if b''.join(volume.read_file(LEDGER)) != live_bytes:
        raise RuntimeError('Live ledger changed during retirement preflight')
    before_path = AUDIT.with_name('seed42-ledger-before-retirement.json')
    before_path.write_bytes(live_bytes)
    new_bytes = (json.dumps(retired, indent=2, sort_keys=True)+'\n').encode()
    audit.update(status='retired', after_sha256=hashlib.sha256(new_bytes).hexdigest())
    audit_bytes = (json.dumps(audit, indent=2, sort_keys=True)+'\n').encode()
    with volume.batch_upload(force=True) as upload:
        upload.put_file(io.BytesIO(new_bytes), LEDGER)
        upload.put_file(io.BytesIO(audit_bytes), REMOTE_AUDIT)
    after_bytes = b''.join(volume.read_file(LEDGER))
    if after_bytes != new_bytes:
        raise RuntimeError('Retirement readback differs; stop and inspect before launching seed44')
    AUDIT.write_bytes(audit_bytes)
    AUDIT.with_name('seed42-ledger-after-retirement.json').write_bytes(after_bytes)
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()
