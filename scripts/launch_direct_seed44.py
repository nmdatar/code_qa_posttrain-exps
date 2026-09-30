"""Seed44 launch guard: require seed42 retirement readback before submission."""
import hashlib
import json
from pathlib import Path
import modal
from training_pipeline.remote import verify_bundle, submit

BUNDLE = 'artifacts/expanded-direct-seed44-v1-isolated-bundle'
manifest = verify_bundle(BUNDLE)
if manifest['bundle_id'] != '96fd9105d9a266a686c3dfc1c9320e0c29dbd4194665910b6ac37642534fdf1c':
    raise RuntimeError('Unexpected prepared seed44 bundle')
audit = json.loads(Path('reports/expanded-studies/seed42-allocation-retirement.json').read_text())
if (audit['status'] != 'retired' or audit['retired_cap_usd'] != 376.22296373843363
        or audit['project_ceiling_usd'] != 6000 or audit['seed43_allocation_still_reserved_usd'] != 1800):
    raise RuntimeError('Required completed seed42 retirement audit absent or changed')
volume = modal.Volume.from_name(audit['volume'])
live = b''.join(volume.read_file(audit['ledger']))
if hashlib.sha256(live).hexdigest() != audit['after_sha256']:
    raise RuntimeError('Retired seed42 ledger changed; reconcile before seed44 launch')
ledger = json.loads(live)
if ledger['cap'] != ledger['reserved_usd'] or ledger['cap'] != 376.22296373843363:
    raise RuntimeError('Seed42 still has unretired allocation')
print(json.dumps(submit(BUNDLE, isolated=True), indent=2))
