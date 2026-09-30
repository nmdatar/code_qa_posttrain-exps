"""Bounded Modal-only archive/restore check of an existing disposable checkpoint."""
import subprocess
from pathlib import Path
from .storage import read, atomic_json, digest
from .remote import verify_bundle, require_remote_worker, reallocate_ledger, controller_reservation


def worker(bundle):
    from .budget import SpendLedger
    from .tinker_backend import TinkerBackend
    m = verify_bundle(bundle)
    c = m['configs'][0]
    require_remote_worker(c)
    root = Path(c['output']); root.mkdir(parents=True, exist_ok=False)
    key = c['spend']['ledger'].removeprefix('/state/')
    allocation = m['budget_allocation']
    reallocate_ledger(c['spend']['ledger'], allocation['ledger_caps'][key], digest(allocation))
    ledger = SpendLedger(c['spend']['ledger'], c['spend']['cap_usd'], c['spend']['prices'], c['model']['checkpoint_ttl_seconds'])
    ledger.reserve_external('modal_controller', controller_reservation(c))
    backend = None
    report = {'status': 'running', 'optimizer_updates': 0, 'execution': 'modal', 'provider': 'tinker'}
    try:
        backend = TinkerBackend(c['model'], c['limits'], ledger)
        backend.create_trainer(c['seed'])
        artifacts = read(Path(bundle)/'archive-target.json')
        backend.load(artifacts, 'resume')
        report['archive'] = backend.archive(artifacts, root/'archive', 14*86400)
        report['status'] = 'passed'
    except BaseException as exc:
        report.update(status='failed', error_type=type(exc).__name__, error=str(exc)[:1000])
        raise
    finally:
        report['reserved_usd'] = read(ledger.path)['reserved_usd']
        atomic_json(root/'validation-report.json', report)
        subprocess.run(['sync','/state'],check=True)
        if backend: backend.close('success' if report['status']=='passed' else 'failed')
