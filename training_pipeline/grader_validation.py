"""Bounded concurrent regrading and controls, without generating policy rollouts."""
import copy
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from .storage import read, atomic_json, digest


def run(config, spec):
    from .budget import SpendLedger
    from .collection import CollectionFactory
    from .contracts import InfrastructureError
    fixture = read(Path('/bundle') / spec['fixture'])
    assert fixture['fixture_hash'] == digest({k:v for k,v in fixture.items() if k != 'fixture_hash'})
    assert 0 < len(fixture['cases']) <= 250
    root = Path(config['output']); root.mkdir(parents=True, exist_ok=False)
    ledger = SpendLedger(config['spend']['ledger'], config['spend']['cap_usd'], config['spend']['prices'], config['model']['checkpoint_ttl_seconds'])
    before = read(ledger.path)['reserved_usd']; started = time.monotonic()
    report = {'status':'running', 'fixture_hash':fixture['fixture_hash'], 'cases':[], 'optimizer_updates':0,
              'purpose':fixture['purpose'], 'grader_identity':None}
    factory = CollectionFactory(config, root, ledger)
    # Reserve enough headroom for every in-flight grading attempt, including repairs.
    workers = 4
    j = config['judge']; p = j['prices']
    calls = (3 if config['environment'].get('grading_version') == 'all-claims-v7' else 2) * (1+j.get('repair_attempts',0))
    per_case_bound = calls * (j['context_tokens']*p['prefill'] + j['max_tokens']*p['sample'])/1e6
    def save():
        report.update(reserved_usd=read(ledger.path)['reserved_usd']-before, elapsed_seconds=time.monotonic()-started)
        atomic_json(root/'report.json', report)
    def grade(case):
        entry = {k:v for k,v in case.items() if k != 'request'}
        try:
            g = factory.grade(copy.deepcopy(case['request']))
            entry.update(grade=g, reward=g['training_feedback']['reward'])
        except (ValueError, KeyError, TypeError, InfrastructureError) as exc:
            entry.update(error_type=type(exc).__name__, error=str(exc)[:1000], reward=None)
        return entry
    try:
        report['grader_identity'] = factory.reward_version; save()
        # Batches keep the reservation ceiling conservative despite concurrent calls.
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for start in range(0,len(fixture['cases']),workers):
                batch = fixture['cases'][start:start+workers]
                if read(ledger.path)['reserved_usd']-before+len(batch)*per_case_bound > spec['reservation_limit_usd']:
                    raise ValueError('Validation reservation limit reached')
                for future in as_completed([pool.submit(grade,c) for c in batch]):
                    entry = future.result();report['cases'].append(entry);save()
                    print(entry['name'],entry['reward'],entry.get('error',''),flush=True)
        report['status']='complete';save()
    except Exception as exc:
        report.update(status='failed', error_type=type(exc).__name__, error=str(exc)[:1000])
        save()
        raise
    finally:
        factory.close();save()
