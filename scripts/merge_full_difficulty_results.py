"""Merge the archived complement with preserved historical teacher outcomes."""
from collections import Counter
from pathlib import Path

from training_pipeline.collection_recovery import scientific_identity, verified_archive_inventory
from training_pipeline.storage import atomic_json, read


def main():
    out = Path('reports/task-difficulty-full-v1')
    old = read('reports/expanded-studies/teacher-merged-audit.json')
    root = Path('artifacts/task-difficulty-full-v1-results/artifacts/experiments/task-difficulty-full-v1-remaining')
    config, records, proof = verified_archive_inventory(root, out / 'archive-checksums.json')
    expected = read('configs/experiments/task-difficulty-full-v1/remaining-tasks.json')
    local = read('configs/experiments/task-difficulty-full-v1/remaining.json')
    if scientific_identity(config) != scientific_identity(local):
        raise ValueError('New run scientific identity changed')
    if config['benchmark']['manifest_hash'] != expected['manifest_hash']:
        raise ValueError('New run evaluated a different cohort')
    counts = Counter(r['trace']['task_id'] for r in records)
    if set(counts) != set(expected['task_ids']) or set(counts.values()) != {2}:
        raise ValueError('The expansion needs exactly two archived attempts for every remaining task')
    if set(counts) & {r['task_id'] for r in old['slot_rows']}:
        raise ValueError('Expansion overlaps preserved outcomes')
    rows = list(old['slot_rows'])
    slots = Counter()
    for record in sorted(records, key=lambda r: r['trace']['episode_id']):
        t = record['trace']
        if t['split'] != 'train' or t['policy_id'] != 'base:' + config['model']['base_model'] or t['run_id'] != config['run_id']:
            raise ValueError('Unexpected task split, model or run')
        v = t['verification']
        rows.append({'task_id': t['task_id'], 'attempt_slot': slots[t['task_id']],
                     'origin': 'full-pool-expansion', 'path': record['path'], 'sha256': record['sha256'],
                     'episode_id': t['episode_id'], 'termination': t['termination'],
                     'verification_status': v['status'], 'strict_score': v.get('diagnostics', {}).get('strict_score')})
        slots[t['task_id']] += 1
    combined = Counter(r['task_id'] for r in rows)
    if set(combined) != set(expected['full_manifest']['task_ids']) or set(combined.values()) != {2}:
        raise ValueError('Merged archive does not cover the full training pool exactly twice')
    ledger = read('artifacts/task-difficulty-full-v1-results/artifacts/project-budget/task-difficulty-full-v1/remaining.json')
    value = {'kind': 'full-training-pool-teacher-union-v1',
             'source_proofs': {**old['source_proofs'], 'full_pool_expansion': proof},
             'slot_rows': rows, 'new_rollouts': len(records), 'reused_rollouts': len(old['slot_rows']),
             'new_run_reserved_usd': ledger['reserved_usd'], 'distinct_training_tasks': len(combined),
             'confirmation_used': False}
    atomic_json(out / 'merged-audit.json', value)
    print({k:v for k,v in value.items() if k not in {'slot_rows','source_proofs'}})


if __name__ == '__main__':
    main()
