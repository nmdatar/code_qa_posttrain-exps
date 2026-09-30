"""Summarize a frozen matched model-capability pilot without dropping failures."""
import argparse
import csv
import random
from collections import Counter
from pathlib import Path

from training_pipeline.storage import atomic_json, read


def summarize(results, output):
    manifest = read('configs/experiments/strong-model-validation-v1/tasks.json')
    ids = manifest['task_ids']
    arms, summary = {}, {}
    for arm in ('student', 'strong'):
        root = Path(results) / 'artifacts/experiments' / ('strong-model-validation-v1-' + arm)
        records = [read(p) for p in (root / 'trajectories').glob('*.json')]
        indexed = {t['task_id']: t for t in records}
        if len(indexed) != len(records) or set(indexed) - set(ids):
            raise ValueError('Unexpected or duplicate task attempt')
        rows = {}
        for ident in ids:
            t = indexed.get(ident, {})
            v = t.get('verification') or {}
            d = v.get('diagnostics') or {}
            coverage = d.get('training_coverage') or {}
            # The verifier resolves empty answers deterministically without a
            # semantic judge call. These are failures, not unknown grades.
            empty_answer_failure = (v.get('status') == 'resolved'
                                    and 'missing_answer_or_citations' in v.get('reasons', [])
                                    and v.get('reward') == 0)
            rows[ident] = {
                'episode_id': t.get('episode_id'),
                'termination': t.get('termination', 'missing'),
                'coverage': 0.0 if empty_answer_failure else coverage.get('score') if coverage.get('status') == 'resolved' else None,
                'strict_score': 0.0 if empty_answer_failure else d.get('strict_score') if d.get('strict_status') == 'resolved' else None,
                'deterministic_empty_answer_failure': empty_answer_failure,
                'coverage_claims': coverage.get('claims', []),
                'strict_reasons': v.get('reasons', []),
            }
        arms[arm] = rows
        values = [r['coverage'] for r in rows.values() if r['coverage'] is not None]
        strict = [r['strict_score'] for r in rows.values() if r['strict_score'] is not None]
        summary[arm] = {
            'attempted': len(records), 'expected': len(ids),
            'coverage_resolved': len(values),
            'mean_coverage_resolved': sum(values) / len(values) if values else None,
            'demonstrated_coverage_all_tasks': sum(values) / len(ids),
            'full_coverage': sum(x == 1 for x in values),
            'strict_resolved': len(strict), 'strict_passes': sum(x == 1 for x in strict),
            'terminations': dict(Counter(r['termination'] for r in rows.values())),
        }
        benchmark = read(root / 'benchmark.json')
        summary[arm]['rollout_and_grading_seconds'] = benchmark['wall_seconds']
        summary[arm]['rollout_and_grading_reserved_usd'] = benchmark['reserved_cost_usd']
        summary[arm]['concurrency'] = benchmark['concurrency']
    paired, deltas = [], []
    for ident in ids:
        a, b = arms['student'][ident], arms['strong'][ident]
        delta = None if a['coverage'] is None or b['coverage'] is None else b['coverage'] - a['coverage']
        if delta is not None:
            deltas.append(delta)
        paired.append({'task_id': ident, 'student_coverage': a['coverage'], 'strong_coverage': b['coverage'],
                       'delta': delta, 'student_strict': a['strict_score'], 'strong_strict': b['strict_score']})
    rng = random.Random(42)
    boot = sorted(sum(rng.choices(deltas, k=len(deltas))) / len(deltas) for _ in range(10000)) if deltas else []
    result = {
        'status': 'complete' if all(s['attempted'] == len(ids) for s in summary.values()) else 'incomplete',
        'task_manifest_hash': manifest['manifest_hash'], 'arms': summary,
        'paired': {'resolved': len(deltas), 'improved': sum(x > 0 for x in deltas),
                   'worsened': sum(x < 0 for x in deltas), 'tied': sum(x == 0 for x in deltas),
                   'mean_delta': sum(deltas) / len(deltas) if deltas else None,
                   'task_bootstrap_95_interval': [boot[250], boot[9749]] if boot else None},
        'assertions_changed': False, 'confirmation_used': False, 'optimizer_updates': 0,
        'limitations': '16 training tasks, one stochastic attempt per model; uncalibrated same-family judge. Exploratory capability check, not independent benchmark or proof every task is solvable. Unknown grades remain null; demonstrated coverage counts only observed credit over all scheduled tasks.',
    }
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    atomic_json(out / 'summary.json', result)
    atomic_json(out / 'task-details.json', arms)
    with (out / 'paired-results.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(paired[0]))
        writer.writeheader()
        writer.writerows(paired)
    return result


if __name__ == '__main__':
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', required=True)
    parser.add_argument('--output', default='reports/strong-model-validation-v1')
    args = parser.parse_args()
    print(json.dumps(summarize(args.results, args.output), indent=2))
