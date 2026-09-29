"""Paired, repository-clustered comparisons of frozen evaluation reports."""
import random
from statistics import mean

from .contracts import ConfigurationError


def compare(base, candidates, repetitions=10000, seed=20260928):
    if not candidates or repetitions < 1:
        raise ConfigurationError('Candidate reports and positive bootstrap repetitions required')
    def index(report):
        rows = {r['task_id']: r for r in report['results']}
        if not rows or len(rows) != len(report['results']) or len(rows) != report['expected']:
            raise ConfigurationError('Evaluation must contain every assigned task exactly once')
        for row in rows.values():
            if row['status'] not in ('resolved', 'unresolved') or not row.get('family_id'):
                raise ConfigurationError('Evaluation lacks status or repository family')
            if row['status'] == 'resolved' and (type(row['reward']) not in (int, float) or not 0 <= row['reward'] <= 1):
                raise ConfigurationError('Invalid resolved quality')
            if row['status'] == 'unresolved' and row['reward'] is not None:
                raise ConfigurationError('Unresolved quality must remain null')
        return rows
    control = index(base)
    arms = []
    for report in candidates:
        for key in ('data_identity', 'environment', 'reward_version', 'cohort'):
            if key not in base or key not in report or base[key] != report[key]:
                raise ConfigurationError('Incompatible evaluation '+key)
        arm = index(report)
        if set(arm) != set(control) or any(arm[t]['family_id'] != control[t]['family_id'] for t in control):
            raise ConfigurationError('Candidate/control tasks or families differ')
        arms.append(arm)
    def score(row, unknown=0):
        return row['reward'] if row['status'] == 'resolved' else unknown
    differences = {t: mean(score(a[t]) for a in arms)-score(control[t]) for t in control}
    families = {}
    for task, row in control.items():
        families.setdefault(row['family_id'], []).append(task)
    names = sorted(families)
    rng = random.Random(seed)
    values = []
    for _ in range(repetitions):
        tasks = [t for family in rng.choices(names, k=len(names)) for t in families[family]]
        values.append(mean(differences[t] for t in tasks))
    values.sort()
    interval = [values[int((repetitions-1)*p)] for p in (.025, .975)]
    coverage = [sum(r['status'] == 'resolved' for r in a.values())/len(a) for a in [control]+arms]
    completion = [mean(a[t]['termination'] == 'completed' for a in arms)-
                  (control[t]['termination'] == 'completed') for t in control]
    return {'method': 'paired_repository_cluster_bootstrap', 'seed': seed, 'repetitions': repetitions,
            'tasks': len(control), 'families': len(families), 'candidate_runs': len(arms),
            'quality_difference': mean(differences.values()), 'quality_difference_95ci': interval,
            'individual_run_differences': [mean(score(a[t])-score(control[t]) for t in control) for a in arms],
            'scoring_coverages': coverage, 'completion_difference': mean(completion),
            'unresolved_sensitivity_bounds': [
                mean(mean(score(a[t]) for a in arms)-score(control[t], 1) for t in control),
                mean(mean(score(a[t], 1) for a in arms)-score(control[t]) for t in control)],
            'passes_quality_gate': all(c >= .95 for c in coverage) and interval[0] > 0 and mean(completion) >= -.02,
            'note': 'Statistical gate only; seed provenance, finalist locking, judge calibration and archive restoration require separate evidence.'}
