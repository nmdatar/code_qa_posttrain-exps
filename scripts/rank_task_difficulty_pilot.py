"""Rank a preselected 100-task cohort using hash-verified existing teacher attempts."""
import csv
import argparse
import html
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from training_pipeline.benchmark import task_manifest
from training_pipeline.collection_recovery import scientific_identity
from training_pipeline.config import inputs
from training_pipeline.storage import atomic_json, digest, read
from training_pipeline.trajectory_viewer import render_report


def rank_and_select(rows):
    scored = [r for r in rows if r['score'] is not None]
    pending = [r for r in rows if r['score'] is None]
    scored.sort(key=lambda r: (-r['score'], digest({'seed': 42, 'task_id': r['task_id']})))
    trim = math.ceil(len(scored) * .2)
    selected = {r['task_id'] for r in scored[trim:len(scored)-trim]}
    for rank, row in enumerate(scored, 1):
        row.update(rank=rank, percentile_band_selected=row['task_id'] in selected,
                   score_band_selected=.2 <= row['score'] <= .8)
    for row in pending:
        row.update(rank=None, percentile_band_selected=False, score_band_selected=False)
    return scored + sorted(pending, key=lambda r: r['task_id']), trim


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--all-training', action='store_true')
    parser.add_argument('--count', type=int, default=100)
    parser.add_argument('--audit', default='reports/expanded-studies/teacher-merged-audit.json')
    parser.add_argument('--output', default='reports/task-difficulty-pilot-v1')
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    config = read('configs/experiments/strong-model-validation-v1/strong.json')
    data = inputs(config)
    count = len(data['tasks']) if args.all_training else args.count
    cohort = task_manifest(data, count)  # Fixed cohort independent of outcomes.
    atomic_json(out / 'sample-manifest.json', cohort)
    tasks = {t['id']: t for t in data['tasks']}
    audit_path = Path(args.audit)
    audit = read(audit_path)
    for proof in audit['source_proofs'].values():
        source = Path(proof['archive_root']) / 'config.json'
        if hashlib.sha256(source.read_bytes()).hexdigest() != proof['config_sha256']:
            raise ValueError('Archived model configuration changed')
        if scientific_identity(read(source)) != scientific_identity(config):
            raise ValueError('Historical teacher settings do not match the current stronger model')
    episodes = defaultdict(list)
    trace_records = []
    versions = set()
    slots = set()
    for row in audit['slot_rows']:
        if row['task_id'] not in cohort['task_ids']:
            continue
        path = Path(row['path'])
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != row['sha256']:
            raise ValueError('Archived trajectory changed')
        t = json.loads(raw)
        key = (row['task_id'], row['attempt_slot'])
        if key in slots or t['task_id'] != row['task_id'] or t['split'] != 'train' or t['policy_id'] != 'base:' + config['model']['base_model']:
            raise ValueError('Duplicate slot or mismatched task/model/split')
        slots.add(key)
        v = t['verification']
        d = v.get('diagnostics') or {}
        score = v['reward'] if v['status'] == 'resolved' else None
        if score is not None and not 0 <= score <= 1:
            raise ValueError('Invalid score')
        versions.add(v['version'])
        episodes[t['task_id']].append({
            'attempt_slot': row['attempt_slot'], 'episode_id': t['episode_id'],
            'score': score, 'strict_score': d.get('strict_score'),
            'termination': t['termination'], 'grading_status': v['status'],
            'reasons': v.get('reasons', []),
            'strict_review_flag': bool((d.get('semantic') or {}).get('needs_review')),
            'path': str(path), 'sha256': row['sha256'],
        })
        trace_records.append((t, {'phase': 'difficulty pilot', 'optimizer_step': 0}))
    if len(versions) != 1 or len(episodes) != count or any(len(v) != 2 for v in episodes.values()):
        raise ValueError('Expected exactly two matching-version attempts for each sampled task')
    rows = []
    for ident in cohort['task_ids']:
        attempts = sorted(episodes[ident], key=lambda r: r['attempt_slot'])
        scores = [a['score'] for a in attempts]
        complete = all(s is not None for s in scores)
        known = [s for s in scores if s is not None]
        rows.append({
            'task_id': ident, 'family': tasks[ident]['family_id'],
            'question': tasks[ident]['public']['user_prompt'],
            'score': sum(scores)/2 if complete else None,
            'resolved_attempts': len(known), 'attempt_scores': scores,
            'score_lower_bound': sum(known)/2,
            'score_upper_bound': (sum(known)+2-len(known))/2,
            'strict_passes': sum(a['strict_score'] == 1 for a in attempts),
            'strict_review_flag': any(a['strict_review_flag'] for a in attempts),
            'attempts': attempts,
        })
    ranked, trim = rank_and_select(rows)
    atomic_json(out / 'ranked-tasks.json', ranked)
    fields = ['rank', 'task_id', 'family', 'score', 'resolved_attempts', 'attempt_scores',
              'score_lower_bound', 'score_upper_bound', 'strict_passes', 'strict_review_flag',
              'percentile_band_selected', 'score_band_selected', 'question']
    with (out / 'ranked-tasks.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: json.dumps(r[k]) if isinstance(r[k], list) else r[k] for k in fields} for r in ranked)
    selections = {}
    for name, field in [('percentile-20-80', 'percentile_band_selected'), ('score-20-80', 'score_band_selected')]:
        selected = [r for r in ranked if r[field]]
        ids = [r['task_id'] for r in selected]
        selection = {
            'kind': 'task-difficulty-selection-v1', 'data_identity': data['identity'],
            'sample_manifest_hash': cohort['manifest_hash'], 'model': config['model']['base_model'],
            'selection_rule': name, 'task_ids': ids, 'count': len(ids),
            'families': dict(Counter(r['family'] for r in selected)),
            'score_distribution': dict(Counter(str(r['score']) for r in selected)),
            'confirmation_used': False,
        }
        selection['manifest_hash'] = digest(selection)
        atomic_json(out / (name + '-manifest.json'), selection)
        with (out / (name + '-tasks.jsonl')).open('w') as f:
            for ident in ids:
                f.write(json.dumps(tasks[ident]['public']) + '\n')
        selections[name] = selection
    summary = {
        'sampled_tasks': count, 'families': sum(n > 0 for n in cohort['families'].values()),
        'reused_attempts': len(trace_records) - audit.get('new_rollouts', 0),
        'new_rollouts': audit.get('new_rollouts', 0),
        'new_model_calls': None if audit.get('new_rollouts') else 0,
        'incremental_model_cost_usd': None if audit.get('new_rollouts') else 0,
        'new_run_reserved_usd': audit.get('new_run_reserved_usd', 0),
        'rankable_tasks': sum(r['score'] is not None for r in ranked),
        'unranked_task_ids': [r['task_id'] for r in ranked if r['score'] is None],
        'ranking_metric': 'Mean factual assertion-coverage reward across two attempts; higher is easier.',
        'score_distribution': dict(Counter(str(r['score']) for r in ranked)),
        'percentile_trim_per_end': trim,
        'tie_policy': 'Score descending, then deterministic seed-42 hash; tied tasks have no evidenced difficulty ordering.',
        'selections': selections, 'grading_version': next(iter(versions)),
        'source_audit_sha256': hashlib.sha256(audit_path.read_bytes()).hexdigest(),
        'scientific_identity': scientific_identity(config), 'confirmation_used': False,
        'limitations': 'Training-only exploratory sample. Two attempts estimate coverage, not calibrated solve probability. Same-family uncalibrated judge; strict review flags preserved. Grading errors stay unknown, never zero. No full-dataset curriculum has been applied.',
    }
    atomic_json(out / 'summary.json', summary)
    if count <= 100:
        (out / 'trajectories.html').write_text(render_report(trace_records, f'{count}-task difficulty pilot · {len(trace_records)} teacher traces'))
    else:
        (out / 'traces').mkdir(exist_ok=True)
        families = defaultdict(list)
        for trace, context in trace_records:
            families[tasks[trace['task_id']]['family_id']].append((trace, context))
        links = []
        for family, records in sorted(families.items()):
            name = digest(family)[:16] + '.html'
            (out / 'traces' / name).write_text(render_report(records, family + ' · teacher trajectories'))
            links.append(f'<li><a href="traces/{name}">{html.escape(family)}</a> — {len(records)} traces</li>')
        (out / 'trajectories.html').write_text('<!doctype html><meta charset="utf-8"><title>Full training-pool traces</title><h1>Full training-pool traces</h1><p>' + str(len(trace_records)) + ' trajectories, grouped by repository.</p><ul>' + ''.join(links) + '</ul>')
    print(json.dumps({k:v for k,v in summary.items() if k not in {'selections','grading_version'}}, indent=2))


if __name__ == '__main__':
    main()
