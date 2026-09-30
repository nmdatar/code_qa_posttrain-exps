"""Publish family coverage and a concise full-pool assessment report."""
import csv
from collections import Counter
from pathlib import Path

from training_pipeline.storage import atomic_json, read


def main():
    root = Path('reports/task-difficulty-full-v1')
    summary = read(root / 'summary.json')
    rows = read(root / 'ranked-tasks.json')
    score_set = [r for r in rows if r['score_band_selected']]
    percentile_set = [r for r in rows if r['percentile_band_selected']]
    fields = ['family', 'total_tasks', 'ranked_tasks', 'unranked_tasks', 'score_band_tasks',
              'percentile_band_tasks', 'retention_rate', 'selected_with_strict_pass', 'selected_with_review_flag']
    families = []
    for family in sorted({r['family'] for r in rows}):
        source = [r for r in rows if r['family'] == family]
        selected = [r for r in source if r['score_band_selected']]
        families.append(dict(zip(fields, [family, len(source), sum(r['score'] is not None for r in source),
            sum(r['score'] is None for r in source), len(selected), sum(r['percentile_band_selected'] for r in source),
            len(selected)/len(source), sum(r['strict_passes'] > 0 for r in selected),
            sum(r['strict_review_flag'] for r in selected)])))
    with (root / 'family-coverage.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader(); writer.writerows(families)
    selected_ids = {r['task_id'] for r in score_set}
    exported = [__import__('json').loads(line) for line in (root/'score-20-80-tasks.jsonl').read_text().splitlines()]
    assert len(exported) == len(selected_ids) and {t['id'] for t in exported} == selected_ids
    assert all(t['split'] == 'train' for t in exported)
    assert len(rows) == 858 and len({r['task_id'] for r in rows}) == 858
    manifest = read(root/'score-20-80-manifest.json')
    assert set(manifest['task_ids']) == selected_ids
    unknown = sum(r['score'] is None for r in rows)
    omitted = [f['family'] for f in families if not f['score_band_tasks']]
    strict = sum(r['strict_passes'] > 0 for r in score_set)
    reviews = sum(r['strict_review_flag'] for r in score_set)
    summary['family_coverage'] = families
    summary['selected_with_at_least_one_strict_pass'] = strict
    summary['selected_with_strict_review_flag'] = reviews
    archive = read(root/'archive-status.json')
    summary['tracking_finalization_pending'] = archive['tracking_finalization_pending']
    atomic_json(root/'summary.json', summary)
    status = read(root/'status.json')
    status.update(status='assessment_complete_tracking_pending' if archive['tracking_finalization_pending'] else 'complete',
                  tracking_finalization_pending=archive['tracking_finalization_pending'],
                  attempted_training_tasks=858, recorded_trajectories=1716,
                  ranked_tasks=858-unknown, unranked_tasks=unknown, selected_tasks=len(score_set),
                  selected_families=len(families)-len(omitted), new_run_reserved_usd=summary['new_run_reserved_usd'])
    atomic_json(root/'status.json', status)
    (root/'README.md').write_text(f'''# Full training-pool difficulty assessment

Assessed **all 858 eligible training tasks across {len(families)} repository families**, using two Qwen3.5-397B-A17B attempts per task. Reused 1,200 hash-verified historical attempts for 600 tasks and collected 516 new attempts for the remaining 258. No development or confirmation tasks were used; no training updates were run.

## Selected dataset

- Recommended inclusive 20%–80% **score band**: **{len(score_set)} tasks across {len(families)-len(omitted)} families**, exported in `score-20-80-tasks.jsonl`. Its manifest freezes IDs, source-data identity and selection rule.
- Requested middle **percentile band**: **{len(percentile_set)} tasks**, exported separately in `percentile-20-80-tasks.jsonl`. Equal-score boundary ties can leave scores of 0 or 1 in this rank-based selection.
- **{858-unknown} tasks ranked; {unknown} unranked** because one or both attempts lack a resolved factual grade. Unknown results are never treated as model failures or zeros. The ranked CSV includes known scores and bounds for these tasks.

Rank uses mean factual assertion-coverage reward over two attempts, highest first. Equal scores are ordered by a fixed seed-42 hash, not an evidenced difference in difficulty. Original questions, tools, episode budgets, source revisions, grading assertions and previous outcomes are preserved. Public task exports refer to the existing private reference/rubric records in the unchanged v6 release; the manifests are selection artifacts, not a rebuilt runnable release.

## Representation and solvability

`family-coverage.csv` reports source counts, retained counts, retention rates, missing grades and strict-pass support for every family. Families with no score-band candidates: {', '.join(omitted) if omitted else 'none'}. All families were assessed; score filtering can still change their proportions, so coverage should not be confused with a distribution-matched dataset.

Of the selected tasks, **{strict} have at least one recorded strict pass** and **{reviews} have a strict-review flag**. These are diagnostics, not additional selection rules. The score band estimates moderate factual solvability under this teacher and harness; it is not proof that every task can pass all strict gates or is trainable by the 4B student. The judge is uncalibrated and from the same model family; the earlier pilot exposed strict-grader inconsistencies. Two attempts provide a quick, noisy signal rather than a calibrated solve probability.

## Traces, execution and cost

`trajectories.html` indexes **all 1,716 traces** by repository. `ranked-tasks.json` retains source paths, hashes, both scores and grader flags; `merged-audit.json` preserves exact attempt-slot provenance. The new run used 32 concurrent rollout workers and 16 judges, with W&B online tables and full JSON traces. Original attempts used their recorded concurrency settings.

The user explicitly approved the expanded Modal upload and $900 ceiling after automatic review requested approval. The new campaign recorded **${summary['new_run_reserved_usd']:.2f} in conservative reservations**, not provider invoices. Full traces and completion metadata were archived; no new optimizer or checkpoint was created. The source dataset is unchanged and this selection has not been silently applied to any existing training job.

Tracking finalization pending at archive time: **{archive['tracking_finalization_pending']}**. All 516 new trace/grade records were independently checked against completed benchmark counts and trajectory events. If pending, the remote controller is still finishing W&B output; this is not an unfinished model evaluation. The local trace viewers contain all records regardless of that upload status.

Validation covered the exact disjoint remaining-task cohort, training-only access, tamper rejection, concurrent execution and trace viewer (19 tests), hash-verified archived inputs, two unique attempts per task, matching scientific and grader identities, and exported-task/manifest/split consistency.
''')
    print({k:status[k] for k in ('status','attempted_training_tasks','ranked_tasks','unranked_tasks','selected_tasks','selected_families','new_run_reserved_usd')})


if __name__ == '__main__':
    main()
