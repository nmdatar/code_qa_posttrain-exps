# Full training-pool difficulty assessment

Assessed **all 858 eligible training tasks across 31 repository families**, using two Qwen3.5-397B-A17B attempts per task. Reused 1,200 hash-verified historical attempts for 600 tasks and collected 516 new attempts for the remaining 258. No development or confirmation tasks were used; no training updates were run.

## Selected dataset

- Recommended inclusive 20%–80% **score band**: **274 tasks across 31 families**, exported in `score-20-80-tasks.jsonl`. Its manifest freezes IDs, source-data identity and selection rule.
- Requested middle **percentile band**: **494 tasks**, exported separately in `percentile-20-80-tasks.jsonl`. Equal-score boundary ties can leave scores of 0 or 1 in this rank-based selection.
- **826 tasks ranked; 32 unranked** because one or both attempts lack a resolved factual grade. Unknown results are never treated as model failures or zeros. The ranked CSV includes known scores and bounds for these tasks.

Rank uses mean factual assertion-coverage reward over two attempts, highest first. Equal scores are ordered by a fixed seed-42 hash, not an evidenced difference in difficulty. Original questions, tools, episode budgets, source revisions, grading assertions and previous outcomes are preserved. Public task exports refer to the existing private reference/rubric records in the unchanged v6 release; the manifests are selection artifacts, not a rebuilt runnable release.

## Representation and solvability

`family-coverage.csv` reports source counts, retained counts, retention rates, missing grades and strict-pass support for every family. Families with no score-band candidates: none. All families were assessed; score filtering can still change their proportions, so coverage should not be confused with a distribution-matched dataset.

Of the selected tasks, **63 have at least one recorded strict pass** and **13 have a strict-review flag**. These are diagnostics, not additional selection rules. The score band estimates moderate factual solvability under this teacher and harness; it is not proof that every task can pass all strict gates or is trainable by the 4B student. The judge is uncalibrated and from the same model family; the earlier pilot exposed strict-grader inconsistencies. Two attempts provide a quick, noisy signal rather than a calibrated solve probability.

## Traces, execution and cost

`trajectories.html` indexes **all 1,716 traces** by repository. `ranked-tasks.json` retains source paths, hashes, both scores and grader flags; `merged-audit.json` preserves exact attempt-slot provenance. The new run used 32 concurrent rollout workers and 16 judges, with W&B online tables and full JSON traces. Original attempts used their recorded concurrency settings.

The user explicitly approved the expanded Modal upload and $900 ceiling after automatic review requested approval. The new campaign recorded **$178.15 in conservative reservations**, not provider invoices. Full traces and completion metadata were archived; no new optimizer or checkpoint was created. The source dataset is unchanged and this selection has not been silently applied to any existing training job.

Tracking finalization pending at archive time: **True**. All 516 new trace/grade records were independently checked against completed benchmark counts and trajectory events. If pending, the remote controller is still finishing W&B output; this is not an unfinished model evaluation. The local trace viewers contain all records regardless of that upload status.

Validation covered the exact disjoint remaining-task cohort, training-only access, tamper rejection, concurrent execution and trace viewer (19 tests), hash-verified archived inputs, two unique attempts per task, matching scientific and grader identities, and exported-task/manifest/split consistency.
