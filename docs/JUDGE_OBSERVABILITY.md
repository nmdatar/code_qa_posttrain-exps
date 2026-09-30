# Judge observability

Training collection already records raw extraction, strict assessment, and factual
coverage responses under each run's `private/` directory. These are structured
model answers, not hidden reasoning. Required-claim findings contain verdicts,
answer-span/evidence IDs, and short reasons; software computes scores afterward.

`Tracker.flush_answers()` now builds `judge-viewer.html` and `judge-viewer.json`
locally, including when W&B is disabled. It runs every 32 recorded trajectories
and at tracker finish. For enabled W&B tracking it also logs `judge_answers`
(one row per stage/attempt), `judge_viewer`, and the HTML artifact. Existing
running remote bundles need the updated code on their next launch.

The viewer joins episodes to their final strict and training scores, validated
semantic/coverage findings, reference claim text, and all available raw attempts.
It shows stop reasons, output token counts when recorded, validation failures,
judge identity, and existing timing fields. Null scores stay null. An absent raw
response is explicitly reported; an unreadable attempt does not hide other ones.
Coverage timing is unavailable in older records. No judge is invoked by this view.

Build a report for a saved run with:

```sh
python -m training_pipeline.judge_viewer /path/to/run --output /path/to/judge-viewer.html
```

The report exports selected grading fields and raw judge answers, not provider
prompts, source evidence blocks, credentials, or entire private directories.
Judge answers and reasons can themselves quote task/source content, so these
reports belong with the run's experiment artifacts. Existing public trajectory
exports retain their previous scope. W&B delivery remains best-effort; report
failures are recorded as tracking failures and cannot change grading or rewards.
