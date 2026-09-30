Open index.html to browse the historical trajectory viewers.

Sources are recorded in sources.json and manifest.json. Exact duplicate archives are deduplicated; recovered or otherwise distinct snapshots are preserved. All 46 W&B run identities present at inventory time have a viewer. Additional local-only runs are included. The active seed43 run is a snapshot with six unreadable/incomplete trajectory files explicitly omitted. W&B publishing is recorded per run in wandb-upload.json. Native tool_calls and trajectories tables are attached to the original runs through a report-only publisher; original training runs are not resumed. Each run summary contains trajectory_inspection_url.

Rebuild: PYTHONPATH=. python3 scripts/build_trajectory_history.py

Publish/backfill (requires authorized W&B access): PYTHONPATH=. .venv-eval/bin/python scripts/publish_historical_trajectories.py

In W&B, open the linked trajectory-inspection artifact, choose Files, then tool_calls.table.json. Filter episode_id and sort step to inspect the sequence. Table columns separate tool, arguments, response, error, elapsed_seconds, and model_output. The trajectories table holds scores, grading status and termination.
