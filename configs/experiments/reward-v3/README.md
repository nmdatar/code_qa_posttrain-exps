# Supported-coverage reward v3 diagnostics

These opt-in configurations use the unchanged Qwen3.5-4B solver and the versioned
all-claims-v5 Qwen397 judge. `diagnostic.json` collects four fresh attempts for each
of four frozen family-stratified training tasks, with no optimizer. The diagnostic
was executed; its complete results and all earlier iterations are in
`reports/reward-shaping-v3/README.md`.

`selection.json` is a prepared fresh strict control if another control is needed.
The implemented v3 change affects only uncited training answers, so the reported
strict control is the already completed v2/grader-v5 selection evaluation. The
selection config in this directory has not been submitted. Do not confuse a
prepared config with a completed run.

These results do not authorize GRPO launch. Current training completion, scoring
coverage and group-variation gates fail. All paid checks use the existing shared
baseline ledger and the $300 cumulative allocation; historical reservations remain.
Do not relaunch any submitted run ID or bundle. Refresh cloud ledger history before
preparing another diagnostic, and retain failed attempts.

The original Experiment 1/2 launch files remain unchanged. Full throughput selection,
fresh full-cohort controls, and reference/judge reliability still need resolution.
See `docs/REWARD_SHAPING.md` for the reward formula, anti-gaming checks and routing.
