# Repository RL acceptance evidence

Model: `Qwen/Qwen3.5-4B`. Run: `live-repository-qwen4b-004`.

Acknowledged optimizer updates: **1**. Development evaluation steps: [0, 1].

Conservatively charged/reserved across campaign: **$14.437**. Actual provider billing is not yet known.

## Evidence

- [Durable events](/Users/ndatar/.codex/worktrees/dataset-generation/action-interview/artifacts/posttrain/live-repository-qwen4b-004/events.jsonl)
- [Local dashboard](/Users/ndatar/.codex/worktrees/dataset-generation/action-interview/artifacts/posttrain/live-repository-qwen4b-004/report.html)
- [Checkpoint manifest](/Users/ndatar/.codex/worktrees/dataset-generation/action-interview/artifacts/posttrain/live-repository-qwen4b-004/checkpoints/step-000001-9eecce30c0/manifest.json)

## Limits

- Diagnostic integration only; not evidence of improved model quality.
- Human reward calibration remains pending.
- Two development tasks used in paid diagnostic; full20-task configuration is separate.

## Checkpoint validation

[Reload evidence](../reports/posttrain/qwen4b-checkpoint-reload.json) verifies sampling, optimizer-state loading, and a fresh-optimizer fork. Same226 assistant tokens had maximum absolute log-probability change0.11457 after the update. This confirms changed sampling behavior, not improved generalization.

## Completion categories

- Core implementation:239 offline regression tests pass; CLI, SFT/GRPO, checkpointing, reports, and periodic evaluation are implemented. Remote deployment remains unverified; live SFT data requires admitted blind-solver trajectories.
- Live repository integration:one Qwen3.5-4B GRPO update, committed checkpoint, reload/optimizer/fork checks, and development evaluation at steps0 and1 succeeded.
- Longer experiment readiness:human reward calibration remains pending. The two-task diagnostic acceptance rate changed from2/2 before training to0/2 after, with both post-update answers receiving partial credit. No model-quality improvement is established.

[W&B run](https://wandb.ai/nmdatar-harvard-university/repo-qa-posttrain/runs/73c08e6acd0f52129ee7c5d9).
