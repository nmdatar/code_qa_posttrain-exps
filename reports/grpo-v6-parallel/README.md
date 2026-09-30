# Frozen v6 baseline and parallel GRPO smoke — completed

Both arms completed successfully in one Modal controller. The user-authorized subagent independently monitored, archived and summarized the GRPO arm. Both use `repo-qa-training-claims-v6`: 78 training rubrics rewritten as 316 explicit claims, with source evidence repaired. Selection rubrics and confirmation membership were preserved; confirmation was not evaluated.

## Results

| Measurement | Independent base | Smoke initial | Smoke final |
|---|---:|---:|---:|
| Selection episodes | 32 | 32 | 32 |
| Resolved strict grades | 28 | 29 | 29 |
| Full strict passes | 7 | 8 | 4 |
| Total strict credit | 7.0 | 8.0 | 4.3333 |
| Strict credit / all attempts | 21.875% | 25.0% | 13.542% |

Unknown grades remain unknown; credit/all attempts measures demonstrated credit, not an assertion that unresolved answers are wrong. One final answer earned one-third credit. The within-smoke comparison is primary; the independent baseline is a repeated untrained control. This run does **not** demonstrate improvement.

The smoke completed six batches (two tasks × four trajectories each), 48 resolved training rewards, four acknowledged optimizer updates and five checkpoint saves including the initial checkpoint. Five of 12 groups had reward variation; 19 trajectories had nonzero advantages. Mean training reward was 0.25625. Batch means were 0.4375, 0, 0, 0.375, 0.475, 0.25. This is a mechanics/signal smoke, not a full GRPO training run.

## Challenges and limits

- None of the 12 sampled training tasks overlapped the 78 repaired rubrics. The release was correctly swapped, but this smoke cannot measure the rewrite's effect. Earlier synthetic/reference controls tested partial-credit behavior separately.
- Seven of 12 groups had no within-group reward variation and supplied no learning signal. More batches alone do not establish that this will improve.
- Three training episodes exhausted their budgets. Training traces contained 20 action errors and 52 truncated tool outputs; 173 executed tools had no execution failures. Interface mistakes and missing context remain distinct from answer correctness.
- Strict scoring coverage was below 95% for all three evaluations. Baseline issues include invalid judge responses and reference ambiguity. Original errors and scores are preserved; evaluation rubrics were not tuned against these answers.
- Final checkpoint `ckpt-12b4df13c46546679d8e33894e4d2cf6` has normal 48-hour remote retention. No checkpoint qualified for extended best-checkpoint retention because evaluation coverage was insufficient. Downloaded manifests and logs are **not** a permanent weight export.
- A symmetric factual-coverage comparison of the 96 saved answers was prepared (94 judge cases, two empty-answer zeros), but **not dispatched**: another chat's SFT controller `sb-AgoieX6Q5qloDHtEzyFmFx` was active. The submission guard blocked it before paid work. Do not claim a factual before/after result. Reconcile the shared budget after that controller finishes before considering submission.

## Reproduction and logs

- [Exact baseline config](../../configs/experiments/grpo-v6-parallel/baseline-v1.json)
- [Exact smoke config](../../configs/experiments/grpo-v6-parallel/smoke-v1.json)
- [Procedure and frozen identities](PROCEDURE.md)
- [Baseline results](baseline/README.md), [grading issues](baseline/grading-issues.json)
- [Smoke results](smoke/RESULTS.md), [task audit](smoke/training-task-audit.json)
- [Baseline W&B](https://wandb.ai/nmdatar-harvard-university/repository-qa-training/runs/grpo-v6-baseline-seed42-eval-1fae5e0b)
- [Smoke W&B: training mean/EMA, updates and evaluation](https://wandb.ai/nmdatar-harvard-university/repository-qa-training/runs/grpo-v6-smoke-seed42-v1)
- [Campaign receipt](submission.json), [successful campaign status](campaign-status.json)
- [Deferred factual config](../../configs/experiments/grpo-v6-parallel/factual-v1.json), [dispatch status](factual/submission.json)

Raw trajectories, verifier reports and events are archived under `artifacts/grader-paraphrase-validation-results/artifacts/experiments/grpo-v6-baseline-seed42/` and `grpo-v6-smoke-seed42-v1/`. Smoke archive contains 877 files and 112 trajectories.

Joint campaign ledger reservations increased **$39.424369**, from $229.992348 to $269.416716 of the unchanged $550 allocation. These are reservations, not provider invoices. Subsequent SFT spending is outside this campaign; this closing balance is not a live budget reading. The project cap remains $1000. Per-arm deltas overlap concurrent spending and must not be added.
