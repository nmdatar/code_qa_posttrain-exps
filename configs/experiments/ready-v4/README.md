# Budget-capped Experiment 1 and 2

These protocol-v4 configurations supersede the original protocol-v3 launch configurations. Existing historical run artifacts remain unchanged. Do not compare their scores as if the protocol and grader were identical.

The approved project ceiling is $1,000, including prior spending. `../project-budget-v4.json` assigns cumulative ledger caps of $300 baseline/throughput, $550 direct GRPO, and $80 lower-LR, plus $70 reserves. Ledger migration preserves every prior reservation. Estimates are conservative reservations, not invoices or provider-enforced account limits.

Experiment 1 includes selection (32), confirmation (85), unchanged-base selection repeat (32), and eight 64-episode throughput arms: concurrency 1/8/16/32 in ascending order, then 32/16/8/1. Run benchmarks sequentially. Keep the end-of-study selection repeat until training finishes. Use the same pinned cohorts, models and reward definition for comparisons.

Experiment 2 is a bounded seed-42 study: 14 attempted batches and at most 14 successful updates, 16 tasks × 4 responses per batch, LR 1e-5, rank 8, evaluation every five successful updates and at termination. Stop after five initial zero-signal batches or two quality regressions of at least 0.1. Additional seeds and learning-rate arms are not funded by this plan. This is a budget-reduced scope, not completion of the original 30-update / three-seed procedure.

The provisional launch setting is 32 rollout slots and four judge slots (up from the original eight rollout slots); formal throughput selection remains part of Experiment 1. Eight judge slots were slower and missed the coverage/completion gates in the readiness check.

Both experiments disable whole-group retries (`group_retries: 0`). Unresolved groups are excluded and counted; no resampling until a favorable reward occurs. Routine checkpoints retain 48 hours. Each selection-best sampler checkpoint is downloaded with checksums to the Modal volume and optimizer/sampler reload is acknowledged before its best pointer changes. Best remote references retain 14 days. Optimizer state cannot be downloaded through the current live Tinker archive endpoint and is retained remotely for 14 days. This bounded study does not satisfy the original procedure’s indefinite off-provider optimizer archive requirement. Raw archive re-import into Tinker is not supported by this integration; durable downloaded bytes do not promise restoration after the remote reference expires.

Execution is entirely in Modal with sampling and training on Tinker. The laptop packages inputs and reads results. Do not launch GRPO until Experiment 1 meets 95% scoring coverage and 90% completion; throughput eligibility additionally requires completion within two percentage points of concurrency 1 across both repeats. Pick the lowest eligible concurrency within 90% of the fastest eligible throughput.

In W&B, open a new run's **Tables → answers** (or search workspace panels for `answers`). Every row has submission, reward, grading_status, grading_reason, phase and optimizer_step. Unresolved rewards are missing values, not model failures scored zero. Use `evaluation/demonstrated_quality` and `evaluation/scoring_coverage` against `optimizer_step`; individual `reward` points are different questions/answers, so spikes are expected. Optimizer steps count successful updates only.

The Qwen 397B grader remains an uncalibrated reference grader shared by training and selection. Positive held-out results are provisional and do not establish independently verified correctness.

## Preparation and launch stages

Use `python -m training_pipeline.remote prepare --configs ... --budget-plan configs/experiments/project-budget-v4.json --output <fresh-bundle>` with the project environment. Submission is a separate `python -m training_pipeline.remote submit --bundle <fresh-bundle>` operation. Never reuse a submitted bundle or output run ID.

Prepare baseline/confirmation separately from throughput so the initial baseline runs first. Throughput config order must be c01-r1, c08-r1, c16-r1, c32-r1, c32-r2, c16-r2, c08-r2, c01-r2. Do not enable parallel training for measurement campaigns. Keep selection-repeat separate for the end of the study. Prepare direct GRPO only after reviewing the baseline and throughput gates, using the selected concurrency unchanged throughout training/evaluation. Refresh cloud ledger history locally before preparing later bundles; history divergence fails closed.

The remote worker verifies bundle hashes, permits only the pinned models/grader, checks aggregate remaining ledger capacity, reserves controller cost, and stops a benchmark campaign after a failed arm. Full-study launch remains a separate action from the readiness diagnostics.

The direct-GRPO allocation also reserves a final 85-task candidate confirmation using `02-direct-grpo-confirmation.json`. Its Modal worker resolves the locked `checkpoints/best.json` pointer to the committed manifest, then calls checkpoint evaluation; confirmation never updates the selection-best pointer. Prepare/submit this separate bundle only after training selects its finalist. The final training cap is 14 batches/updates so all readiness-check reservations and this confirmation fit within the $550 cumulative ledger.

The live archive/restore validation passed in `readiness-archive-restore-v4-r3`: a 72,980,480-byte sampler archive was checksummed on the durable Modal volume, and both optimizer and sampler loads were acknowledged using a fresh training client. Tinker rejected loading optimizer state into an already-used client, so recovery creates a fresh client first.
