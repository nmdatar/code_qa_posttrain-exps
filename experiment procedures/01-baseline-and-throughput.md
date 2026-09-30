# 01 — Baseline quality and rollout throughput

## Current runnable scope

Current-v8 provides fresh baseline/selection-repeat/confirmation configs and eight throughput arms, all bound to training-claims-v6 and the current strict v7 judge/context policies. The throughput manifest is regenerated against the new data identity. Original throughput results remain incomplete; these are prepared measurements. Keep timing arms isolated.

See the [current suite](../configs/experiments/current-v8/README.md) for authoritative settings. The design below includes historical larger-scope extensions; it does not override current configs.

**Question:** Can the unchanged Qwen3.5-4B policy produce reliably scored investigations, and what concurrency provides useful throughput without degrading them?

**Hypothesis:** Bounded parallel episodes reduce wall time while preserving isolation, completion, and scoring coverage. This experiment makes no learning claim and performs no optimizer updates.

**Dependencies:** [Roadmap items 1 and 9](../experiments.md); [measurement contract](README.md). Complete this before [direct GRPO](02-direct-grpo.md).

## Arms, controls, and prerequisites

- Freeze unchanged Qwen/Qwen3.5-4B weights, compact action protocol, tools, renderer, grader, pinned source images, and the initial episode limits in the README. Do not use a learned pilot checkpoint.
- Baseline evaluation: temperature 0 on all 32 selection and 85 confirmation tasks. Repeat the selection evaluation once to estimate unchanged-model variability; retain both measurements.
- Throughput arms: concurrency **1, 8, 16, 32**, each collecting four attempts for the same 16 training tasks at temperature 1, without updates. Pick tasks by the same deterministic family-stratified method as the cohorts, using seed string `throughput-v1` and 16 slots. Use one task group per selected task, 64 episodes per arm.
- Run two measured repetitions per concurrency: first in ascending concurrency, then descending. Report cold provisioning and the second pass separately; identical sampling seeds are not a guarantee of identical outputs under provider scheduling.
- Prerequisites: bounded global concurrency, SDK concurrency safety, semaphore-limited sandbox creation, thread-safe logging, and atomic shared reservations. The `concurrency` JSON setting is implemented and offline-tested; provider capacity still requires measurement. Explicit 32/85 cohort routing and baseline evaluation without trainer allocation are now implemented and offline-tested; live verification remains pending. See [implementation details](../docs/EXPERIMENT_READINESS.md).

## Procedure

1. Validate release integrity, public/private separation, capabilities, prices, and cohort manifests. Record exclusions before assigning the benchmark cohort; never replace an inconvenient task after seeing its score.
2. Obtain a budget for 117 baseline episodes, 32 repeat evaluations, and 512 throughput episodes, plus bounded retries and storage. If unaffordable, create a separately labeled reduced diagnostic instead of calling it this completed study.
3. Evaluate and freeze the base-control tables. If scoring coverage is below 95%, resolve infrastructure/grader problems and repeat the baseline under a newly versioned configuration before training comparisons.
4. Execute the eight throughput runs with no updates and one isolated environment per episode. Bound concurrent judge requests at four for this benchmark and use the same limit in every arm; record grader queuing separately.
5. Measure full-coordinator wall time, generation time, provisioning, tool time, grading, throughput, and actual/reserved cost. Start the timer before the first dispatch and stop after the last grade and cleanup acknowledgment.
6. Select the lowest concurrency achieving at least 90% of the fastest eligible measured throughput. Eligibility requires at least 95% scoring coverage and completion no more than 0.02 below the concurrency-1 arm, aggregated over both repetitions. If no higher-concurrency arm qualifies, retain concurrency 1 and diagnose before scaling.

## Metrics and stop rules

Report completed and resolved episodes/minute, generated tokens/second, p50/p95 episode latency, startup latency, queue time, tool/provider errors, scoring coverage, invalid actions, and all-attempt cost per completed/scored episode. Include quality distributions to detect an apparent speedup caused by early failures; do not assert noninferiority from this small sample alone.

Pause escalation when the next concurrency produces lower throughput in both repetitions or scoring coverage falls below 95%. Never increase concurrency to compensate for a parser bug, equal rewards, or context overflow. Do not alter prompt or budgets between arms; any repair restarts the comparison with a new protocol identity.

## Decision and handoff

Publish the chosen concurrency, frozen baseline tables, per-stage timing breakdown, and an estimated 16-task × four-attempt update cost/time including evaluation and storage. Label extrapolations, including full-dataset duration, as estimates. Advance to GRPO only with at least 95% baseline scoring coverage, at least 90% baseline answer completion, tested accounting/recovery, and a separately approved training budget. A failed readiness gate means repair the protocol or limits and rerun this procedure; it does not imply SFT is mandatory.

## Checkpoints, safety, and required artifacts

Apply the [shared measurement, checkpoint, and budget contract](README.md). Keep Qwen/Qwen3.5-4B and the pinned 858/117 source-reading split; use the frozen 32-task selection and 85-task confirmation cohorts. Repurposed benchmark provenance and uncalibrated reference grading must remain visible. The one-update pilot demonstrated integration, not quality improvement.

Promotion requires at least **95% scoring coverage** (31/32 selection tasks and 81/85 confirmation tasks). Repeat training finalists with seeds 42, 43, and 44; select their checkpoints before inspecting confirmation results. Frozen-policy-only comparisons use the repeated-inference protocol stated above instead of claiming training-seed replication.

For any training arm, commit resumable state and sampler weights after every acknowledged GRPO update; use the frozen arm’s evaluation cadence (every six successful updates and final evaluation for current GRPO). Skips do not count as updates. Retain routine checkpoints for 48 hours and retain selection-best remote state for 14 days and preserve sampler archives/checksums; indefinite durable optimizer restoration is not established. For frozen-weight studies, retain the source checkpoint plus the complete harness variant bundle instead of inventing an optimizer checkpoint.

Exclude unresolved groups; use zero whole-group retries in the current v7 campaign; quarantine unresolved groups. Equal-reward groups contribute zero, and all-zero batches skip optimization. Stop on ambiguous optimizer outcomes and restore the last committed boundary without blindly retrying. Stop at the separately authorized total budget; the earlier $5 smoke ceiling does not authorize this experiment. Count unsuccessful attempts, graders, evaluation, storage, and retries in cost.

Persist a frozen experiment specification, resolved configurations, dataset/cohort hashes, model and grader identities, source revisions, all raw trajectories and grades, local events, cost ledger, per-task evaluation tables, checkpoint/archive manifests, and a decision report. W&B is optional. Mark unsupported capabilities as prerequisites; use only the [documented CLI](README.md#budget-stopping-and-commands), never invented flags. Report implementation readiness, live execution, and quality evidence separately.


Implemented benchmark configs and exact commands are in the [concurrency guide](../docs/CONCURRENCY.md). They pin 16 training tasks, four attempts, and two repetitions at each worker count. No paid concurrent benchmark has run yet.
