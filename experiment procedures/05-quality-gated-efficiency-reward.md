# 05 — Quality-gated efficiency reward

## Current runnable scope

The tiered reward bridge is now implemented in the collection path. Current-v8/05-quality-only.json and 05-efficiency.json share the fixed current base/GRPO/data/evaluator setup and differ in their named training objective. The bridge consumes actual strict-verifier failed/partial/accepted tiers and trusted token/tool-time counters; it never thresholds positive factual coverage into acceptance. The Q/E formulas below are implemented. Software readiness does not establish broad live calibration of the model judge. This initial fixed-recipe comparison need not await a claimed winning optimizer.

See the [current suite](../configs/experiments/current-v8/README.md) for authoritative settings. The design below includes historical larger-scope extensions; it does not override current configs.

**Question:** Can RL reduce investigation cost while preserving grounded answer quality?

**Hypothesis:** A small efficiency bonus restricted to genuinely accepted answers can reduce unnecessary work without rewarding premature stopping.

**Dependencies:** [Tuned GRPO](03-grpo-learning-rate-and-group-size.md), optionally [SFT initialization](04-optional-sft-before-grpo.md); [roadmap item 5](../experiments.md); [reward implementation](../qa_eval/grading.py) and [reward contract](../requirements/AGENT_HARNESS.md).

## Prerequisites and exact reward arms

The current collection judge returns scalar reference-comparison scores. Those are **not** accepted/partial/failed tiers. This study requires actual evidence-aware acceptance decisions, supported-claim coverage, trusted input/output-token and tool-time measurements, and compatible private rubrics. Do not invent acceptance by thresholding the experimental score. If that verifier bridge or its eligible data is missing, record the study as blocked. Any narrower eligible cohort must be declared before results and used identically in both arms; it is a separate scoped study, not the full 858-task experiment.

Use the established tier definitions in both arms:

| Tier | Q: quality-only control | E: quality plus efficiency |
|---|---|---|
| Failed | 0 | 0 |
| Partial | `0.2 × supported_coverage` | Same |
| Accepted | 0.9 | `0.9 + 0.1 × efficiency` |
| Unresolved | No numerical reward | No numerical reward |

For this first comparison disable the latency term in both arms. Set compute units to `input_tokens + 2 × output_tokens + 100 × tool_seconds`, and `efficiency = clip(1 - compute_units / task_compute_budget, 0, 1)`. Pin the declared task budgets. This matches the existing compute-only formula; do not alter its coefficients after viewing results. Runtime/noise-sensitive reward variants are deferred.

## Procedure

1. Freeze the trained verifier/judge versions and establish a **new common quality baseline** for both arms. Scores from this tiered experiment must not be plotted as interchangeable with earlier scalar-reference scores.
2. Choose the initialization established by procedure 4 if that study has confirmed a benefit; otherwise use unchanged base. Both arms load the same weights and fresh optimizers. Never initialize one arm from the other arm's trained output.
3. Hold rank, learning rate, group size, task order, sampler temperature, concurrency and rollout budgets at the winning GRPO settings. Screen Q/E at seed 42 for at most 30 successful updates / 60 attempted batches, with the same rollout-output allowance.
4. Grade selection and confirmation with the **same quality-only evaluator**, ignoring shaped training reward. Commit each update, evaluate every five, and archive new selection-best models. Report the entire quality–cost tradeoff rather than only E's scalar training reward.
5. Inspect all newly failed/incomplete selection examples and false execution claims. Flag early stopping, omitted required claims, unexplained evidence gaps, and lower scoring coverage. Inspection may diagnose a future experiment but must not change this arm's rubric or score retroactively.
6. Repeat a promising comparison with seeds 43 and 44; lock seed-specific checkpoints before confirmation. Use paired repository-clustered intervals for quality and cost differences.

## Metrics, stopping, and promotion

Primary endpoint here is accepted-answer rate over **all assigned** evaluation tasks, with partial-claim coverage and unresolved counts reported separately. Require the shared 95% scoring coverage gate. Report all-attempt dollars, input/output tokens, tool calls, tool seconds, and latency per accepted answer; if no answers are accepted, cost per acceptance is undefined, never zero. Include paid judges, failed rollouts, and infrastructure retries. Unknown actual billing permits only explicitly estimated cost claims.

An efficiency winner must have a lower 95% confidence bound for acceptance change of at least -0.02 and an upper 95% bound for relative all-attempt cost-per-accepted-answer change below -0.10. Reject lower cost caused by missing grades or reduced completion. If intervals are too wide, retain quality-only training and report the study as inconclusive. Preserve failed/partial/accepted ordering, exclude unresolved groups, and apply the common training stops. Pass a confirmed quality–cost winner to [frozen-policy harness comparisons](06-frozen-policy-tools-and-context.md).

## Checkpoints, safety, and required artifacts

Apply the [shared measurement, checkpoint, and budget contract](README.md). Keep Qwen/Qwen3.5-4B and the pinned 858/117 source-reading split; use the frozen 32-task selection and 85-task confirmation cohorts. Repurposed benchmark provenance and uncalibrated reference grading must remain visible. The one-update pilot demonstrated integration, not quality improvement.

Promotion requires at least **95% scoring coverage** (31/32 selection tasks and 81/85 confirmation tasks). Repeat training finalists with seeds 42, 43, and 44; select their checkpoints before inspecting confirmation results. Frozen-policy-only comparisons use the repeated-inference protocol stated above instead of claiming training-seed replication.

For any training arm, commit resumable state and sampler weights after every acknowledged GRPO update; use the frozen arm’s evaluation cadence (every six successful updates and final evaluation for current GRPO). Skips do not count as updates. Retain routine checkpoints for 48 hours and retain selection-best remote state for 14 days and preserve sampler archives/checksums; indefinite durable optimizer restoration is not established. For frozen-weight studies, retain the source checkpoint plus the complete harness variant bundle instead of inventing an optimizer checkpoint.

Exclude unresolved groups; use zero whole-group retries in the current v7 campaign; quarantine unresolved groups. Equal-reward groups contribute zero, and all-zero batches skip optimization. Stop on ambiguous optimizer outcomes and restore the last committed boundary without blindly retrying. Stop at the separately authorized total budget; the earlier $5 smoke ceiling does not authorize this experiment. Count unsuccessful attempts, graders, evaluation, storage, and retries in cost.

Persist a frozen experiment specification, resolved configurations, dataset/cohort hashes, model and grader identities, source revisions, all raw trajectories and grades, local events, cost ledger, per-task evaluation tables, checkpoint/archive manifests, and a decision report. W&B is optional. Mark unsupported capabilities as prerequisites; use only the [documented CLI](README.md#budget-stopping-and-commands), never invented flags. Report implementation readiness, live execution, and quality evidence separately.
