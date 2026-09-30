# 04 — Optional SFT warm start versus direct GRPO

## Current runnable scope

The collection SFT adapter now supports verified full investigations as well as the separate existing tool-prefix release. Full-investigation admission binds current rubric/judge identity, clean successful tool history, source-backed citations, train lineage and native token alignment. Only one full investigation qualifies in the archived data, so current-v8/04-investigation-sft.json is a smoke test until more current trajectories are collected. The 30-example tool-only pilot is completed and did not improve end-to-end quality. The new 04-sft-then-grpo config uses an implemented remote fork from the selection-best SFT checkpoint with fresh optimizer; no expired or unselected checkpoint is substituted.

See the [current suite](../configs/experiments/current-v8/README.md) for authoritative settings. The design below includes historical larger-scope extensions; it does not override current configs.

**Question:** Does supervised investigation training improve final quality or reduce the RL sample cost enough to justify collecting and training on teacher trajectories?

**Hypothesis:** SFT may improve reliable tool use and grounded answers for a small model, but may add cost or constrain exploration. Its benefit must be measured rather than assumed.

**Dependencies:** [Direct GRPO](02-direct-grpo.md) and preferably [the tuned recipe](03-grpo-learning-rate-and-group-size.md); [roadmap items 1 and 9](../experiments.md). This procedure is optional. Missing SFT data must not block direct GRPO.

## Arms and prerequisites

| Arm | Initialization and training |
|---|---|
| A | Unchanged Qwen/Qwen3.5-4B, no updates |
| B | Base → SFT only |
| C | Base → winning direct GRPO recipe |
| D | Base → the exact B checkpoint → same GRPO recipe, fresh optimizer |

Require manifest-backed, verified investigations for training-split tasks, with public solver inputs, actual tool observations, provenance, accepted targets, and no development questions or private reference leakage. Private references may verify a trajectory; they must not be presented as if a teacher discovered them blindly. Preserve machine-review versus human-review labels. If eligible examples are missing, publish a blocked/data-needed report; do not manufacture examples or perform collection as an unbudgeted side effect.

The collection adapter, manifest-backed admission, native renderer alignment and assistant-only masks are implemented. Broad full-investigation data collection remains necessary; the current one-example release only supports a smoke test. Do not bypass strict human-gold requirements by falsifying metadata. Use an explicitly labeled experimental admission route if required and verified.

## Procedure

1. Freeze the eligible SFT release and teacher model/version. Use all admitted trajectories from the 858-task training pool, deduplicated by lineage; record the resulting count before allocation. Freeze train/example order at seed 42 and preserve repository-family evaluation separation.
2. Train B with rank-8 LoRA, learning rate `1e-4`, batch size eight examples, exactly one shuffled pass. Use a smaller final batch rather than repeating examples to fill it; that final-batch behavior is now implemented and offline-tested in the generic training pipeline; the collection SFT adapter is implemented; the full-answer data supply remains limited. Number of updates is `ceil(admitted_examples / 8)`.
3. Train only selected assistant outputs, including authorized investigation actions; zero loss on prompts and tool observations. Validate next-token alignment and context limits before allocation. Reject ineligible/overlength examples at release validation and report their counts; do not silently truncate targets.
4. Commit SFT recovery state each update and evaluate every five updates and at the end. Select B's best checkpoint using only the 32-task selection cohort. Use that exact checkpoint for D; load weights with a fresh optimizer rather than carrying Adam state into GRPO.
5. Run C and D with identical winning GRPO hyperparameters, task order, rollout limits, and at most 30 successful updates / 60 attempted batches. Give both arms the same RL rollout-output allowance. Reuse C only if every comparison setting matches a prior recorded run.
6. Report both RL-only learning curves and total-cost curves including teacher collection, verification, SFT, grading, and checkpoint storage. An SFT warm start is not free. Screen at seed 42; if promising, repeat the entire applicable training sequence for seeds 43 and 44 using the frozen SFT examples, then perform confirmation.

## Metrics, stopping, and promotion

Compare A→B, A→C, and C→D. Measure action validity, completed/cited answers, quality, invalid schema rates, generated tokens to reach a common selection score, and total cost. Preserve the SFT-only arm so a gain is not wrongly attributed to RL.

Use common operational/no-signal/regression stops for GRPO; stop SFT on malformed alignment, leakage, ambiguous updates, or budget exhaustion. Promote SFT+GRPO only if it passes confirmation quality criteria against direct GRPO, or if quality is noninferior within 0.02 with a positive confirmed cost saving. Otherwise retain the simpler direct-GRPO recipe. A useful SFT-only model may be retained as a separate result without claiming RL helped. Pass any confirmed initialization decision to [efficiency reward](05-quality-gated-efficiency-reward.md).

## Checkpoints, safety, and required artifacts

Apply the [shared measurement, checkpoint, and budget contract](README.md). Keep Qwen/Qwen3.5-4B and the pinned 858/117 source-reading split; use the frozen 32-task selection and 85-task confirmation cohorts. Repurposed benchmark provenance and uncalibrated reference grading must remain visible. The one-update pilot demonstrated integration, not quality improvement.

Promotion requires at least **95% scoring coverage** (31/32 selection tasks and 81/85 confirmation tasks). Repeat training finalists with seeds 42, 43, and 44; select their checkpoints before inspecting confirmation results. Frozen-policy-only comparisons use the repeated-inference protocol stated above instead of claiming training-seed replication.

For any training arm, commit resumable state and sampler weights after every acknowledged GRPO update; use the frozen arm’s evaluation cadence (every six successful updates and final evaluation for current GRPO). Skips do not count as updates. Retain routine checkpoints for 48 hours and retain selection-best remote state for 14 days and preserve sampler archives/checksums; indefinite durable optimizer restoration is not established. For frozen-weight studies, retain the source checkpoint plus the complete harness variant bundle instead of inventing an optimizer checkpoint.

Exclude unresolved groups; use zero whole-group retries in the current v7 campaign; quarantine unresolved groups. Equal-reward groups contribute zero, and all-zero batches skip optimization. Stop on ambiguous optimizer outcomes and restore the last committed boundary without blindly retrying. Stop at the separately authorized total budget; the earlier $5 smoke ceiling does not authorize this experiment. Count unsuccessful attempts, graders, evaluation, storage, and retries in cost.

Persist a frozen experiment specification, resolved configurations, dataset/cohort hashes, model and grader identities, source revisions, all raw trajectories and grades, local events, cost ledger, per-task evaluation tables, checkpoint/archive manifests, and a decision report. W&B is optional. Mark unsupported capabilities as prerequisites; use only the [documented CLI](README.md#budget-stopping-and-commands), never invented flags. Report implementation readiness, live execution, and quality evidence separately.
