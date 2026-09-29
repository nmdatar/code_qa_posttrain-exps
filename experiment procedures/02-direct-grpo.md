# 02 — Direct GRPO against unchanged Qwen3.5-4B

**Question:** Does verifier-reward GRPO improve held-out repository-answer quality without SFT?

**Hypothesis:** Comparing fresh attempts at the same question supplies a useful gradient and improves development quality, not merely training reward.

**Dependencies:** [Baseline and throughput](01-baseline-and-throughput.md); [roadmap item 3](../experiments.md). Use the frozen evaluator and validated concurrency from procedure 1.

## Arms and fixed settings

| Setting | Control | Training arm |
|---|---|---|
| Initialization | Unchanged Qwen/Qwen3.5-4B | Same unchanged model |
| SFT | None | None |
| LoRA | No learned adapter | Rank 8 |
| Optimizer | None | Adam, learning rate `1e-5` |
| Batch | Fixed evaluation cohorts | 16 different training tasks × four attempts = 64 episodes |
| Sampling | Temperature 0 evaluation | Temperature 1 training |
| Duration | Baseline plus end-of-study control repeat | At most 30 successful updates / 60 attempted batches |

Use population-standardized within-task advantages. Make one built-in `importance_sampling` forward/backward and optimizer pass per batch; no clipping, reference KL, critic, replay, or multiple epochs. Average generated-token contributions within each trajectory and then across contributing trajectories. All groups in a batch share one immutable policy; refresh sampling weights only after an acknowledged update.

## Prerequisites and procedure

1. Freeze configuration, eligible task order, evaluator identities, 32/85 cohorts, and price evidence. Use seed 42 for screening. Configure `batch_size: 16`, `group_size: 4`, `max_updates: 30`, `max_batches: 60`, `evaluation.every: 5`, `evaluation.max_tasks: 32`, and `checkpoint_every: 1` with the explicit selection-cohort manifest and hash described in [implementation details](../docs/EXPERIMENT_READINESS.md). Setting `max_tasks` alone does not create that cohort. Enable the documented `stopping` controls for the no-signal and regression rules.
2. Verify concurrency, best-checkpoint selection, and durable retention from the README. These are prerequisites, not features demonstrated by the one-update pilot. Do not scale a one-hour checkpoint configuration into a research run.
3. Shuffle training tasks deterministically without replacement. Finish each pass before reshuffling; avoid duplicate tasks within a batch at pass boundaries. Do not silently truncate the data to 480 tasks. Thirty contributing batches use 480 task slots, so this is **not** necessarily a full pass over 858 tasks; report actual unique coverage and repeats.
4. Collect and grade 64 complete episodes before constructing the batch. Exclude unresolved groups, log zero-variance groups, and apply one update only if contributions remain. Count exclusions in attempted-token and cost budgets.
5. Commit and evaluate at steps 5, 10, 15, 20, 25, 30 and the final committed step. Retain the selection-best model, not automatically the last model. Repeat the unchanged base on selection at study end to expose evaluator/runtime drift.
6. If the seed-42 best beats base selection quality by at least 0.01 with required coverage and no completion regression greater than 0.02, repeat the same recipe from base for seeds 43 and 44. Select each seed's checkpoint before exposing confirmation results. Otherwise report the negative/inconclusive screening result and diagnose reward variance, protocol completion, and sample size.

## Metrics, stopping, and promotion

Plot selection quality and scoring coverage versus successful update, generated tokens, and cumulative all-service cost. Report loss as an optimization diagnostic, alongside reward distributions, zero-variance/excluded-group fractions, valid action rate, answer completion, citation failures, and unique training-task coverage. A near-zero policy loss is not proof of no gradient or proof of improvement.

Apply the common 60-attempt, five-initial-zero-batch, budget, and regression stops. Never resample until a favorable group appears. Run all three locked finalists on the 85-task confirmation cohort and apply the paired repository-clustered comparison in the README. Promote only a confirmed quality winner; retain useful integration artifacts even if learning is inconclusive. Hand the recipe, checkpoint lineage, controls, and token/cost curves to [GRPO tuning](03-grpo-learning-rate-and-group-size.md).

## Checkpoints, safety, and required artifacts

Apply the [shared measurement, checkpoint, and budget contract](README.md). Keep Qwen/Qwen3.5-4B and the pinned 858/117 source-reading split; use the frozen 32-task selection and 85-task confirmation cohorts. Repurposed benchmark provenance and uncalibrated reference grading must remain visible. The one-update pilot demonstrated integration, not quality improvement.

Promotion requires at least **95% scoring coverage** (31/32 selection tasks and 81/85 confirmation tasks). Repeat training finalists with seeds 42, 43, and 44; select their checkpoints before inspecting confirmation results. Frozen-policy-only comparisons use the repeated-inference protocol stated above instead of claiming training-seed replication.

For any training arm, commit resumable state and sampler weights after every acknowledged GRPO update; evaluate every five successful updates and at the final committed checkpoint. Skips do not count as updates. Retain routine checkpoints for 48 hours and archive every new selection-best model durably, with checksums and verified sampling/optimizer restoration, before changing the best pointer. For frozen-weight studies, retain the source checkpoint plus the complete harness variant bundle instead of inventing an optimizer checkpoint.

Exclude unresolved groups; allow only one whole-group infrastructure retry before quarantine. Equal-reward groups contribute zero, and all-zero batches skip optimization. Stop on ambiguous optimizer outcomes and restore the last committed boundary without blindly retrying. Stop at the separately authorized total budget; the earlier $5 smoke ceiling does not authorize this experiment. Count unsuccessful attempts, graders, evaluation, storage, and retries in cost.

Persist a frozen experiment specification, resolved configurations, dataset/cohort hashes, model and grader identities, source revisions, all raw trajectories and grades, local events, cost ledger, per-task evaluation tables, checkpoint/archive manifests, and a decision report. W&B is optional. Mark unsupported capabilities as prerequisites; use only the [documented CLI](README.md#budget-stopping-and-commands), never invented flags. Report implementation readiness, live execution, and quality evidence separately.
