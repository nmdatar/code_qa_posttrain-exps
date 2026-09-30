# Experiment designs — campaign v8

See [experiment classification](../docs/EXPERIMENT_CLASSIFICATION.md) for all designed model, harness, reward, data/evaluator and infrastructure studies, including deferred proposals. New [procedure 07: question decomposition](07-question-decomposition.md) and [procedure 08: matched investigation distillation](08-investigation-distillation.md) have separate [research extension configs](../configs/experiments/research-extensions-v1/README.md). They do not replace campaign v8; distillation remains one-example smoke-only.

Start with the [research progression](../experiments/README.md) for completed studies and results. These procedures describe versioned designs, not a universal latest campaign.

Use [the campaign-v8 suite](../configs/experiments/current-v8/README.md) and its [study plan](../configs/experiments/current-v8/study-plan.json) for new work. Campaign v8 pins training-claims-v6, all-claims-v7, definition-context-v1, action-alias-v1 and paginate-v1. Earlier atomic-claims-v4/current-v7 settings remain frozen historical inputs.

| Procedure | Current configuration and scope |
|---|---|
| [1: baseline / throughput](01-baseline-and-throughput.md) | Fresh controls and repeated 1/8/16/32-worker throughput configs on the current data. |
| [2: direct GRPO](02-direct-grpo.md) | 16 bounded batches, 2 tasks × 4 attempts, LR 1e-5, initial/final strict selection evaluation. |
| [3: LR / group size](03-grpo-learning-rate-and-group-size.md) | Fixed 2×2 of rates 5e-6/1e-5 and groups 4/8; eight episodes per batch. |
| [4: SFT](04-optional-sft-before-grpo.md) | Full-investigation adapter and remote fresh-optimizer fork are implemented. Current matched full-answer release has one example: smoke-only until further data collection. Existing tool-only SFT is a distinct 30-example completed pilot. |
| [5: efficiency](05-quality-gated-efficiency-reward.md) | Explicit verifier-tier quality-only/efficiency objectives, trusted token/tool counters, matched strict evaluation. |
| [6: tools / context](06-frozen-policy-tools-and-context.md) | Frozen-base Python AST tools, lexical/subword-hybrid retrieval and archived-observation history, two inference repeats each. |

The current suite README is authoritative for exact identities, limits, rewards, data sufficiency, proposed budgets and execution. All run IDs and ledgers are versioned; existing jobs and historical artifacts are unchanged. All 43 configurations validate offline; this is not evidence of live improvement or judge calibration. Preparing configs/bundles does not launch experiments.

## Evaluation and recovery contract

The source-reading pool has 858 training and 117 development tasks. Frozen selection/confirmation cohorts have 32/85 tasks with repository families separated from training, but shared between those evaluation cohorts. Repurposed benchmarks are not an untouched final test. Tune on selection only and lock finalists before confirmation. Require 95% scoring coverage for promotion and best-checkpoint eligibility; retain unresolved sensitivity rather than converting missing training grades into zero.

Routine remote checkpoint retention is 48 hours, best retention 14 days, with sampler archives and provenance. Indefinitely durable optimizer export/import is not established. Current GRPO uses zero whole-group retries, skips equal-reward groups and all-zero updates, and stops on hard budget/batch/runtime limits or integrity/ambiguous-update failures. The current matched screen does not claim a three-seed result. Additional seeds and larger horizons must be explicit matched arms.

## Budget, stopping, and commands

Use the explicit current-v8 budget plan rather than obsolete per-arm caps. It accounts for old allocations separately and does not reset historical spending. The source-code budget check now validates the supplied total ceiling instead of hardcoding $1,000. Price snapshots and reservations must still be reconciled before dispatch. See [remote execution](../docs/REMOTE_EXPERIMENTS.md).

Older full-study designs in individual procedures remain research extensions; current-scope sections and current-v8 configs control actual runnable settings. Dependencies on a yet-unknown winning recipe have been replaced by a preregistered matched fixed-recipe screen where possible, without asserting that recipe has won.
