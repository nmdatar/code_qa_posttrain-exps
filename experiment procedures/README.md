# Qwen3.5-4B experiment procedures

These are proposed, budget-gated experiment instructions, not scheduled jobs or claims of model improvement. Start with **Qwen/Qwen3.5-4B**, the exact catalog model used in the pilot; do not silently substitute a different model or a separately named pretrained `-Base` initialization.

The [live pilot report](../reports/qwen-repository-rl-pilot.md) records one acknowledged repository GRPO update, checkpoint reload, and a one-task score of 0.5 before and after. It proved integration, **not quality improvement**. Its one-hour diagnostic checkpoints are not durable retained winners; verify availability before referencing them. Start the controlled studies from the unchanged catalog model, not that pilot adapter.

## Run order and roadmap mapping

| Order | Procedure | Corresponding items in [experiments.md](../experiments.md) |
|---|---|---|
| 1 | [Baseline and throughput](01-baseline-and-throughput.md) | Starting-policy baseline (1), measurement before scaling (9) |
| 2 | [Direct GRPO](02-direct-grpo.md) | GRPO (3), moved ahead of SFT and REINFORCE |
| 3 | [Learning rate and group size](03-grpo-learning-rate-and-group-size.md) | Initial GRPO ablations (3) |
| 4 | [Optional SFT before GRPO](04-optional-sft-before-grpo.md) | Initialization (1), winning recipe with/without SFT (9) |
| 5 | [Quality-gated efficiency reward](05-quality-gated-efficiency-reward.md) | Efficiency reward (5) |
| 6 | [Frozen-policy tools and context](06-frozen-policy-tools-and-context.md) | Tools (6), retrieval/history (7), interactions/retraining (9) |

Defer REINFORCE (2), PPO and learned rewards (4), stronger-model annotations (8), clipping/reference-KL extensions (3), and 9B scaling (9). SFT is optional and must not block the first GRPO experiments. A skipped or blocked study is recorded, not treated as a negative result.

## Current capabilities versus prerequisites

| Capability | Current evidence / required work |
|---|---|
| Qwen sampling, direct GRPO, assistant-token attribution, loss reduction, checkpoints and checkpoint evaluation | Implemented in `training_pipeline`; a small live update is verified. |
| SFT, fork, optimizer-aware resume | Implemented; toy/backend checks exist. Collection inputs currently permit GRPO only, so repository SFT requires an eligible dataset adapter. |
| Public/private separation, bounded repository tools, immutable policy groups | Implemented for the experimental source-reading path. |
| Batch of 16 tasks | Configuration supports task batch size; this scale has not been benchmarked live. |
| 8–32 simultaneous episodes | Implemented with bounded episode and judge concurrency, serialized SDK submission/logging, shared reservations, and cleanup barriers. Offline tested; provider capacity and throughput remain unmeasured. See [concurrency guide](../docs/CONCURRENCY.md). |
| Frozen 32/85 evaluation cohorts | Implemented: `cohorts` writes a hashed family-stratified manifest; pin it in `evaluation`. Training uses selection; `baseline` and `evaluate` support explicit `--cohort`. See [readiness implementation](../docs/EXPERIMENT_READINESS.md). |
| Confidence intervals and reporting | Implemented: demonstrated quality over all assigned tasks, coverage/completion, and offline paired repository bootstrap with unresolved sensitivity. Statistical gates do not establish judge calibration or seed provenance. |
| Automatic best selection and durable archive/restore | **Prerequisite:** implement and verify. Saving expiring remote state is insufficient. |
| Reference-based reward | Implemented but uncalibrated. The completed pilot used a frozen Qwen judge. A separate Nemotron judge is now configurable via `../examples/training-repository-qwen-nemotron.json`; its grading quality remains uncalibrated. Freeze the judge and re-evaluate the base before any comparisons. |
| Accepted/partial/failed efficiency reward | Exists in `qa_eval`, but is not the collection scalar-reference reward. A compatible verifier/rubric bridge is required for procedure 5. |
| Symbol navigation, bounded retrieval, history compression | Treat as prerequisites until the training adapter exposes and tests them; their presence elsewhere does not prove training integration. |

Do not add unsupported fields to current JSON configs or invent CLI flags. Each missing capability needs a tracked implementation/verification result before dependent paid work. Automated admission is allowed; retain `human_reviewed: false` and never claim calibrated or human-approved quality without evidence. Freeze any stronger/independent evaluator change before starting an arm and re-evaluate all controls under it.

## Frozen data and evaluation protocol

Use the versioned experimental source-reading pool: **858 training tasks and 117 development tasks**, derived from the collection release. Pin the release hash, task IDs, repository commits, environment images, and source inventories. Preserve train/development repository-family separation. These include repurposed benchmark questions: this is not an untouched public benchmark or a final-test set.

Create two immutable development manifests before training:

1. Allocate 32 selection slots proportionally to each repository family's count among the 117 tasks. Take integer floors, then assign remaining slots by descending fractional remainder, with family-name order breaking ties.
2. Within each family, sort tasks by SHA-256 of `qwen4b-procedures-v1|family_id|task_id`; take the allocated prefix. The complement is the **85-task confirmation cohort**. Save IDs, counts, hashes, seed string, and selection script revision.
3. Verify disjoint IDs and no overlap with training families. Selection and confirmation are task-disjoint, not independent repository-family samples from each other; disclose this limitation.
4. Evaluate the unchanged base on both cohorts before training. Use selection only for tuning/checkpoint choice; keep confirmation results hidden until each study's finalists and selection rules are locked. Do not tune from confirmation failures. Reuse of confirmation across this research program is exploratory, not final-test validation.

Use temperature 0 for evaluation and temperature 1 for RL, with identical harness, grader, task budgets, and task order within comparisons. Initial episode defaults match the working compact-protocol pilot: 6 generations, 512 tokens per generation, 3,072 output tokens total, 8,192-token context, 5 tool calls, 3,500-byte observation limit, 300-second episode limit, and 120-second provider timeout. If procedure 1 shows these limits/protocol are unsuitable, fix and version them, rerun the baseline, then freeze the replacement before procedure 2. Do not quietly change them within an experiment.

For scalar-reference experiments, the primary endpoint is **demonstrated quality**: sum of resolved [0,1] quality scores divided by all assigned tasks. Unresolved tasks add no demonstrated points to this evaluation aggregate; they remain null/unresolved records and never become zero training rewards. Also report mean score among resolved tasks and scoring coverage. Report completion, invalid actions, citation failures, tokens, tool calls, latency, and all-attempt cost. Accepted-answer rate is reported only when a real verifier acceptance tier exists, not by inventing a threshold on reference scores.

Promotion requires at least 95% scoring coverage: 31/32 selection tasks and 81/85 confirmation tasks. Use the same matched tasks for candidate/control comparisons and show unresolved sensitivity explicitly. Rank eligible selection checkpoints by primary quality; differences under 0.01 are ties, broken by lower all-attempt evaluation cost, then fewer output tokens, then earlier update. Unknown billing cannot win a measured-cost tie; label estimates separately.

Run screening at seed 42. Repeat finalists with seeds 43 and 44 from the same initialization, each with its own deterministic training order. Select each seed's checkpoint using selection data only, then evaluate all three on confirmation. Compute paired candidate-minus-base differences, average across seeds per task, and bootstrap repository families with replacement 10,000 times (fixed bootstrap seed 20260928), retaining all tasks/seeds in a sampled family. Report 95% intervals, family count, and individual-seed results. “Winner” here means an experimental retained research candidate, not a human-calibrated production promotion; these documents do not change existing strict-verifier gates. A quality winner needs a positive lower confidence bound and no completion regression greater than 0.02; otherwise report an inconclusive or negative result. Small family counts and an uncalibrated judge limit interpretation.

## Checkpoint and recovery policy

Commit train-loadable state **including optimizer state**, immutable sampler weights, and framework state after every acknowledged GRPO update. Mark steps 5, 10, 15, 20, 25, 30 and the final committed step as evaluation checkpoints; skipped batches never advance the update counter. Keep routine remote checkpoints for **48 hours** (`checkpoint_ttl_seconds: 172800`) after current storage costs are included in the approved budget. Do not substitute the pilot's one-hour TTL.

Include model/rank, tokenizer/template/protocol, algorithm/reward/grader, dataset/cohort hashes, source revision, parent checkpoint, counters, deterministic data order/cursor, client RNG state, exact policy identity, and archive checksums in each manifest. Publish a committed manifest only after remote saves succeed. Record partial saves without selecting them.

Treat the unchanged base as the initial champion. Every new selection-best checkpoint, including a provisional best awaiting confirmation, must be archived **before** updating the `best` pointer. Archive actual sampler weights and complete resumable training state, plus configuration/framework state and provenance. Keep winners without automatic deletion in durable private project storage; retain the previous champion until restoration of its replacement passes. A signed download URL, remote URI, W&B reference, or JSON manifest alone is not an archive.

The installed SDK exposes checkpoint archive URLs and TTL management, but project-owned export/import and optimizer restoration are not yet verified. Test archive contents and independent restoration, including loading sampling weights and restoring optimizer/cursor state. If provider reimport cannot restore optimizer state, do not claim durable resume: retain verified non-expiring provider state as well, reserve ongoing cost, and mark archive restoration incomplete. Never publish private checkpoints merely to preserve them. W&B is an optional mirror, not the authoritative archive.

On unresolved grading, exclude the entire same-task group. Retry the entire group at most once for infrastructure failures, then quarantine. Equal-reward groups contribute zero. Skip the optimizer for an all-zero batch. For ambiguous optimizer outcomes, stop and restore the last committed checkpoint; never blindly repeat the call. Do not mix policy versions, reuse an updated-policy group, or relax grading to force learning signal.

## Budget, stopping, and commands

Every study needs a separately authorized total ceiling covering baseline and repeated evaluation, all arms/seeds, rollout/judge sampling, Modal time, updates, checkpoint storage/archives, retries, and cleanup. The prior **$5 smoke authorization is spent on that pilot and is not authorization for these studies**. Recheck prices and model capabilities at launch. Persist worst-case in-flight reservations before dispatch; unknown outcomes retain reservations. Distinguish measured bills, token-based estimates, and conservative reservations.

Training arms target at most 30 successful updates and stop after 60 attempted batches or the authorized ceiling, whichever comes first. If the first five attempted batches produce no contributing groups, stop for diagnosis rather than silently spending 60 batches. Stop immediately on ambiguous updates, data leakage, mixed-policy batches, or failed checkpoint commitment. If selection quality falls by at least 0.10 versus base at two consecutive scheduled checks, stop the arm and retain its earlier best. All incomplete runs remain in the report.

Current CLI operations below are real. Variables must point to newly prepared configs/manifests; these commands are **not** executed by creating these documents. `validate --remote` contacts the provider. `baseline`, `run`, `evaluate`, `fork`, and `resume` may incur charges. Cohort routing is implemented; concurrency and archiving remain prerequisites. See [new commands and settings](../docs/EXPERIMENT_READINESS.md).

```sh
# From the project root, after preparing and funding this experiment's configuration:
.venv-eval/bin/python -m training_pipeline validate --config "$CONFIG"
.venv-eval/bin/python -m training_pipeline validate --remote --config "$CONFIG"
.venv-eval/bin/python -m training_pipeline run --config "$CONFIG"
.venv-eval/bin/python -m training_pipeline evaluate --checkpoint "$CHECKPOINT" --output "$EVAL_OUTPUT"
.venv-eval/bin/python -m training_pipeline fork --config "$NEW_CONFIG" --checkpoint "$CHECKPOINT"
.venv-eval/bin/python -m training_pipeline resume --checkpoint "$CHECKPOINT"
```

Changing training semantics requires a new run/fork. A fork loads weights with a fresh optimizer; resume restores compatible optimizer/framework state. Save local events, raw rollouts and grading before optional W&B upload. See the [training documentation](../docs/TRAINING_PIPELINE.md), [current pilot config](../examples/training-repository-qwen-v3.json), and [reward implementation](../qa_eval/grading.py).


## Current project allocation

The user allocated up to $1,000 for the whole project, with experiment execution
still paused. Canonical configs and sub-budgets are in
[configs/experiments](../configs/experiments/README.md). Routine retention is now
48 hours, superseding the original 30-day proposal. The single live Nemotron
grading check is documented in [its report](../reports/nemotron-single-rollout-check.md).
