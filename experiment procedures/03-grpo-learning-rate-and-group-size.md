# 03 — GRPO learning rate, then group size

## Current runnable scope

Current-v8 has a fixed 2×2 rate/group-size comparison: 5e-6/1e-5 × four/eight attempts, two/one tasks per batch, 16 attempted batches, and the same maximum 393,216 rollout-output tokens per arm. No winning-rate placeholder remains. Use selection to compare fixed arms; do not claim the 1e-5 default is an established winner. Independent arms can overlap, subject to shared provider capacity.

See the [current suite](../configs/experiments/current-v8/README.md) for authoritative settings. The design below includes historical larger-scope extensions; it does not override current configs.

**Question:** Which small GRPO setting improves quality per sampled token and per dollar?

**Hypothesis:** Learning-rate choice and within-question sampling diversity affect useful updates; eight attempts may reduce zero-variance groups but also reduce question diversity at a fixed episode batch size.

**Dependencies:** [Direct GRPO](02-direct-grpo.md), [roadmap item 3](../experiments.md). Diagnose a completely signal-free baseline before tuning. The run may proceed after a functional but statistically inconclusive procedure 2; do not claim there is an established winner in that case.

## Arms and controls

Stage A compares learning rates with rank 8, 16 tasks × four attempts, and all other direct-GRPO settings fixed:

- A1: `5e-6`.
- A2: `1e-5` (reuse procedure 2 only if its full frozen specification and budget are identical).

Stage B holds the selection-winning learning rate fixed:

- B1: 16 tasks × four attempts, 64 episodes per batch.
- B2: eight tasks × eight attempts, 64 episodes per batch.

Every arm starts from the **unchanged** Qwen/Qwen3.5-4B model, not the final checkpoint of another arm. Use fresh optimizer state, seed 42 screening, temperature 1, identical task permutation prefixes, tools, grader and per-episode limits. The groups consume that order at different rates in stage B; report this deliberate diversity tradeoff.

## Procedure and prerequisites

1. Pre-register all arm IDs, initial policy identity, task orders, metrics, and selection rule. Fund both stages and confirmation repetitions separately from the smoke budget.
2. Run A1/A2 for at most 30 successful updates or 60 attempted batches each. Retain checkpoints/evaluation every five successful updates, using the common archive rules. Choose the learning rate using selection quality; apply the README cost/earlier-step tie-break. If neither beats the unchanged base by 0.01, label the rate provisional and do not claim improvement.
3. Run B1/B2 with that fixed rate under the same limits. Reuse B1 only when it is exactly the corresponding A arm; otherwise rerun it and explain the incompatibility.
4. Set the same hard rollout-output allowance for every arm: 3,840 episodes × the frozen per-episode output-token cap (60 attempted batches × 64 episodes). Reserve retry costs separately and count retry tokens within that allowance; a shared token-stop controller is a prerequisite because the current CLI has no such field. Stop before dispatch if a complete worst-case batch would exceed the remaining allowance.
5. Compare quality curves at common observed token budgets as well as at common optimizer steps. Use the last committed checkpoint at or below each common budget; do not interpolate a nonexistent checkpoint. Sixty-four episodes per batch is not a guarantee of equal actual tokens, wall time, or cost.
6. Lock one screening recipe using selection data, repeat it from unchanged base for seeds 43 and 44, and evaluate all seed-selected checkpoints on confirmation. Preserve the direct-GRPO control under the same evaluation protocol.

## Metrics, stopping, and promotion

Report primary quality, completion/scoring coverage, quality per million generated tokens, cost, unique task count, reward variance, group exclusion, and zero-variance rate. Explain whether eight attempts improves within-task signal at the expense of task diversity. Do not change clipping, KL, loss reduction, optimizer epochs, or renderer in this study.

Apply common budget, no-signal, regression, and ambiguous-update stops. Prefer the four-attempt recipe when quality ties and it has lower cost; otherwise use the shared tie-break. Confirm the selected recipe across three seeds before calling it a quality winner. A faster/higher-reward run with no confirmation quality gain is not a demonstrated improvement. Pass the complete initialization and recipe to [optional SFT](04-optional-sft-before-grpo.md) and [efficiency reward](05-quality-gated-efficiency-reward.md).

## Checkpoints, safety, and required artifacts

Apply the [shared measurement, checkpoint, and budget contract](README.md). Keep Qwen/Qwen3.5-4B and the pinned 858/117 source-reading split; use the frozen 32-task selection and 85-task confirmation cohorts. Repurposed benchmark provenance and uncalibrated reference grading must remain visible. The one-update pilot demonstrated integration, not quality improvement.

Promotion requires at least **95% scoring coverage** (31/32 selection tasks and 81/85 confirmation tasks). Repeat training finalists with seeds 42, 43, and 44; select their checkpoints before inspecting confirmation results. Frozen-policy-only comparisons use the repeated-inference protocol stated above instead of claiming training-seed replication.

For any training arm, commit resumable state and sampler weights after every acknowledged GRPO update; use the frozen arm’s evaluation cadence (every six successful updates and final evaluation for current GRPO). Skips do not count as updates. Retain routine checkpoints for 48 hours and retain selection-best remote state for 14 days and preserve sampler archives/checksums; indefinite durable optimizer restoration is not established. For frozen-weight studies, retain the source checkpoint plus the complete harness variant bundle instead of inventing an optimizer checkpoint.

Exclude unresolved groups; use zero whole-group retries in the current v7 campaign; quarantine unresolved groups. Equal-reward groups contribute zero, and all-zero batches skip optimization. Stop on ambiguous optimizer outcomes and restore the last committed boundary without blindly retrying. Stop at the separately authorized total budget; the earlier $5 smoke ceiling does not authorize this experiment. Count unsuccessful attempts, graders, evaluation, storage, and retries in cost.

Persist a frozen experiment specification, resolved configurations, dataset/cohort hashes, model and grader identities, source revisions, all raw trajectories and grades, local events, cost ledger, per-task evaluation tables, checkpoint/archive manifests, and a decision report. W&B is optional. Mark unsupported capabilities as prerequisites; use only the [documented CLI](README.md#budget-stopping-and-commands), never invented flags. Report implementation readiness, live execution, and quality evidence separately.
