# Main budget-bounded GRPO experiment — launched

This is an actual training experiment intended to measure held-out improvement, following
the machinery pilot. It is not a full epoch or a claim of training to convergence.

## Fix

For v7 shaped training, malformed citation paths, hashes, ranges, and oversized spans no
longer bypass factual grading. Only verified citations are passed onward; the original
submission remains in telemetry. Citation defects still force strict score zero and appear
in diagnostics. Missing/empty answers still get zero; wrong factual answers still get zero.
Evaluation routing is unchanged. The reward identity now records the citation-routing fix.
Historical frozen bundles and results remain unchanged.

## Proposed run

- Start from Qwen3.5-4B base weights, rank 8, seed 42, learning rate 1e-5.
- Frozen Qwen3.5-397B judge, atomic-claims-v4 data, v7 independent factual coverage.
- 12 attempted batches maximum; two tasks × four attempts per task each batch.
- 96 training attempts over 24 distinct training tasks, at most 12 optimizer updates.
  Groups with identical rewards cannot train, so actual updates can be fewer.
- This covers about 2.8% of the 858-task training set, not a full epoch.
- Strict evaluation on the same 32 selection tasks initially, every six successful
  updates, and at completion. Early completion still receives final evaluation.
- Retain the best coverage-eligible selection checkpoint for 14 days. No confirmation
  tasks are used for optimization or checkpoint selection.
- Stop after four initial batches with no updates; existing regression stopping uses
  a 0.10 demonstrated-quality drop on two qualifying checks. Maximum 12 batches and
  four-hour controller timeout remain hard bounds even if later signal is sparse.
- Keep the same reward curves, coverage metrics, excluded-group counts, and local CSV.

Starting from base weights gives this corrected reward version a clean initial evaluation;
we do not resume the two-update pilot trained under the defective gate. No separate broad
judge calibration is needed. Strict evaluation semantics remain the same, while training
reward comparisons across versions must note the routing change.

## Budget

The reconciled GRPO ledger has $58.243593 reserved from previous work. The new conservative
model/sandbox/storage estimate is $477.455189 plus $3.038976 controller reservation:
$480.494165 new, $538.737758 combined, within the existing $550 allocation. The estimate
includes worst-case repairs, evaluation bounds and best-checkpoint storage. It is not an
expected invoice. No spending cap was raised. Submission rechecks authoritative remote state.

A full epoch would need roughly 429 batches / 3432 attempts at this batch/group size,
before evaluations; it does not fit this prepared run's budget envelope.

## Verification

Full suite passed: Ran 514 tests in 43.169s. Citation routing tests passed, and the prepared bundle matches the fixed code.

## Artifacts

- [Run config](../../configs/experiments/grpo-main-v7/run.json)
- [Cost estimate](estimate.json)
- [Prepared bundle](bundle.json)
- [Citation routing regression tests](citation-tests.log)
- [Full tests](tests.log)

The user authorized launch, and the run was submitted successfully.

[Live W&B results](https://wandb.ai/nmdatar-harvard-university/repository-qa-training/runs/02-grpo-main-qwen4b-v7-citationfix-seed42). See [submission receipt](submission.json).
Rewards, EMA, grading coverage and evaluation scores stream to W&B. Remote durable logs
are under `artifacts/experiments/02-grpo-main-qwen4b-v7-citationfix-seed42/` on
`repository-qa-training-state-v2`, including `reward-curve.csv`, `events.jsonl`,
`answers.json`, evaluations, trajectories and checkpoints.
