# Frozen v6 GRPO smoke

Config: `configs/experiments/grpo-v6-parallel/smoke-v1.json`.
Run: `grpo-v6-smoke-seed42-v1`.

This tests whether the corrected rubric/environment yields usable training groups and real optimizer updates. Six batches are a smoke test, not sufficient evidence of generalization or a full epoch.

- Data: immutable `repo-qa-training-claims-v6`; training-only rubric repairs, unchanged selection and confirmation membership.
- Policy: Qwen3.5-4B, rank8 LoRA, seed42, learning rate1e-5.
- Grader: Qwen3.5-397B-A17B, strict all-claims-v7 plus positive-coverage-v4 training reward; definition-context-v1.
- Environment: paginate-v1 and action-alias-v1; original episode/token limits retained.
- Up to6 attempted batches,2 tasks each,4 attempts per task:48 training trajectories and at most6 optimizer updates. Stop if the first4 batches yield no updates.
- Save initial checkpoint and checkpoint after each acknowledged optimizer update. Evaluate32 selection tasks at initial/final boundaries; `evaluation.every=0` disables intermediate scheduled checks only. Retain best eligible checkpoint using existing coverage requirements.
- Fresh baseline is a separate sampling-only arm in the same controller. Confirmation tasks are not evaluated.

One Modal campaign runs baseline and smoke as parallel subprocesses. This is necessary because ledger locks are local to a controller; separate simultaneous controllers would not safely share the ledger. Both arms retain the $550 GRPO ledger and $1000 project cap; authoritative ledger must be reconciled before parent dispatch. No optimizer call is replayed after uncertain acknowledgment.

Report all attempted batches, mean/EMA reward, contributing groups, actual updates, checkpoint identifiers, strict scoring coverage and initial/final scores. Strict evaluation and training factual reward are different measurements. If factual scoring is not emitted by evaluation, it must be explicitly reported as unavailable unless a separately bounded diagnostic grades saved answers. Do not infer factual improvement from strict pass rates or from a noisy six-batch reward curve.

The conservative smoke estimate is $263.24535146 including controller reservation. This intentionally overbounds grading, checkpoint storage and an extra terminal evaluation; it is not a provider invoice or expected bill. Combined campaign preparation and runtime spending checks enforce existing caps.

Cost attribution caveat: evaluation `reserved_cost_usd` takes a before/after difference of the shared ledger. During parallel execution this can include the other arm's reservations. Do not sum arm evaluation deltas or present them as isolated arm costs. Use the authoritative campaign ledger delta for total campaign reservations.
