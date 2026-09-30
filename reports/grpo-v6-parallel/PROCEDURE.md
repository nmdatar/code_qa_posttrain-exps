# Frozen v6 baseline and parallel GRPO smoke

User requested a fresh baseline under the repaired rubric and a subagent-owned GRPO smoke in parallel. Both arms use frozen configs, one immutable source bundle and one Modal controller with shared process-safe budget locks. Only the smoke optimizes weights. The subagent owns smoke monitoring and results; the parent owns baseline and joint dispatch.

## Configs

- [Baseline](../../configs/experiments/grpo-v6-parallel/baseline-v1.json): fresh Qwen3.5-4B, selection32, temperature0, no optimizer updates.
- [GRPO smoke](../../configs/experiments/grpo-v6-parallel/smoke-v1.json): same base model, seed42, rank8 LoRA, LR1e-5, six attempted batches maximum,2tasks×4rollouts, temperature1; at most48trainingattempts/6updates. Initial/finalselection32; checkpoint after every acknowledged update. Early stop if first4batches yield no updates.
- Shared immutable release: repo-qa-training-claims-v6, SHA256 dc5f67caa93ad0c3ca3d19e4fc4d0c18eae1e4d110e9d68d29b6a1a355310bda.
- Same judge Qwen3.5-397B-A17B; all-claims-v7, positive-coverage-v4 training reward; definition-context-v1, action-alias-v1, paginate-v1. Judge max8192 output/context65536; one schema repair. Policy context8192, max6generations/5toolcalls/512tokenspercall, max3072outputtokens,3500bytetoolobservations.
- Cohorts: fixes-v1-cohorts.json, hash045f39a6a05d72be45f5c42dc9fa12e3fdc2838a1fa92da436ea3f478ad6382e. Selection and confirmation membership unchanged; confirmation never evaluated here.

The repaired claims affect training tasks. Held-out task rubrics are deliberately unchanged. The fresh baseline matches the new environment/evidence policy and data identity; it is not evidence that evaluation task content was rewritten.

## Dispatch and spending

```sh
.venv-eval/bin/python -m training_pipeline.remote prepare \
  --configs configs/experiments/grpo-v6-parallel/baseline-v1.json configs/experiments/grpo-v6-parallel/smoke-v1.json \
  --output artifacts/grpo-v6-parallel-v1-bundle --parallel-training \
  --budget-plan configs/experiments/project-budget-v4.json
.venv-eval/bin/python -m training_pipeline.remote submit --bundle artifacts/grpo-v6-parallel-v1-bundle
```

These commands describe this dispatch, not a retry recipe. Read the receipt before any action; never resubmit an attempted bundle or replay uncertain optimizer calls. Both processes run in one named controller (1800-second timeout,2CPU,4GiB). Do not launch separate controllers sharing the ledger.

Before launch: authoritative GRPO reservations229.9923477857134/$550. Baseline upper52.31278464 plus smoke263.24535146 gives315.5581361 new conservative reservations, combined545.5504838857134. Both arms debit the existing GRPO experiment ledger; the $1000 project cap and allocations stay unchanged. These are conservative bounds, not predicted invoices; hard runtime checks remain active.

## Measurements

Baseline: attempted/resolved counts, strict scores, terminations, W&B and artifact links. Smoke: all attempted batches, mean/EMA training reward, contributing groups, actual updates, checkpoints and strict initial/final evaluation. Six batches tests mechanics and reward signal; it does not establish a stable learning trend.

After both arms complete, use their saved answers for a separately bounded factual-coverage diagnostic if the reconciled budget allows. Score baseline and smoke checkpoints under identical frozen conditions, preserve strict scores, report all missing grades. Do not infer factual reward from strict pass rate. The smoke's own initial evaluation is another untrained sample and helps expose run-to-run variability.

No tasks from this selection evaluation may become training data. Do not tune the rubric to make saved evaluation answers pass. Retain all original outputs, including failures. These conditions require a fresh baseline; old reward curves must not be spliced into this run.

Logs: `reports/grpo-v6-parallel/`, W&B project repository-qa-training, and Modal volume repository-qa-training-state-v2 under artifacts/experiments/<run_id>/ and campaigns/<bundle_id>/.

Parallel accounting note: per-arm evaluation reservation deltas read a shared counter and can include concurrent spending from the other arm. Use the final authoritative campaign ledger delta; do not sum those arm counters as separate costs.

Completion: both arms exited successfully; see README.md. The saved-answer factual comparison is prepared but not dispatched because a separate SFT controller is active. Before any future diagnostic launch, reconcile the latest authoritative budget and inspect active controllers. No factual comparison result is claimed.
