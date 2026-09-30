# Matched efficiency-reward RL pilot

[W&B aggregate comparison](https://wandb.ai/nmdatar-harvard-university/repository-qa-training/runs/efficiency-rl-v1-comparison). Both training runs completed; the comparison contains five training charts and four held-out charts.

User requested a live reward-shaping experiment on 2026-09-29. This is a fresh, single-seed exploratory comparison, not a claim of improvement.

## Frozen design

Two fresh Qwen/Qwen3.5-4B rank-8 trainers, seed 42; GRPO learning rate 1e-5, two tasks × four rollouts per batch, 16 attempted batches / at most 16 acknowledged updates. Both arms use identical task order, limits, source data, and strict all-claims-v7 Qwen3.5-397B judge. No confirmation tasks are used.

Quality-only: failed=0, partial=0.2×supported required coverage, accepted=0.9. Efficiency: identical failed/partial rewards; accepted=0.9+0.1×clip(1−compute/task budget,0,1). Compute=input tokens+2×output tokens+100×tool seconds. Task budgets remain frozen. Unresolved grades remain null and exclude their GRPO group. The grader bridge was already implemented; this experiment activates it in new configs. New code records efficiency observability without changing reward arithmetic.

The same 32 selection tasks are evaluated at step 0 and the final checkpoint, using unchanged strict quality scoring. Primary outcome: accepted-answer rate over all assigned tasks. Efficiency: all-attempt compute and output tokens per accepted answer, plus mean input/output tokens, tool calls/time, completion and scoring coverage. No acceptance implies undefined cost per acceptance. Token counts include the complete investigation, not only final-answer prose. Wall time includes provisioning/grading and is diagnostic only. Concurrent execution makes latency unsuitable as a causal efficiency endpoint.

Compare final efficiency versus final quality-only and each arm versus its own initial baseline; report difference-in-differences. Report paired repository-cluster bootstrap 95% intervals, not independent-episode intervals. No winner unless both final scoring coverages are ≥95%, acceptance-change lower bound ≥−0.02, and relative compute-per-acceptance-change upper bound <−0.10. A single small seed remains preliminary even if these thresholds pass. Training reward alone never proves improvement.

Record acknowledged updates, skipped/excluded groups, bonus-bearing accepted trajectories, and complete raw evaluation tables. Inspect missing grades and newly failed answers. Preserve all results, including failure/inconclusive outcomes; no post-result reward or cohort tuning.

## Execution and cost

Frozen bundle: `0512c3ed7b01669509461a169a0ec4b5ab3d784c2d3d9bab67e1b42e08df1b1a`. Dedicated isolated Modal controller and volume. Per-arm ledger cap $320, total accounting allocation $650 including $10 infrastructure contingency. Conservative combined operation/controller estimate $580.67; reservations are not provider invoices. Fresh provider model prices verified at dispatch.

W&B group: efficiency-rl-v1, project repository-qa-training. Live `training/*` metrics use attempted batches; `evaluation/*` metrics use optimizer steps. Answers, trajectory viewers, and judge inspection remain available through existing observability artifacts.

Verification: 67 targeted tests passed, including reward-tier ordering, null-grade handling, all-attempt efficiency aggregation and pipeline/storage integration. An additional 22 collection/concurrency tests and 5 paired-analysis tests passed. Run status and results will be appended after collection.

## Interpretation notes recorded during execution

The initial quality-only and efficiency evaluations accepted 11/32 and 7/32 answers, with scoring coverage 28/32 and 30/32, respectively. Only 6/32 submissions were byte-identical; none of the nine differing grade/status outcomes came from identical submissions. Initial recipe matching does not imply identical generated trajectories. Report this baseline variability with final effects.

Training uses different sampled tasks each batch. A downward training token curve alone can reflect easier tasks; only matched held-out evaluations can substantiate efficiency improvement. GRPO standardizes rewards within each group, so a 0.1-bounded bonus is small in raw reward units but can supply a full standardized advantage signal when accepted candidates otherwise tie. No claim of a small parameter update follows from a small reward coefficient.

## Diagnostic review

Reviewed every lost-acceptance, incomplete and unresolved final-selection case in outcome-transitions.json against its saved answer and grading reason. Efficiency lost three initially accepted answers and gained five; the three losses were judge-reported citation entailment/support issues, with missing required facts or unsupported assertions on one. Two final efficiency episodes exhausted the rollout budget without an answer; two completed answers had unresolved grading. Control lost six and gained two; the reported losses involved citation/support or missing-citation checks. Control had one budget exhaustion and four unresolved grades. These are saved verifier findings, not a new independent correctness audit. No scores, rubrics or rewards were changed after inspection.

Only 2/26 eligible efficiency training groups had different normalized GRPO advantages than the quality-only reward would produce on those same trajectories; one broke an all-accepted tie. Seven bonus-bearing trajectories contributed gradients, but most raw bonus changes cancel under normalization. This explains why successful reward wiring alone is weak evidence of a behavioral effect.

The comparison publisher exports only aggregate metrics and summary.json. Automatic approval review rejected a proposed upload containing saved answers and per-task diagnostic artifacts, so those comparison artifacts remain local. The original training tracker retains its existing observability behavior.

## Results

**Decision:** no_demonstrated_efficiency_win.

| Arm | Phase | Accepted | Scored | Output tokens/task | Compute/accepted |
|---|---|---:|---:|---:|---:|
| quality-only | initial | 11/32 | 28/32 | 242.84 | 35,931.6 |
| quality-only | final | 7/32 | 28/32 | 238.91 | 55,284.5 |
| efficiency | initial | 7/32 | 30/32 | 240.22 | 54,633.1 |
| efficiency | final | 9/32 | 30/32 | 245.31 | 44,027.7 |

Acknowledged optimizer updates: quality-only: 10 of 16 attempted batches, efficiency: 9 of 16 attempted batches.
Recorded combined reservations: $101.12; not actual provider billing.

See summary.json for paired repository-cluster intervals, undefined-bootstrap counts and promotion gates; per-task.csv preserves all assigned tasks. Single seed, 32 selection tasks, same-family model judge with no human calibration. Training bonus may be sparse. Wall time is confounded by parallel service load. Reservations are not invoices. No post-result tuning or confirmation.
