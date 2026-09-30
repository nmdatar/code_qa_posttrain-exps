# Current experiment configurations — v7 citation fix

Use these configurations with the [current experiment procedures](../../../experiment%20procedures/README.md). They inherit the corrected [main GRPO config](../grpo-main-v7/run.json): atomic-claims-v4, experimental-reference-v4, all-claims-v7, positive-coverage-v4 training reward, independent-factual-coverage-v3 and answer-first-invalid-citations-v1. Strict evaluation remains separate from training reward. The judge has 65,536 context / 8,192 output tokens and one mechanical repair. The original policy rollout limits remain fixed.

All 16 execution configs pass offline schema, release and cohort validation. Throughput configs additionally validate their regenerated current-data manifest. This does not establish live readiness or authorize additional spending. No new runs were launched by this update.

| Files | Purpose | Conservative per-run estimate | Readiness |
|---|---|---:|---|
| `01-baseline.json`, `01-baseline-selection-repeat.json` | Matched strict selection controls | $51.93 each | Planned; shared baseline ledger |
| `01-baseline-confirmation.json` | Locked confirmation control | $137.95 | Planned; do not use for tuning |
| `01-throughput-c*-r*.json` | 16 training tasks × four attempts, no updates; two repeats at each concurrency | $103.87 each | Complete schedule exceeds baseline allocation |
| `02-direct-grpo.json` | Exact copy of submitted corrected main run | $477.46 | Existing run identity; do not launch a duplicate |
| `03-lr-5e-6.json` | Matched lower learning rate, from base | $477.46 | Exceeds existing $80 LR ledger cap |
| `03-lr-1e-5.json` | Explicit matched control template | $477.46 | Reuse experiment 2 if fully matched; no extra funding |
| `03-group-4-template.json`, `03-group-8-template.json` | Two tasks × four vs one task × eight | $477.46 each | Set both to selected LR before freezing; no extra funding |

Estimates use the inherited recorded price snapshots, include worst-case judge repairs, and are reservation bounds rather than invoices. They exclude prior ledger commitments and the separate controller reservation. The complete experiment-1 schedule is approximately $1,072.74 before controller costs and prior commitments, above the $300 baseline ledger. Individual `within_ceiling` estimator labels do not imply cumulative affordability. Refresh prices and reconcile the authoritative remote ledger before any new launch.

The project ceiling remains $1,000 and original ledger identities are preserved. No caps or existing authorizations were expanded. LR and group comparisons use 12 batches maximum, evaluation initially/every six successful updates/finally, four initial no-update batches, zero group retries, and current retention settings. This is a bounded screen rather than the original 30-update three-seed study. Both group arms have eight episodes per batch; equal episode counts do not ensure equal generated tokens. Freeze equal rollout-output allowances before dispatch.

[study-plan.json](study-plan.json) records all six procedures and their blockers. It is planning metadata, **not** a runner config. Procedures 4–6 remain specified in their documents until their SFT adapter, tiered reward, or tool/context integrations exist. Do not pass this metadata or `throughput-tasks.json` to the config validator. The broader roadmap's REINFORCE, PPO/learned reward, annotations and 9B extensions remain deferred.

For offline validation:

```sh
.venv-eval/bin/python -m training_pipeline validate --config configs/experiments/current-v7/03-lr-5e-6.json
```

[Validation and estimates](../../../reports/experiment-current-v7/validation.json) include every config. Frozen old configs and historical result reports remain unchanged so past runs can still be reproduced. New bundles must pin the current source revision/reward identity; a version string alone does not prove inclusion of the citation-routing fix.
