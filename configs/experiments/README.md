# Current experiment configurations — prepared, not launched

Use Qwen/Qwen3.5-4B, rank 8, a frozen NVIDIA Nemotron-3-Nano-30B-A3B judge,
and 48-hour (`172800` second) retention for both sampler and optimizer/training
checkpoints. These settings apply to new saves; they cannot revive expired
historical checkpoints. No experiment has been authorized to start yet.

The project allocation is $1,000 total, not $1,000 per run. See
[project-budget.json](project-budget.json): $50 historical/other-worktree reserve,
$5 single grading check, $45 baseline, $650 direct GRPO, $160 combined learning-rate
screens, and $90 confirmation/contingency. The historical reserve is not a bill;
reconcile all worktree spending before launch. Caps are enforced per ledger,
not globally across independent worktrees. Do not copy a config to a fresh ledger
and treat that as additional authorization.

| Config | Operation when execution is authorized | Conservative estimate | Cap |
|---|---|---:|---:|
| `01-baseline.json` | `baseline --cohort selection` | <= $6.66 | $45 shared |
| `01-baseline-confirmation.json` | `baseline --cohort confirmation` | <= $6.66 | Same $45 |
| `02-direct-grpo.json` | `run`: 16 tasks × 4 attempts, LR 1e-5, at most 30 updates / 30 batches | ~$613 | $650 |
| `03-lr-5e-6-screen.json` | Later screen: 4 tasks × 4 attempts, 5 updates / 8 batches | ~$78 | $80 |
| `03-lr-1e-5-screen.json` | Matched later screen at LR 1e-5 | ~$78 | $80 |

Estimates are conservative reservations, not measured bills. The estimator
allows evaluation after every update and the larger 85-task cohort, even though
these configs schedule selection evaluation every five updates. Unresolved or
zero-variance groups can yield fewer updates than the maximum. The two learning-
rate screens intentionally use smaller batches than the full procedure; label
results as screens, not completion of procedure 03. Three-seed finalist repeats
and group-size comparisons require reallocation within the ceiling before launch.

All five configs pin the same 858 training / 117 development release and frozen
32/85 selection/confirmation manifest. Use the existing native runner:

```sh
# Offline only; this does not start an experiment.
python -m training_pipeline validate --config configs/experiments/02-direct-grpo.json
```

Install optional `[training,tinker,remote]` dependencies in the execution environment.
The existing `.venv-posttrain` was used for the one live grading check; `.venv-eval`
does not yet have cookbook. Baseline and cohort commands depend on the experiment
readiness code, so use the tested integrated revision rather than an old checkout.

No runnable SFT, efficiency-reward, symbol-navigation, retrieval, or compression
arms are claimed ready: their data/adapter prerequisites remain. Concurrency
benchmark arms and durable best-checkpoint archives also remain unimplemented.
48-hour remote retention is not a permanent model archive.

Other worktree `posttrain` configs use separate fresh training releases, different
schemas, rank/temperature settings, and a $20 hard cap. They are not interchangeable
with these configs. Historical Qwen-9B/397B evaluation configs remain unchanged and
are not the current experiment defaults. Detailed inventory and source checks are
in `reports/experiment-config-audit.json`.
