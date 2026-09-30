# Current configuration entry point

The [experiment classification](../../docs/EXPERIMENT_CLASSIFICATION.md) maps the complete research inventory to model, harness, evaluator/data and infrastructure changes. [research-extensions-v1](research-extensions-v1/README.md) adds decomposition and matched distillation controls alongside current-v8.

Use [current-v8/README.md](current-v8/README.md) for the latest dataset-v6/judge-v7 suite and implementation readiness. Historical current-v7 and other submitted configs below remain frozen. No new paid experiments were launched by this preparation.

# Experiment configurations

Use [current-v7/README.md](current-v7/README.md) for new experiment planning and version-aligned configs across procedures 1–6. The source of the active corrected GRPO run is [grpo-main-v7/run.json](grpo-main-v7/run.json). Current-v7 contains an exact copy of that run, not a second launch.

The older root configs, ready-v4 configs, pilot and diagnostic bundles remain frozen historical inputs. Their older costs, readiness statements and launch instructions below do not describe the current campaign.

## Historical configuration notes

> **Current budget-capped plan:** see [ready-v4/README.md](ready-v4/README.md). The original configs below describe historical protocol-v3 runs; use the versioned ready-v4 configurations for the new study.

# Current execution: Modal + full claim audit

The experiment configs now require detached Modal execution, use the
`all-claims-v3` extraction-and-assessment grader, and pin Qwen3.5-397B-A17B as the
judge. Follow [the remote execution guide](../../docs/REMOTE_EXPERIMENTS.md) to
prepare, submit and download the initial campaign. Local collection execution is
disabled; offline validation remains available.

Existing caps/ledgers are preserved. The stronger grader changes costs and scores:
small screens are now up to **4 updates / 4 attempted batches**, larger direct
GRPO **18 / 18**, and older LR screens **5 / 6**. Run IDs/output paths have a new
`all-claims-v3-modal` suffix. Prior results remain separate. No cloud deployment or
new paid experiment was performed for this migration.

The following is historical preparation context; its local launch commands,
old grader, run counts and cost estimates are superseded by the remote guide.

---

# Current experiment configurations — prepared, not launched

Use Qwen/Qwen3.5-4B, rank 8, a frozen Qwen3.5-397B-A17B claim judge,
and 48-hour (`172800` second) retention for both sampler and optimizer/training
checkpoints. These settings apply to new saves; they cannot revive expired
historical checkpoints. No experiment has been authorized to start yet.

The project allocation is $1,000 total, not $1,000 per run. See
[project-budget.json](project-budget.json): $50 historical/other-worktree reserve,
$5 single grading check, $120 baseline, $650 direct GRPO, $160 combined learning-rate
screens, and $15 confirmation/contingency. The historical reserve is not a bill;
reconcile all worktree spending before launch. Caps are enforced per ledger,
not globally across independent worktrees. Do not copy a config to a fresh ledger
and treat that as additional authorization.

| Config | Operation when execution is authorized | Conservative estimate | Cap |
|---|---|---:|---:|
| `01-baseline.json` | `baseline --cohort selection` | <= $11.17 | $120 shared |
| `01-baseline-confirmation.json` | `baseline --cohort confirmation` | <= $11.17 | Same $120 |
| `02-direct-grpo.json` | `run`: 16 tasks × 4 attempts, LR 1e-5, at most 30 updates / 30 batches | ~$637.22 | $650 |
| `03-lr-5e-6-screen.json` | Later screen: 4 tasks × 4 attempts, 5 updates / 8 batches | ~$57.31 | $80 |
| `03-lr-1e-5-screen.json` | Matched later screen at LR 1e-5 | ~$57.31 | $80 |

Estimates are conservative reservations, not measured bills. The estimator
uses the configured selection cohort and evaluation cadence for training, with
conservative allowances for stage-end and early-stop evaluations. Standalone
baseline estimates still allow the larger confirmation cohort. Unresolved or
zero-variance groups can yield fewer updates than the maximum. The two learning-
rate screens intentionally use smaller batches than the full procedure; label
results as screens, not completion of procedure 03. Three-seed finalist repeats
and group-size comparisons require reallocation within the ceiling before launch.

All experiment configs pin the same 858 training / 117 development release.
The full configs use the frozen 32/85 manifest; the new small screens use a
16-task subset of that selection cohort, preserving the original confirmation holdout. Use the existing native runner:

```sh
# Offline only; this does not start an experiment.
python -m training_pipeline validate --config configs/experiments/02-direct-grpo.json
```

Install optional `[training,tinker,remote]` dependencies in the execution environment.
The existing `.venv-posttrain` was used for the one live grading check; `.venv-eval`
does not yet have cookbook. Baseline and cohort commands depend on the experiment
readiness code, so use the tested integrated revision rather than an old checkout.

No runnable SFT, efficiency-reward, symbol-navigation, retrieval, or compression
arms are claimed ready: their data/adapter prerequisites remain. Durable best-checkpoint archives remain unimplemented. Concurrency benchmark
arms are now prepared; see [the concurrency guide](../../docs/CONCURRENCY.md).
48-hour remote retention is not a permanent model archive.

Other worktree `posttrain` configs use separate fresh training releases, different
schemas, rank/temperature settings, and a $20 hard cap. They are not interchangeable
with these configs. Historical Qwen-9B/397B evaluation configs remain unchanged and
are not the current experiment defaults. Detailed inventory and source checks are
in `reports/experiment-config-audit.json`.


The eight `01-throughput-c*-r*.json` configs and the additional baseline selection
repeat share the $120 baseline ledger. All other allocations are unchanged;
the project ceiling remains $1,000. Use `qa-train benchmark` for throughput
configs, which never allocates training. No experiments have been launched.

## Small screens before the larger run

Start with `02-direct-grpo-screen.json`. For a matched learning-rate comparison,
use `03-lr-5e-6-small-screen.json`; the direct screen is the 1e-5 control.
Both use rank-8 LoRA, four training tasks × four answers, at most five updates
and eight attempted batches, eight rollout workers, and four judge workers.
Three initial batches with no usable update stop the run. Each evaluates the
same 16 frozen selection tasks initially and at the end, without intermediate
checks. A normal five-batch run collects 80 training + 32 evaluation answers;
retries and skipped updates can increase that count. The estimate reserves up
to 304 episodes (including an extra conservative terminal evaluation) and is
$51.91 per arm against a $60 cap. These are screening signals, not conclusive
comparisons; confirm promising changes in the larger run.

The small configs reuse the existing direct-GRPO / LR-5e-6 ledgers. Their spending
counts against those campaign allocations. They are alternatives to running the
larger configs immediately, not extra allocations. Later large launches must
fit the remaining balance. Fresh run IDs and outputs isolate screening results.

For experiment 1, use only the c08 and c16 throughput arms initially (64 answers
per arm); defer the slow c01 arms and confirmation/repeats until needed. The
throughput benchmark measures execution speed, not training quality. Use its
measured episode and queue timings to estimate subsequent wall-clock duration;
cost reservations and per-call timeouts are not runtime forecasts.

The current judge uses `source-claims-v2`: it selects numbered candidate passages
and software saves their exact text, avoiding quote copying/ellipsis failures.
The previous 15-case diagnostic rejected all six wrong/injection answers with
zero scores. Offline tests cover the passage protocol; no additional paid grader
validation or experiment was launched for this change. See
[claim grading](../../docs/CLAIM_GRADING.md).

Current estimates after the evaluation-schedule fix are in
[the screening cost audit](../../reports/screening-config-cost-audit.json), which
supersedes the prior Qwen397 cost audit. The full eight-arm throughput schedule
still exceeds its shared $120 budget. Historical Nemotron examples and grading
artifacts remain unchanged. Old v1 runs must not resume as v2 runs.

## Logging and checkpoint storage

Prepared experiment configs enable online W&B logging to `repository-qa-training`
using the authenticated/default W&B entity. Metrics, evaluations, and checkpoint
manifests are logged; full local traces and raw judge responses remain in each
run output directory, with selected public traces uploaded. W&B initialization
or upload failures are recorded as `tracking_failure` in local `events.jsonl`;
the runner continues locally, so verify the live W&B run at launch. Enabling
logging does not itself authenticate or verify connectivity.

GRPO saves initial and per-update Tinker training-state and sampler checkpoints,
with verified remote references in local/W&B checkpoint manifests. The configured
retention is 48 hours; these are not permanent archives. Throughput and unchanged
base evaluations perform no training and intentionally save no checkpoints.
