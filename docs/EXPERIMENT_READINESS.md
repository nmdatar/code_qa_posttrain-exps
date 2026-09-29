# Experiment readiness implementation

The current implementation adds the following offline-tested capabilities. It does not establish judge accuracy, concurrent provider safety, or durable archive restoration.

## Frozen cohorts and baseline

`experiments/qwen4b-cohorts-v1.json` contains the procedure's 32 selection and 85 confirmation task IDs, dataset identity, family counts, seed, generator source checksum, and manifest hash. It was generated from the existing admitted release without provider calls. The manifest is task-disjoint; both development cohorts share repository families.

Pin both fields in the experiment configuration's `evaluation` object:

```json
{
  "every": 5,
  "max_tasks": 32,
  "temperature": 0,
  "cohort_manifest": "experiments/qwen4b-cohorts-v1.json",
  "cohort_sha256": "4b229d10481e6cad2a7c0284ed8de0aa045ab7ed55bed69d115fc74e64186d19"
}
```

Training automatically routes to selection. Explicit cohort evaluation always evaluates the complete cohort, including all 85 confirmation tasks, regardless of the legacy `max_tasks` prefix setting. Input validation checks manifest integrity, dataset identity, task membership, family allocation, and train/development separation before remote allocation. Without a pinned manifest, legacy prefix evaluation remains available and is not labeled as a procedure cohort.

```sh
# Offline; refuses to overwrite an existing manifest.
.venv-eval/bin/python -m training_pipeline cohorts --config "$CONFIG" --output "$NEW_COHORT_MANIFEST"

# Paid operations: require a separately funded config and a fresh output directory.
# Baseline creates neither a training client nor a checkpoint.
.venv-eval/bin/python -m training_pipeline baseline --config "$CONFIG" --cohort selection
.venv-eval/bin/python -m training_pipeline baseline --config "$CONFIRMATION_CONFIG" --cohort confirmation
.venv-eval/bin/python -m training_pipeline evaluate --checkpoint "$CHECKPOINT" --output "$OUTPUT" --cohort confirmation
```

Baseline still uses the solver backend's capability checks and renderer. Provider behavior and throughput remain to be verified live. Evaluation estimates omit training/checkpoint costs and conservatively allow for the larger pinned cohort. Per-call spending reservations continue to enforce the configured local ceiling.

## Measurement and comparisons

Evaluation reports now include family IDs, episode usage, demonstrated quality (resolved points / all assigned tasks), resolved-only mean, scoring coverage, completion rate, and the 95% coverage gate. Unresolved rows retain null rewards.

```sh
# Offline, 10,000 family resamples, seed 20260928.
.venv-eval/bin/python -m training_pipeline compare --base "$BASE_REPORT" --candidates "$SEED42_REPORT" "$SEED43_REPORT" "$SEED44_REPORT" --output "$COMPARISON"
```

Comparison requires identical task sets, family assignments, dataset, environment, reward version, and cohort identities. It averages candidate differences per task before resampling whole families, retains individual-run results, and reports best/worst unresolved sensitivity. Its statistical gate is not automatic model promotion: seed provenance, frozen finalist selection, small-family limitations, and archive restoration still require review. Reports from different harness variants intentionally require a separately scoped comparison protocol.

## Training controls

Optional top-level configuration:

```json
"stopping": {
  "initial_zero_batches": 5,
  "regression_delta": 0.10,
  "regression_checks": 2
}
```

The no-signal stop counts attempted GRPO batches before the first contributing update in each stage. The regression stop uses selection demonstrated quality against the initial baseline and counts consecutive scheduled/end-of-stage checks. The initial baseline requires 95% scoring coverage. Baseline/regression state is persisted in additional recovery checkpoints; launch estimates include their storage costs. A stopped checkpoint refuses blind resume; diagnosis and a new fork are required.

Task batches no longer repeat a task when crossing an epoch boundary. The next permutation defers already-picked tasks without dropping them; cursor/order/RNG restoration remains deterministic. SFT returns a partial batch at epoch end rather than filling it with repeated examples. Repository collection SFT data/admission remains unsupported.

## Remaining work

- Independently reviewed judge calibration, adversarial answers, repeated live grading, and a frozen evaluator decision.
- Bounded parallel rollout/judge execution, provider concurrency verification, and the procedure 01 throughput study.
- Durable checkpoint export/import with verified optimizer restoration and archive-gated best-model selection.
- Shared rollout-output-token allowance with crash-safe accounting across retries and resume.
- SFT collection bridge, tiered verifier integration, and the tool/retrieval/history experiment arms.
- Separately authorized budgets and live runs. No paid experiments were launched for this implementation.
