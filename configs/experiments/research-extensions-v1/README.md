# Decomposition and distillation extensions

This supplements current-v8 without changing its frozen configs, manifests, run IDs or ledgers. All **13 execution configs** passed offline schema/input, cost and (where applicable) native SFT rendering validation. No paid jobs were launched. See the [classification](../../../docs/EXPERIMENT_CLASSIFICATION.md), [procedure 07](../../../experiment%20procedures/07-question-decomposition.md) and [procedure 08](../../../experiment%20procedures/08-investigation-distillation.md).

| Config names | Contrast | Readiness |
|---|---|---|
| `07-{control,decompose}-selection-r{1,2}.json` | Frozen model, unchanged tools/limits, baseline vs question-checklist prompt | Four offline-validated selection measurements |
| `07-{control,decompose}-confirmation-r{1,2}.json` | Same paired contrast on frozen confirmation cohort | Four finalist-only measurements; lock selection decisions first |
| `08-investigation-sft.json` | Supervise all admitted tool actions and final answer | One-lineage smoke only |
| `08-answer-only-sft.json` | Same examples/context; supervise final answer only | One-lineage smoke only |
| `08-{investigation,answer-only}-then-grpo.json` | Fork respective selection-best SFT weights, fresh optimizer, identical GRPO | Two sequential dependent smoke arms; parent checkpoint required |
| `08-direct-grpo.json` | Matched base→GRPO control | Offline-validated; reuse exact prior comparator if fully matched |

The decomposition prompt does not expose an explicit plan or add a planning call. The answer-only control keeps teacher observations and actions as context; it changes loss selection, not the evidence available during SFT. Entire original demonstrations are still verified before selecting targets. This is not question-only answer SFT and not yet a stronger-teacher distillation release.

All configs inherit current-v8's model, dataset-v6, grader-v7, 32/85 cohorts, policy limits and version locks. There is only one admitted current full investigation. Collect and freeze more examples before an effectiveness study; do not mistake successful config validation for adequate data or demonstrated gains. The complete new training sequence is seed 42 only; additional training seeds and distillation confirmation require fresh matched configs.

[study-plan.json](study-plan.json) records dependencies, counts, readiness and offline estimates. [budget-plan.json](budget-plan.json) preserves the current-v8 plan and adds **proposed**, separate ledger caps using existing price snapshots plus 10% rounded allowance. It is not a new spending authorization or instruction to run all optional controls. Reconcile actual reservations and live prices before dispatch.

```sh
# Validate without allocating providers
.venv-eval/bin/python -m training_pipeline validate \
  --config configs/experiments/research-extensions-v1/07-decompose-selection-r1.json
.venv-eval/bin/python -m training_pipeline validate \
  --config configs/experiments/research-extensions-v1/08-answer-only-sft.json

# Reproduce in a fresh directory; never overwrite a frozen campaign
PYTHONPATH=. .venv-eval/bin/python scripts/prepare_research_extensions.py \
  --source configs/experiments/current-v8 \
  --output configs/experiments/research-extensions-v2
```

After collecting more verified trajectories, first prepare a fresh current campaign with `scripts/prepare_current_experiments.py --output <fresh-campaign>`, then use it as `--source` above. Preparation and packaging stay offline. Package SFT→GRPO dependencies sequentially using the existing remote interface; do not submit them in a dependency-containing parallel campaign.
