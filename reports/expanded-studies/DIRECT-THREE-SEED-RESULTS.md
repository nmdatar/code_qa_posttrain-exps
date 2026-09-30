# Direct REINFORCE: three-seed selection results

**The expanded direct REINFORCE arm is complete across three seeds. More training did not demonstrate an improvement:** mean demonstrated strict credit fell from **25.35% to 20.83%**, a **−4.51 percentage-point** change. All three final seed results were lower than their respective initial results. The uncertainty intervals include no population change; these data do not prove the algorithm is generally worse.

| Seed | Acknowledged updates | Training tasks / attempts | Strict passes before → after | Demonstrated credit before → after | Resolved grades before → after |
|---|---:|---:|---:|---:|---:|
| 42 | 32 | 256 / 1,024 | 7 → 6 | 23.96% → 21.88% | 28 → 29 |
| 43 | 32 | 256 / 1,024 | 7 → 5 | 21.88% → 15.63% | 28 → 31 |
| 44 | 32 | 256 / 1,024 | 9 → 8 | 30.21% → 25.00% | 30 → 30 |

Each evaluation uses the same **32 unique selection tasks**. Across three repeated evaluations per endpoint there are 96 task-seed outcomes, **not 96 independent held-out questions**. Strict passes total 23 before and 19 after; resolved grades total 86 and 90. These totals describe repeated outcomes only. Fractional demonstrated credit is a different metric: partial strict scores contribute to it, and unresolved grades provide no demonstrated credit without being declared incorrect. Initial coverage fails the configured scoring gate in every seed; final coverage passes only in seed 43.

## Paired uncertainty and sensitivity

For each task, average its paired final-minus-initial credit changes over the three seeds, then resample those **32 task averages**. A 20,000-draw percentile bootstrap gives **95% interval −12.50 to +3.82 percentage points**. RNG seed is 42. This conditions on the three observed training seeds and does not estimate broad training-seed variability, judge noise or within-repository dependence. It is not a 96-row independent bootstrap.

There are only **seven repositories**, with `pydata/xarray` contributing 16/32 tasks. An exploratory repository-cluster bootstrap, retaining all task/seed pairings and recalculating task-weighted means, gives **−9.38 to +6.67 points** (20,000 draws; RNG seed 43). Equal weighting of repository means yields **0.00 points**. Excluding xarray yields **+2.08 points**; the xarray task subset itself changes by −11.11 points. This concentration limits generalization and makes the pooled negative mean sensitive to repository mix.

| Repository | Selection tasks | Mean change over seeds |
|---|---:|---:|
| cekit/cekit | 3 | +11.11 pp |
| dwavesystems/dwave-cloud-client | 3 | 0.00 pp |
| ethereum/web3.py | 2 | 0.00 pp |
| frictionlessdata/frictionless-py | 3 | +11.11 pp |
| microsoft/pybryt | 3 | −11.11 pp |
| pydata/xarray | 16 | −11.11 pp |
| tox-dev/tox | 2 | 0.00 pp |

Allowing every missing grade independently any value from zero to one yields an overall mean-change sensitivity range of **−14.93 to +1.74 points**. These bounds concern unknown observed grades, not population sampling or grader correctness. All three observed seed changes are negative, but three seeds are too few for a precise seed-variance claim.

## Training accounting and matching conditions

The three successful lineages contain **96 acknowledged updates, 3,072 scheduled training trajectories, 3,034 resolved grades and 2,907 contributing trajectories**. There are 768 task-seed pairs across **554 unique training task IDs**; each seed samples 256 tasks, so task subsets vary by seed. There were 21 excluded groups and 373 zero-variance groups. The whole-group exclusion rule can also exclude resolved members of an incomplete group.

Seed 42 combines two original updates with 30 continuation updates. Seed 43 combines 26 original updates with six continuation updates. Seed 44 completes in one run. The 64 trajectories in the two abandoned unacknowledged batch attempts are excluded from learned-step totals but retained in cost accounting. Earlier abandoned seed-42 infrastructure runs remain separate records.

All seeds share data identity, environment identity, rubric/grader reward identity, selection manifest and task membership. Base model, judge, training stages, reward, solver limits, evaluation settings, concurrency and checkpoint policy match. Seed-specific scientific config hashes differ as expected. Operational IDs, ledgers, controller timeout and recovery history differ. Seed 44 has additive efficiency telemetry; the training objective remains the same positive factual-coverage REINFORCE objective. This verifies matching scientific settings, not bitwise-identical executable histories.

## Cost, artifacts and scope

Final private-ledger reservations for the three successful seed lineages are **$376.22, $364.85 and $343.10**, totaling **$1,084.17**. These include interrupted attempts within those lineages and their continuations. They **exclude** the earlier abandoned seed-42-v1 run, teacher/SFT and aligned arms, future confirmation, and invoice reconciliation. All three figures now come from verified authoritative archived ledgers. Reservations are not final provider charges.

The retired shared student ledger adds **$62.58**, bringing completed lineages plus retired student reservations to **$1,146.76**. That ledger includes **$18.23 for an earlier aborted phase-1 controller**, so this supplemental total is not solely the cost of the three completed runs. The remaining **$44.35** covers stopped direct seed-42-v1 and its controller; all 96 episode IDs in that ledger match the stopped run. Teacher spending is absent from this student ledger.

Seed 44’s completed archive has 8,339 files, zero JSON parse errors and all hashes verified. Its local sampler tar is **step 16 best-eligible checkpoint `ckpt-b9292bdde8ec4386a94da77a1d42049e`**, not the fixed final step-32 checkpoint `ckpt-3eb1a5bfe97745b9ba9b3a691b242e1b`. The final checkpoint manifest and evaluation are archived, but final sampler/optimizer weights have no local export; remote TTL applies. Per-seed reports preserve this distinction, and no permanent fixed-final weight export is implied. Runtime is not inferred from individual episode/evaluation durations.

This is **one completed arm on selection**, with confirmation untouched. The other two arms are not included in this result. No Holm-adjusted three-arm result, final model-promotion decision or comparative superiority claim is available. The intermediate seed-43 peak is reported transparently but does not replace its preregistered final checkpoint.

[Machine-readable three-seed results](direct-three-seed-results.json) · [Seed 42](DIRECT-SEED42-RESULTS.md) · [Seed 43](DIRECT-SEED43-RESULTS.md) · [Seed 44](DIRECT-SEED44-RESULTS.md)
