# Expanded three-experiment study

Status: staged launch authorized after the user instructed continuing until credits run out. Enforce the proposed $5,000 additional limit ($6,000 total including the historical allocation); stop sooner on provider-credit exhaustion. No credit purchases or automatic ceiling increases. Phase 1 collects teacher demonstrations, then runs direct REINFORCE seed42. Teacher and student require separate price-bound ledgers, initially allocated $2,200 and $2,800; unused allocation may be transferred after authoritative reconciliation while retaining the same total ceiling. Launch receipt is recorded separately.

## Experiments

1. Direct running-baseline REINFORCE, using factual-coverage reward.
2. Identical REINFORCE with source-grounded error/citation penalties added to the same factual-coverage base. Report stricter reward eligibility separately as part of this intervention.
3. Complete verified investigation SFT followed by the same factual REINFORCE as experiment 1. Evaluate the SFT-only checkpoint diagnostically as well.

All three use Qwen3.5-4B, rank 8, seeds 42/43/44, the v6 task/rubric release and all-claims-v7 grader. For each seed, all arms share training task order, rollout counts and episode limits. SFT starts from the same base, and subsequent RL starts a fresh optimizer and running baseline. Historical GRPO is context, not a new replicated control.

## Revised staged funding proposal

The initial ceilings below sum separate worst-case allocations for every job. A smaller shared ledger can reuse unused headroom after completed jobs. The revised proposal keeps the expanded 256-task/three-seed design and uses a **$5,000 additional shared limit ($6,000 total including the existing allocation)**, with sequential dispatch and conservative checks before each launch. This staged ceiling is now authorized as the bounded interpretation of the user’s instruction to continue. Expected additional consumption remains $3,000–$5,000; completion is not guaranteed if usage or demonstration rejection is higher than expected. The code will stop when the remaining budget cannot safely cover the next operation. Teacher collection, reward validation, SFT and confirmation all share this limit. No automatic increases. Configurations are in `configs/experiments/expanded-studies-v1/staged/`.

## Initial independently allocated choices

| | Expanded | Full training set |
|---|---:|---:|
| Distinct training tasks per seed | 256 | 858 |
| Trajectories per optimizer batch | 32 | 52 |
| Scheduled batches per arm/seed | 32 | 66 |
| Seeds per arm | 3 | 3 |
| Total RL trajectories | 9,216 | 30,888 |
| Rough expected incremental spend | $3,000–$5,000 | $9,000–$14,000 |
| Proposed total project ceiling | $24,000 | $61,000 |
| Planning elapsed time including teacher/SFT | 1–2 days | 2–4 days |

Expected costs extrapolate prior measured reservations and are uncertain, not invoices or guarantees. Conservative ceilings account for maximum context, generations, grader repairs, retained checkpoints and bounded controller time; they are much higher than expected consumption. These ceilings include the prior $1,000 allocation, teacher collection, reward validation, SFT, confirmation and contingency. A lower user-imposed ceiling can be enforced but may stop the experiment before it completes. Runtime assumes 2–3 concurrent jobs and available service capacity; revise from observed throughput.

The exact estimator output and proposed allocations are in `design-and-costs.json` and `configs/experiments/expanded-studies-v1/{expanded,full}/budget-proposal.json`. Refreshed model prices are frozen in `prices.json`. Generated configurations are not spending authorization.

## SFT data gate

Only one existing unique investigation meets complete strict/source/token admission. Prepare 1,200 teacher trajectories over 600 training tasks across 31 families. The SFT arm requires at least 200 distinct admitted demonstrations across at least 15 families, plus source audit and exact student token-rendering checks. Teacher success rate is unknown. If the gate fails, report it and revise the plan explicitly; do not pad with flawed traces or silently increase the budget. Freeze the admitted manifest before SFT. Downstream SFT templates deliberately cannot launch before this gate.

## Reporting and confidence

Freeze the final scheduled checkpoints and analysis before evaluating the untouched 85-task confirmation cohort. Primary comparisons are direct REINFORCE versus fresh base, aligned reward versus direct, and SFT+REINFORCE versus direct. Use three paired seeds, paired task bootstrap intervals, Holm-adjusted tests, per-repository results and missing-grade sensitivity. Report acknowledged optimizer updates and actual contributing trajectories: scheduled batches do not guarantee updates when grading fails.

The 85 confirmation tasks cover only seven repository families. Three seeds do not create 255 independent tasks, and more training cannot guarantee statistical significance. This supports an honest replicated comparison, but small 5–10 percentage-point gains may remain inconclusive. A strong small-effect claim requires a larger independently held-out benchmark. Do not tune against confirmation results.

## Execution gates and order

After the user chooses a scope and raises the explicit ceiling: freeze hashes and named ledger allocations; validate/upload teacher and independent direct runs; execute training-only reward controls before aligned runs; collect/admit/audit demonstrations before SFT and SFT+RL; then freeze all final models and run symmetric confirmation. No confirmation results have been used to design these interventions. Keep all original artifacts and authoritative budget ledgers and report reservations rather than invoices.

Details: `statistical-design.md`, `reward-design.md`, `sft-design.md`, and `sft-collection-validation.json`.
