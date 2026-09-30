# Experiment 3 results: provisional preference for LR 5e-6, group size 4

All four scheduled runs exited successfully. **LR 5e-6 with group size 4 had the best observed final result: 10/32 strict passes (31.25%), up from 6/32 (18.75%).** This is a provisional recipe choice, not an established optimum. Every final evaluation missed the prespecified 95% scoring-coverage requirement, and LR 1e-5/group4 was severely affected by infrastructure failures during the billing interruption.

## Final scheduled checkpoint results

Strict quality is the evaluator's aggregate reward divided by all32 tasks, with unresolved grades contributing no demonstrated credit. It can include partial credit: initial quality does not always equal strict-pass rate. Final scores here equal strict-pass rates. Unknown grades are not known wrong answers.

| LR | Group | Initial quality | Final quality | Change (pp) | Final passes | Graded | Updates | Reserved USD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 5e-6 | 4 | 18.75% | 31.25% | +12.50 | 10/32 | 29/32 | 12 | $63.12 |
| 5e-6 | 8 | 20.83% | 12.50% | -8.33 | 4/32 | 27/32 | 9 | $68.15 |
| 1e-5 | 4 | 27.08% | 15.62% | -11.46 | 5/32 | 27/32 | 2 | $32.84 |
| 1e-5 | 8 | 18.75% | 18.75% | +0.00 | 6/32 | 28/32 | 7 | $59.61 |

The LR1e-5/group4 row is compromised and must not be interpreted as an algorithm-quality result. Its final evaluation completed after the billing top-up, but training had already consumed the scheduled attempt count with failures.

## What the comparisons support

- **Group size at LR5e-6:** group4 exceeded group8 by18.75percentage points in final quality; descriptive paired-task bootstrap95% interval [+3.125,+34.375] points. Improvement relative to each initial evaluation favored group4 by20.83points, interval [+5.21,+37.5]. Even the extreme missing-grade bounds retain a3.125point advantage on this fixed set. This is directional evidence for group4 in this small screen, despite incomplete grading and no repeated seeds or multiplicity adjustment.
- **Learning rate at group8:** LR1e-5 exceeded LR5e-6 by6.25points; interval [−6.25,+18.75]. Difference in improvements +8.33points, interval [−10.42,+25]. This does not resolve the better learning rate.
- **Learning rate at group4 and group size at LR1e-5:** cannot be fairly interpreted because the LR1e-5/group4 training run was compromised. An apparent interaction is inseparable from that failure; do not pool it into an average LR/group effect.

The practical provisional choice is5e-6/group4 because it has the highest observed quality among the substantially completed training arms. Establishing a winner requires a clean comparison of the compromised arm, reliable symmetric grading and replicated validation; none were launched by this read-only analysis. The85-task confirmation cohort remains untouched.

## Training, tokens and cost

All arms scheduled16batches of8trajectories (128attempts), fresh Qwen3.5-4B rank8 seed42, same v6 data/v7 judge and episode limits. Group4 used2questions/batch (32unique questions); group8 used1 (16unique questions). Thus group size trades question diversity for repeated attempts at fixed rollout count. Matched task multiplicities across learning rates, evaluation membership and data/environment/grader identities were verified. Generated token counts differ despite equal worst-case rollout budgets.

| LR/group | Resolved training | Contributing trajectories | Zero-variance groups | Excluded groups | Policy input tokens (train) | Policy output tokens (train) |
|---|---:|---:|---:|---:|---:|---:|
| lr-5e-6-group-4 | 128/128 | 60 | 16/32 | 0 | 1,233,381 | 30,373 |
| lr-5e-6-group-8 | 128/128 | 72 | 7/16 | 0 | 1,328,473 | 30,953 |
| lr-1e-5-group-4 | 24/128 | 8 | 4/32 | 26 | 342,212 | 7,267 |
| lr-1e-5-group-8 | 127/128 | 56 | 8/16 | 1 | 1,323,227 | 32,326 |

Total reservations: **$223.72** against the sweep's$1,280 cap. These are conservative ledger reservations, not invoices. Ledger reservation sums were checked. The compromised run's lower spend is not an efficiency win.

## Billing and completion audit

LR1e-5/group4 recorded104 infrastructure-error trajectories, only24resolved rewards,26excluded four-attempt groups, and2acknowledged optimizer updates. Its log contains31 HTTP402 billing-error messages. The other three logs have no HTTP402 mentions. Original controller remained alive during billing suspension, then completed after credits were added; failed attempts were not rerun. All four final run states and campaign results report successful process exit, which does not establish successful training or complete evaluation.

In the other arms, `budget_exhausted` is an episode termination label and is not by itself evidence of account-credit exhaustion. LR1e-5/group8 had one unresolved training reward and excluded its8-attempt group; the two5e-6 arms resolved all128training rewards. Final answer completion was32/32 except LR1e-5/group4 (30/32).

## Uncertainty and missing grades

Final grading coverage:84.375%–90.625%, below95% in every arm. Missing-grade score bounds, assigning each unresolved score anywhere in[0,1]:

- lr-5e-6-group-4: [31.250%, 40.625%].
- lr-5e-6-group-8: [12.500%, 28.125%].
- lr-1e-5-group-4: [15.625%, 31.250%].
- lr-1e-5-group-8: [18.750%, 31.250%].

[comparison.json](comparison.json) includes all four prespecified paired contrasts, bootstrap intervals, and complete-case sensitivity using tasks with all four relevant initial/final grades resolved. Intervals resample32paired tasks10,000times with seed42; they are descriptive and unadjusted for multiple comparisons. They omit repeated-generation/judge uncertainty and do not establish generalization from one seed. No grade was changed, no additional evaluation called and no new job launched.

## Artifacts

- [Structured comparison](comparison.json)
- [Archive manifest](archive.json) and [file hashes](archive-checksums.json)
- [Launch receipt](submission.json)
- Download: `../../artifacts/lr-group-v6-sweep-v1-results`

- [W&B: lr-5e-6-group-4](https://wandb.ai/nmdatar-harvard-university/repository-qa-training/runs/lr-5e-6-group-4-v6-sweep-v1-seed42)
- [W&B: lr-5e-6-group-8](https://wandb.ai/nmdatar-harvard-university/repository-qa-training/runs/lr-5e-6-group-8-v6-sweep-v1-seed42)
- [W&B: lr-1e-5-group-4](https://wandb.ai/nmdatar-harvard-university/repository-qa-training/runs/lr-1e-5-group-4-v6-sweep-v1-seed42)
- [W&B: lr-1e-5-group-8](https://wandb.ai/nmdatar-harvard-university/repository-qa-training/runs/lr-1e-5-group-8-v6-sweep-v1-seed42)
