# Experiment 1 — protocol-v4 results

**Status: stopped at the baseline coverage gate; the full throughput study is incomplete.** All 149 prescribed baseline episodes finished. The eight throughput arms (512 episodes) were not launched. No training or optimizer updates occurred.

The procedure requires at least 95% scoring coverage (31/32 selection, 81/85 confirmation) before a reliable comparison. The initial selection and confirmation both failed. The single preplanned selection repeat was retained to quantify variability; it does not replace the first result. No questions were removed, no unresolved score was converted to a zero training reward, and no grader, rubric, protocol or budget was changed to force a pass.

## Baseline results

| Cohort | Scored | Coverage | Completed answers | Demonstrated quality | Resolved mean | Wall time |
|---|---:|---:|---:|---:|---:|---:|
| Selection | 28/32 | 87.50% | 30/32 (93.75%) | 12.50% | 14.29% | 77.34 s |
| Confirmation | 76/85 | 89.41% | 79/85 (92.94%) | 12.94% | 14.47% | 735.77 s |
| Selection repeat | 31/32 | 96.88% | 29/32 (90.62%) | 6.25% | 6.45% | 73.15 s |

Demonstrated quality is the sum of resolved quality scores divided by all assigned tasks. Missing grades remain null; they contribute no demonstrated points. This is an uncalibrated reference-grader endpoint, not an accepted-answer rate or independently verified correctness. All observed resolved scores were 0 or 1.

The same unchanged Qwen/Qwen3.5-4B policy was evaluated at temperature 0 with Qwen/Qwen3.5-397B-A17B, protocol experimental-reference-v4, four judge slots and 32 rollout slots. Inference/grader scheduling can still vary. The frozen dataset contains 858 training and 117 development tasks; selection and confirmation are task-disjoint but share repository families. These are repurposed benchmark questions, not an untouched final test set.

## Repeat and unresolved sensitivity

The repeat changed demonstrated quality from 12.50% to 6.25%. Both runs resolved 27 matched tasks; their mean repeat-minus-initial difference was -0.0741. These are repeated observations of the same model, not learning gains. No confidence or noninferiority claim is made.

- Selection: assigning unresolved tasks anywhere in [0,1] gives an overall-quality sensitivity range of 12.50%–25.00%.
- Confirmation: assigning unresolved tasks anywhere in [0,1] gives an overall-quality sensitivity range of 12.94%–23.53%.
- Selection repeat: assigning unresolved tasks anywhere in [0,1] gives an overall-quality sensitivity range of 6.25%–9.38%.

## Why coverage failed

The following counts describe failures, not corrections to confirmation questions. Confirmation contents were not used to tune the protocol.

| Failure | Episodes |
|---|---:|
| Evidence exceeds judge context budget; no silent truncation | 4 |
| Semantic assessment requires adjudication | 7 |
| Supported assertion lacks verified evidence keys | 1 |
| Truncated full-claim audit | 2 |

[Detailed unresolved records](/Users/ndatar/Documents/ChatGPT/action-interview/reports/experiment-01-v4/unresolved.json). Selection-only diagnosis found ambiguity about the reference/source support, an evidence payload exceeding the fixed byte limit, and a supported assertion without verified evidence keys. Reference ambiguity cannot safely be repaired by automatically assigning a score. Truncation and evidence-size failures require a versioned evidence/grading design and fresh controls.

## Runtime and cost

| Cohort | Completed/min | Scored/min | Output tokens/s | Latency p50 / p95 | Episode reservations |
|---|---:|---:|---:|---:|---:|
| Selection | 23.27 | 21.72 | 104.52 | 37.72 / 71.62 s | $4.2140 |
| Confirmation | 6.44 | 6.20 | 29.14 | 285.04 / 414.38 s | $10.8877 |
| Selection repeat | 23.79 | 25.43 | 109.29 | 39.16 / 69.33 s | $4.1540 |

Reported wall time runs from evaluation dispatch through the final episode’s grading and cleanup; model/tokenizer/controller startup and W&B finalization are outside this timer. Startup was not separately instrumented, so no cold-start comparison is claimed. Latencies include grading and its queue; the 300-second action-loop limit is not a deadline on subsequent grading.

| Cohort | Provisioning sum | Generation sum | Tools sum | Verification sum | Judge queue sum | Judge sampling sum |
|---|---:|---:|---:|---:|---:|---:|
| Selection | 77.04 s | 244.40 s | 41.08 s | 897.46 s | 625.85 s | 257.58 s |
| Confirmation | 144.36 s | 620.93 s | 97.14 s | 20338.13 s | 17413.01 s | 2881.02 s |
| Selection repeat | 72.52 s | 314.71 s | 41.64 s | 816.82 s | 555.71 s | 244.52 s |

| Cohort | Invalid actions | Episodes with failed citation links / audited | Tool/provider error terminations | Tracking failures |
|---|---:|---:|---:|---:|
| Selection | 3 | 12/26 | 0 | 0 |
| Confirmation | 9 | 36/66 | 0 | 0 |
| Selection repeat | 4 | 14/26 | 0 | 0 |

Citation counts include only episodes with a semantic citation audit; missing audits are not treated as successful citations. No final episode ended with an infrastructure-error or agent-error termination.

Stage values sum work across overlapping episodes and must not be added to obtain coordinator wall time. Judge queue and sampling are included within verification. Cleanup is included in per-episode timing; the repository sandbox is closed before grading.

New Experiment 1 reservations: **$23.8141**, including controller allowances. Prior ledger reservations: **$37.8298**. Cumulative Experiment 1 ledger: **$61.6440 / $300**, leaving **$238.3560** under that cap. These conservative reservations are not provider invoices. Actual billing remains unavailable; image builds and persistent-volume storage are not fully metered by this ledger and use the project infrastructure reserve. The $1,000 project allocation was not increased.

Prices were rechecked against [Tinker’s model catalog](https://tinker-docs.thinkingmachines.ai/tinker/models.json) and [Modal pricing](https://modal.com/pricing); they matched the frozen configuration. [Final ledger snapshot](/Users/ndatar/Documents/ChatGPT/action-interview/reports/experiment-01-v4/ledger-final.json).

## Logs and reproducible artifacts

- **Selection:** [W&B](https://wandb.ai/nmdatar-harvard-university/repository-qa-training/runs/01-baseline-qwen4b-qwen397-seed42-all-claims-v3-modal-protocol-v4-eval-d825914d) · [controller log](/Users/ndatar/Documents/ChatGPT/action-interview/artifacts/experiment-01-v4-results/campaigns/741d78a87bfa4bf2ae9b68ad1e5d7498a729283edb0df3d7602629616e6c8a1d/01-baseline-qwen4b-qwen397-seed42-all-claims-v3-modal-protocol-v4.log) · [evaluation JSON](/Users/ndatar/Documents/ChatGPT/action-interview/artifacts/experiment-01-v4-results/artifacts/experiments/01-baseline-qwen4b-qwen397-seed42-all-claims-v3-modal-protocol-v4/evaluations/unchanged-base-28bbbd5eed6648ecaca3dae337025667.json) · [events](/Users/ndatar/Documents/ChatGPT/action-interview/artifacts/experiment-01-v4-results/artifacts/experiments/01-baseline-qwen4b-qwen397-seed42-all-claims-v3-modal-protocol-v4/events.jsonl) · [answers](/Users/ndatar/Documents/ChatGPT/action-interview/artifacts/experiment-01-v4-results/artifacts/experiments/01-baseline-qwen4b-qwen397-seed42-all-claims-v3-modal-protocol-v4/answers.json)
- **Confirmation:** [W&B](https://wandb.ai/nmdatar-harvard-university/repository-qa-training/runs/01-baseline-qwen4b-qwen397-seed42-confirmation-all-claims-v3-modal-protocol-v4-eval-dd22000d) · [controller log](/Users/ndatar/Documents/ChatGPT/action-interview/artifacts/experiment-01-v4-results/campaigns/741d78a87bfa4bf2ae9b68ad1e5d7498a729283edb0df3d7602629616e6c8a1d/01-baseline-qwen4b-qwen397-seed42-confirmation-all-claims-v3-modal-protocol-v4.log) · [evaluation JSON](/Users/ndatar/Documents/ChatGPT/action-interview/artifacts/experiment-01-v4-results/artifacts/experiments/01-baseline-qwen4b-qwen397-seed42-confirmation-all-claims-v3-modal-protocol-v4/evaluations/unchanged-base-1375aa831859450fb2b3e1d9be0328c0.json) · [events](/Users/ndatar/Documents/ChatGPT/action-interview/artifacts/experiment-01-v4-results/artifacts/experiments/01-baseline-qwen4b-qwen397-seed42-confirmation-all-claims-v3-modal-protocol-v4/events.jsonl) · [answers](/Users/ndatar/Documents/ChatGPT/action-interview/artifacts/experiment-01-v4-results/artifacts/experiments/01-baseline-qwen4b-qwen397-seed42-confirmation-all-claims-v3-modal-protocol-v4/answers.json)
- **Selection repeat:** [W&B](https://wandb.ai/nmdatar-harvard-university/repository-qa-training/runs/01-baseline-qwen4b-qwen397-seed42-repeat-all-claims-v3-modal-protocol-v4-eval-74c3249d) · [controller log](/Users/ndatar/Documents/ChatGPT/action-interview/artifacts/experiment-01-v4-results/campaigns/f62692a5a3d0b3f932c6bd95fbfd016d42d5b559cde79f7b9252e74cb9744f1c/01-baseline-qwen4b-qwen397-seed42-repeat-all-claims-v3-modal-protocol-v4.log) · [evaluation JSON](/Users/ndatar/Documents/ChatGPT/action-interview/artifacts/experiment-01-v4-results/artifacts/experiments/01-baseline-qwen4b-qwen397-seed42-repeat-all-claims-v3-modal-protocol-v4/evaluations/unchanged-base-28aefb4685084535823c201a649f1280.json) · [events](/Users/ndatar/Documents/ChatGPT/action-interview/artifacts/experiment-01-v4-results/artifacts/experiments/01-baseline-qwen4b-qwen397-seed42-repeat-all-claims-v3-modal-protocol-v4/events.jsonl) · [answers](/Users/ndatar/Documents/ChatGPT/action-interview/artifacts/experiment-01-v4-results/artifacts/experiments/01-baseline-qwen4b-qwen397-seed42-repeat-all-claims-v3-modal-protocol-v4/answers.json)

[Per-task CSV](/Users/ndatar/Documents/ChatGPT/action-interview/reports/experiment-01-v4/per-task.csv) · [Full metrics JSON](/Users/ndatar/Documents/ChatGPT/action-interview/reports/experiment-01-v4/metrics.json) · [Frozen specification](/Users/ndatar/Documents/ChatGPT/action-interview/reports/experiment-01-v4/specification.json) · [Artifact checksums](/Users/ndatar/Documents/ChatGPT/action-interview/reports/experiment-01-v4/checksums.json)

W&B Tables → answers contains submissions, rewards, grading status/reason, phase and optimizer step. Private raw judge responses and full trajectories are retained in the local downloaded artifact directory and the private Modal volume `repository-qa-training-state-v2`.

## Decision

Do not advance to GRPO or select a throughput winner from these results. Answer completion passed 90% on the initial cohorts, but scoring coverage failed 95%. The provisional 32-rollout setting is therefore not a benchmark-selected concurrency. No 16-task × 4-attempt update-cost estimate or full-dataset runtime extrapolation is justified without the unrun repeated throughput arms.

A continuation needs an independently justified, versioned grader/evidence repair and fresh baseline controls while preserving the failed runs. The frozen cohorts must not be edited based on these scores. The original full Experiment 1 remains incomplete; this report records a completed baseline attempt with a failed readiness gate.
