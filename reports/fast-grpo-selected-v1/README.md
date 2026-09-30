# Fast baseline versus GRPO on selected tasks

Qwen3.5-4B, rank 8, seed 42. Train on 32 tasks drawn by deterministic family-balanced sampling from the selected 274-task score band. All 31 training families are represented. Evaluate 16 fixed tasks from the existing 32-task validation cohort before and after four scheduled GRPO batches; the 85-task confirmation cohort remains untouched. The initial untrained checkpoint supplies the baseline. Each batch contains 8 tasks × 4 attempts; learning rate is 1e-5.

| Metric | Baseline | After GRPO |
|---|---:|---:|
| Optimizer step | 0 | 4 |
| Resolved validation grades | 15/16 | 14/16 |
| Strict passes | 2/16 | 1/16 |
| Demonstrated strict score (all 16 tasks) | 0.1250 | 0.0625 |

Acknowledged updates: **4** out of 4 scheduled. Recorded trajectories: **160**. Conservative reservations: **$48.74**, against a $250 cap; these are not invoices.

`paired-results.csv` keeps every validation task and unresolved grades. `summary.json` includes paired differences, the descriptive task-bootstrap interval, contributing trajectories and zero-variance/excluded groups. This tiny, single-seed screen cannot establish a general improvement. The train reward is factual coverage; the validation score also enforces strict correctness/citation gates. The grader is uncalibrated and related to the training-data difficulty model.

Efficiency: 32 rollout workers / 16 judges, no intermediate validation, one shared baseline/training setup, and final-only local viewers. Every raw trajectory was retained; slow live dashboard media uploads were disabled. `trajectories.html` exposes prompts, model outputs, tool calls, observations and final grades. Checkpoint manifests and private grading artifacts are retained in the local archive; remote sampler checkpoints use the configured two-day retention.

The subset manifest pins both task lists and the source selected pool. Held-out leakage, tampering, concurrency, optimizer batching and trace retention checks passed before launch (43 tests). Inputs and the remote code bundle are frozen for reproducibility; no other experiment was modified or resumed.
