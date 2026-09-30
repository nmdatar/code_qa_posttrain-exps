# Bash GRPO token-efficiency comparison

Treatment: fresh Qwen3.5-4B, rank 8, seed 42, GRPO LR 1e-5, 15 attempted batches, batch size 8, group size 8. Control is the existing bash-correctness-grpo-15-v1 run, with the same initialization, selected training tasks, eval cohort, judge, sampling and rollout limits. Reuse avoids a duplicate paid control. A batch can skip optimization if it contains no usable signal; report acknowledged updates separately.

Training reward is `correctness * (1 - 0.10 * min(output_tokens / 6000, 1))`. Counts cover all model-generated investigation and answer tokens. Zero correctness earns zero regardless of brevity; unresolved grades remain null. The reward retains at least 90% of correctness, but does not guarantee lexicographic correctness ordering for close scores. GRPO normalization can amplify small reward differences, especially when correctness ties.

Eval uses unpenalized correctness on the fixed 32 selection tasks, before training, every three updates, and final. Confirmation is not used. Charts track correctness, fully correct fraction, output/input tokens, bash calls, latency, tokens per fully correct answer, scoring coverage and completion. Cost ratios include all attempts. Unknown denominators are undefined. Latency includes provisioning, generation, grading and cleanup, and is diagnostic because concurrent services introduce load variance. Same-family model grading is not an independent human assessment.

Compare endpoint correctness and efficiency together and inspect baseline differences. A token reduction with lower correctness/completion/scoring coverage is not an established improvement. This is a single-seed exploratory comparison.

Local code validation: 78 tests passed across reward integration, metrics, tracking, config and training orchestration. Fresh provider prices checked during preparation. The conservative upper reservation estimate is $3,998.82 under a $4,000 ceiling; reservations are not actual invoices. Existing control spending is separate. The immutable bundle records source and inputs.

Use `scripts/compare_bash_efficiency_grpo.py --publish` to refresh the aggregate W&B comparison. It uploads aggregate eval metrics only; no answers or per-task source evidence are included by this publisher.

Frozen source comparison verified that the only changed source files are the reward, config validation, collection reward application, efficiency metrics and metric logging. Model, judge settings, environment, eval cohort, limits, stages, task subset, seed and concurrency match the control. See control-source-diff.txt.
