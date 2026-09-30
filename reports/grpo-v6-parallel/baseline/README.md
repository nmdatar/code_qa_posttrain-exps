# Fresh v6 baseline

Completed 32 selection tasks using unchanged Qwen3.5-4B weights, with no trainer/optimizer/checkpoint. Same v6 release and environment/grader policies as the parallel smoke; confirmation untouched.

- Strict full passes: **7/32**; total strict credit 7.0.
- Strict scoring coverage: **28/32 (87.5%)**. Four unknown grades are preserved, not silently counted as judged failures.
- Strict mean among resolved: 0.25; demonstrated strict quality over all attempts: 0.21875.
- Episodes:31 completed, 1 budget exhausted.
- Independent factual coverage: not measured yet. The symmetric saved-answer comparison was prepared but not dispatched because the separate SFT controller is active. See ../factual/submission.json.

[W&B](https://wandb.ai/nmdatar-harvard-university/repository-qa-training/runs/grpo-v6-baseline-seed42-eval-1fae5e0b), [summary](summary.json), [grading issues](grading-issues.json), [config](../../../configs/experiments/grpo-v6-parallel/baseline-v1.json), [procedure](../PROCEDURE.md).

Raw baseline artifacts are archived under `artifacts/grader-paraphrase-validation-results/artifacts/experiments/grpo-v6-baseline-seed42/`. Per-arm cost counter deltas include concurrent smoke spend; authoritative joint cost is reported after the campaign finishes.
