# Original three-tool correctness GRPO: 15 iterations

[Live plots](https://wandb.ai/nmdatar-harvard-university/repository-qa-training/reports/Three-tool-GRPO-15-iterations---correctness-and-timing--VmlldzoxODAyOTE2Nw==) · [Run](https://wandb.ai/nmdatar-harvard-university/repository-qa-training/runs/structured-correctness-grpo-15-v1)

Submitted to isolated Modal controller `sb-IYJN6HTyf8n6YuFX0YMMaM`. This is a fresh base-model experiment, separate from the existing bash-only run. Submission is not completion; check the live run for current progress.

- Qwen/Qwen3.5-4B, rank-8 LoRA, seed 42, GRPO learning rate 1e-5.
- 15 attempted batches, up to 15 optimizer updates; batch size 8, group size 8; 960 planned training attempts. A batch with no contributing trajectories cannot update the model.
- Original `list_files`, `search_code`, `read_file` tools; action aliases and read pagination retained.
- Same frozen Qwen3.5-397B-A17B judge and `correctness-only-v1` reward as the bash pilot. Citations remain diagnostic and do not reduce reward. Required-fact coverage drives reward; extra incorrect claims have no separate penalty.
- Same 32 training tasks cycle through shuffled epochs. Fixed, disjoint 32-case validation at optimizer steps 0, 3, 6, 9, 12, 15, with final evaluation if training finishes at another step. Confirmation cohort untouched.
- Training temperature 1; evaluation temperature 0. Training and evaluation reward levels are not directly comparable.
- 30 tool calls, 31 generations, 6,000 total generated tokens, 512 tokens per generation, 65,536-token context, 300-second rollout limit.
- Six-hour controller timeout; conservative reservation ceiling $4,000, estimated upper reservation $3,998.82. Reservations are not actual provider billing.

## What to watch

The live report contains 20 panels in four sections:

1. **Correctness and citation diagnostics:** training reward and EMA, fixed-cohort evaluation reward, separate citation scores. Read evaluation correctness first to assess learning.
2. **Tool use and rollout latency:** tool execution seconds, tool calls, and rollout latency, separately for training and evaluation. Tool seconds are total execution time per attempt, not average latency of an individual call.
3. **Iteration and evaluation wall time:** batch duration, collection plus grading, optimizer update, checkpoint save, and evaluation duration. Batch duration excludes periodic evaluation. Concurrent rollout times should not be summed to infer batch wall time.
4. **Scoring health and sampling cost:** scoring coverage, excluded/zero-variance groups, contributing trajectories, input/output token means, and tool timing measurement coverage. Unresolved grades are excluded rather than treated as zeros.

Charts populate when measurements arrive. Training charts use attempted batch count; evaluation charts use optimizer step. Raw trajectories and judge viewers are uploaded on completion.

## Reproduction and status

Configuration: `configs/experiments/structured-correctness-grpo-15-v1/run.json`; task IDs: adjacent `subset.json`. Submitted immutable bundle: `artifacts/structured-correctness-grpo-15-v1-launch-bundle`. The earlier `-bundle` folder was never submitted; the launch bundle includes the explicit isolated budget allocation.

Read live status with `.venv-eval/bin/python scripts/inspect_structured_correctness_grpo.py`. The submission receipt records the isolated volume for result retrieval. Do not resubmit an existing receipt or replay an interrupted optimizer update.

Validation before launch: 18 correctness, timing, and efficiency tests passed; submitted bundle verified against the pilot's judge, reward, and limits.
