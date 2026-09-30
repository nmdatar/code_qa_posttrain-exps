# Running-baseline REINFORCE comparison

User authorized implementation and launch. Comparator: completed grpo-long-v6-seed42-v1. Fresh Qwen3.5-4B, seed42, rank8, LR1e-5; exact same 16 batches x2 tasks x4 attempts, unchanged v6 rubrics/v7 grader, environment limits, fixed initial/final32 selection evaluation. Confirmation untouched. This is a sequential matched single-seed comparison, not a simultaneous randomized experiment. Same task shuffle; provider samples can vary.

REINFORCE prior-mean-v1: advantage = reward minus cumulative mean over eligible trajectories from strictly earlier batches, starting at zero. No per-question centering or standard deviation normalization. After computing current advantages, update reward sum/count for subsequent batches. Baseline is checkpointed and resumes exactly. Complete unresolved groups are excluded under the same GRPO admission rule; equal groups may contribute. Preserve assistant-only loss masks and existing per-trajectory token averaging / contributing-trajectory normalization. Same nominal LR does not ensure equal effective gradient scale; this is a recipe comparison, not a tuned claim of algorithm superiority.

Primary: within-run change in demonstrated strict credit and comparison of that change with GRPO. Also compare final strict passes, resolved coverage, completion, task-matched outcomes, zero-variance groups, contributing trajectories, updates, costs and runtime. Reward curves use changing tasks and do not establish generalization. Initial GRPO evaluation had28/32 resolved; preserve unresolved values. Do not tune on confirmation.

Estimator correction: v7 evaluation runs extract+assess, while training additionally runs independent coverage. Tinker rejects prompt + max_output > context before reservation, so maximum judge prompt is context minus configured output. Mechanical repair allowance remains included for each stage. Benchmark training inputs retain three stages. Tests cover both bounds. No grading behavior changed.

Conservative incremental reservation $288.814942; prior GRPO ledger $362.569454; combined $651.384395. Previously authorized allocation rebalanced to GRPO $655, baseline $260, LR $15, reserves $70, total $1000. Runtime estimate25–40minutes, hard timeout2hours.

## Verification and launch

540 workspace tests passed; 85 tests passed against the frozen bundle. Exact comparator collection.py and training_runner.py are retained in the bundle to exclude unrelated opt-in harness edits; grader/solver hash audit is in matched-source.json. All 324 dataset/repository bundle input files match the previously authorized GRPO upload. Comparison report script passes an identical-run zero-difference check.

Submitted successfully: Modal sandbox `sb-PWMHLwFWL48plPAQhRni42`; bundle `f0716c3c69a836d8d2e6873e7f65bed194d7d6081aeb52e4b6ee047637710876`. Five-minute monitoring enabled through heartbeat follow-longer-grpo-run, now named Compare REINFORCE with GRPO.
