# 100-task solvability ranking pilot

Completed using 200 existing, hash-verified Qwen3.5-397B-A17B trajectories: two attempts for each of 100 training tasks across 31 repository families. The 100-task cohort was selected with the existing deterministic proportional-family sampler before examining scores. Model, tools, episode limits, rubric release, grader settings, reward and temperature match the recent stronger-model comparison. All 200 episodes share one grading-version identity. No new model calls or costs were needed.

## Ranking

`ranked-tasks.csv` is the easiest-to-hardest ranking, including questions, both attempt scores, family, strict passes, review flags, and selection membership. `ranked-tasks.json` additionally contains exact source trace paths and checksums. Score is mean factual assertion coverage across two attempts, not a calibrated probability of solving the task. Episode-limit failures retain the recorded zero score. Grading errors remain null.

98 tasks have complete scores. Two tasks have one unresolved judge result and remain unranked: `import-7318b2894e8550d08fc091e1` and `import-dd8293e32bae2047df308e4d`. Their possible mean-score bounds and known attempt scores are retained. They are excluded from both selections.

The distribution is coarse: 36 tasks score 1, 31 score 0, and 31 have scores strictly between 0 and 1. Equal scores have no evidenced difficulty ordering; a fixed seed-42 task hash makes boundary selections reproducible.

## Two exported selections

- **Requested percentile band:** `percentile-20-80-manifest.json` and `percentile-20-80-tasks.jsonl`. Trim ceil(20% × 98) = 20 tasks from each end, retaining ranks 21–78: **58 tasks**. Because of ties, this includes 16 tasks scoring 1 and 11 scoring 0. It implements rank trimming but does not reliably remove the observed easiest/hardest groups.
- **Recommended next-iteration subset:** `score-20-80-manifest.json` and `score-20-80-tasks.jsonl`. Keep mean scores inclusively between 0.2 and 0.8: **27 tasks**. This is a separate score-threshold selection, not a relabeling of the requested percentile band. It excludes the observed score extremes but has fewer examples and changes repository-family balance.

The JSONL exports preserve the original public tasks, source revision identifiers, budgets and tool contracts. Manifest IDs refer to the unchanged private references/rubrics in the source v6 release. These are selection manifests, not drop-in manifests for the throughput-only benchmark sampler. The original release, training split, and held-out confirmation set are unchanged; no curriculum was applied to the full dataset.

## Evidence and caveats

`trajectories.html` contains all 200 model investigations, tool calls, observations and answers. `sample-manifest.json` freezes cohort selection; `summary.json` records model/grader identity, source-audit checksum, score distributions and family counts. The script is `scripts/rank_task_difficulty_pilot.py`.

Ranking checks passed for direction, trimming, tie determinism and unresolved-grade exclusion. Both exported task sets were checked against manifest IDs and training split. This is a quick exploratory curriculum signal under tight episode limits, based on two stochastic attempts and an uncalibrated same-family judge. The previous comparison found strict-grader inconsistencies; strict review flags remain visible and the ranking uses factual coverage instead of strict pass count. Scores estimate performance under this harness, not intrinsic task difficulty or guaranteed trainability for the 4B student.
