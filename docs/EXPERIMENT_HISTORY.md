# Retired experiments and preserved history

The complete pre-cleanup checkout is pinned at [`2b70198df1db`](https://github.com/nmdatar/action-interview/tree/2b70198df1db7b3028448d8f05772e55c52ea6fc). Files were removed from `main` to simplify navigation, not to hide unfavorable results. Existing experiment branches also remain available.

Use the [research progression](../experiments/README.md) for substantive comparisons and the [exact retirement manifest](EXPERIMENT_RETIREMENT.json) for every removed path, its SHA-256, and dependency exceptions.

## Early readiness and small screens

149 baseline episodes completed, but initial selection and confirmation missed the 95% scoring-coverage gate; the throughput study was incomplete.

**Why retire from main:** Superseded protocol and readiness inputs obscure the later versioned suites. Keep the failed gate visible; use campaign-v8 designs and the retained v6 baseline for subsequent study context.

[Original configs](https://github.com/nmdatar/action-interview/tree/2b70198df1db7b3028448d8f05772e55c52ea6fc/configs/experiments/ready-v4) · [Original evidence](https://github.com/nmdatar/action-interview/tree/2b70198df1db7b3028448d8f05772e55c52ea6fc/reports/experiment-01-v4)

## Campaign v7

A prepared version-aligned suite, not evidence that every arm ran.

**Why retire from main:** Campaign v8 supersedes these planning inputs. Removing duplicate entry points avoids conflicting instructions about which campaign is current.

[Original configs](https://github.com/nmdatar/action-interview/tree/2b70198df1db7b3028448d8f05772e55c52ea6fc/configs/experiments/current-v7) · [Original evidence](https://github.com/nmdatar/action-interview/tree/2b70198df1db7b3028448d8f05772e55c52ea6fc/reports/experiment-current-v7)

## Reward revisions v1–v4

The v4 calibration matched 4/7 expected scores and resolved 6/7; it blocked further rollouts. Earlier revisions investigated citation and evidence-routing failures.

**Why retire from main:** Atomic-claim and paraphrase comparisons preserve the substantive evaluator story. Old diagnostic launch bundles are not recommended run recipes; reward implementations and regression tests remain.

[Original configs](https://github.com/nmdatar/action-interview/tree/2b70198df1db7b3028448d8f05772e55c52ea6fc/configs/experiments/reward-v4) · [Original evidence](https://github.com/nmdatar/action-interview/tree/2b70198df1db7b3028448d8f05772e55c52ea6fc/reports/reward-shaping-v4)

## GRPO pilot v7

Four attempted batches were a machinery/signal pilot; the later main-run report records only two pilot updates under the older gate.

**Why retire from main:** A tiny pilot does not establish quality improvement. The longer GRPO and REINFORCE studies provide the main learning-signal comparison.

[Original configs](https://github.com/nmdatar/action-interview/tree/2b70198df1db7b3028448d8f05772e55c52ea6fc/configs/experiments/grpo-pilot-v7) · [Original evidence](https://github.com/nmdatar/action-interview/tree/2b70198df1db7b3028448d8f05772e55c52ea6fc/reports/grpo-pilot-v7)

## GRPO main v7

A bounded 12-batch design followed a citation-routing fix; archived launch/readiness records and saved training curves alone do not establish a final held-out result.

**Why retire from main:** This older reward/data campaign is superseded as a reader entry point by the retained longer GRPO study. Do not relabel it a successful or failed final quality comparison without final evidence.

[Original configs](https://github.com/nmdatar/action-interview/tree/2b70198df1db7b3028448d8f05772e55c52ea6fc/configs/experiments/grpo-main-v7) · [Original evidence](https://github.com/nmdatar/action-interview/tree/2b70198df1db7b3028448d8f05772e55c52ea6fc/reports/grpo-main-v7)

## Selected-task fast GRPO

Four updates completed; demonstrated strict score fell from 0.125 to 0.0625 on 16 validation tasks, with resolved coverage 15/16 to 14/16. Single-seed screen; no general improvement established.

**Why retire from main:** Keep this negative finding, but retire detailed monitoring/results clutter. The run and subset remain unchanged as frozen inputs to the Bash correctness generator.

[Original configs](https://github.com/nmdatar/action-interview/tree/2b70198df1db7b3028448d8f05772e55c52ea6fc/configs/experiments/fast-grpo-selected-v1) · [Original evidence](https://github.com/nmdatar/action-interview/tree/2b70198df1db7b3028448d8f05772e55c52ea6fc/reports/fast-grpo-selected-v1)

## Retrieval and dependency rules

For an individual retired file, use `git show 2b70198df1db7b3028448d8f05772e55c52ea6fc:PATH`. To reproduce a historical experiment, inspect its full pinned revision and original environment requirements; a historical checkpoint may have expired. Do not combine an old config with current code and call it an exact reproduction.

The manifest preserves the complete path list, including the original root baseline/throughput/LR screens, all retired reward revisions, and their dedicated scripts. The three dependency exceptions retain their original locations and bytes. Dedicated test fixtures replace tests’ incidental dependencies on campaign launch configs; the bounded remote validation example retains its original settings.

Historical cost audits, launch receipts, branch-stack manifests, and source-inheritance records outside the retired bundles remain unchanged. Paths inside those records describe the original checkout; resolve removed paths against the pinned revision above. These records are not current launch instructions.

The v6 baseline, SFT admission inputs, cohorts, budgets, and shared runtime implementations remain. Local untracked reports/artifacts and external W&B/Modal records are outside this cleanup.
