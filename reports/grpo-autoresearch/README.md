# GRPO autoresearch results — 29 September 2026

The runtime performs real adapter updates and saves checkpoints. Pagination improved tool reliability in a small benchmark. The follow-up GRPO run completed, but **we have not demonstrated a reliable improvement over the base model**. Factual coverage was nearly unchanged; strict passes declined. All original results are preserved and the confirmation set remains untouched.

## Completed learning run

[W&B training curves and evaluations](https://wandb.ai/nmdatar-harvard-university/repository-qa-training/runs/autoresearch-grpo-paginate-v2-seed42)

Run `autoresearch-grpo-paginate-v2-seed42` used fresh Qwen3.5-4B, rank-8 LoRA, learning rate 1e-5, seed 42, two tasks × four attempts per batch, and the frozen v7 grader. The only environment change was versioned read pagination. Ten batches / 80 attempts produced **7 acknowledged optimizer updates**, with a checkpoint after each update. There were 20 groups: 10 had zero reward variance, none were excluded for unresolved grading, and 40 trajectories contributed to updates. Batch reward remained noisy; its overall mean was 0.2329.

| Selection-set measure | Base | Final |
|---|---:|---:|
| Independent factual coverage, 32/32 resolved | 0.59896 | 0.61458 |
| Full factual credit | 18/32 | 18/32 |
| Positive factual credit | 21/32 | 22/32 |
| Original strict passes | 11/32 | 6/32 |
| Original strict scoring coverage | 30/32 | 31/32 |

Paired factual delta: **+0.015625**, with task-bootstrap 95% interval **[-0.03125, +0.0625]**. Two tasks improved, one worsened, 29 tied. This is one small selection-set diagnostic using saved answers, not confirmation evidence. The interval does not account for repeated generation or judge variability. No missing factual grades were dropped: all 32 pairs resolved, including deterministic zeros for three empty answers.

[Training summary](training-pagination-summary.json), [factual summary](factual-v2/summary.json), [paired scores](factual-v2/paired-results.csv), [raw factual judgments](factual-v2/results.json).

Final checkpoint: `ckpt-d0093c8e5e294bb1a4cd02aacddf8503`. All checkpoint records, final sampler archive, trajectories, events and reward CSV are in [the local run archive](../../artifacts/grader-paraphrase-validation-results/artifacts/experiments/autoresearch-grpo-paginate-v2-seed42/). The initial strict evaluation was below the 95% eligibility threshold, so an eligible “best” checkpoint label must not be interpreted as outperforming the initial score.

## Environment fix and validation

Oversized source reads previously failed outright. Optional `tool_read_policy=paginate-v1` now returns at most 120 lines and a continuation position. Invalid ranges and unrecognized paths remain errors; source hashes and sandbox restrictions remain enforced. Legacy behavior remains the default. Config identity and immutable experiment bundles capture the changed condition.

On the same eight predetermined TRAINING tasks, with four attempts each:

| Measure | Original reads | Paginated reads |
|---|---:|---:|
| Completed attempts | 25/32 | 28/32 |
| Tool execution failures | 5 | 1 |
| Mean factual reward | 0.05469 | 0.15625 |
| Contributing groups | 2/8 | 2/8 |
| Truncated outputs | 16 | 30 |

This small stochastic comparison supports keeping pagination, but does not isolate a causal learning effect. Output clipping remains unresolved. [Comparison](pagination-comparison.json), [516 passing tests](tests.log), [14 focused pagination tests](pagination-tests.log).

The first pagination training run failed before any update because strict baseline scoring coverage was 30/32, below the automatic regression gate. A separately named v2 run removed that gate, consistent with the user's earlier acceptance of unresolved strict grades. It retained explicit coverage reporting and 95% best-checkpoint eligibility. The failed run and its costs remain recorded; no optimizer call was replayed.

## Remaining bottlenecks and judge evidence

- The latest 80 training attempts had 65 completions, 15 budget exhaustions, 54 action errors, 87 clipped tool outputs, and only one tool execution failure. Of 21 invalid-tool errors, eight used `action` instead of `tool`, ten omitted the tool name entirely. These are protocol failures worth testing independently; missing tool names should not be guessed. [Patterns](invalid-tool-patterns.json).
- **78/858 training tasks have only a generic placeholder claim**, rather than explicit answer facts. The factual grader's request supplies that claim and source evidence without the original reference answer. This leaves the criterion underspecified. Source/reference inputs are staged in [placeholder-repair-inputs.json](placeholder-repair-inputs.json), explicitly inactive. No task or grader was changed to improve selection scores.
- Repeated untrained evaluations under the same configuration gave 6 versus 11 strict passes. Answer text differed on 23/32 tasks; seven scores differed, including one on identical answer text. This establishes substantial combined generation/judge variability, not a clean estimate of either component alone. [Audit](baseline-repeat-variability.json).
- Six initial strict passes became final failures; all six final answers received factual coverage 1. Five had citation/support-related rejection reasons. The sixth was marked materially false for attributing code to `execute_notebook`. **The saved pinned source explicitly defines `execute_notebook` at line 26 and contains the quoted docstring at lines 35–36.** The strict judge's inferred alternative name was wrong. [Evidence](pybryt-judge-error.json), [saved source](pybryt-judge-error-source.txt), [all six original assessments](factual-v2/strict-declines.json). This diagnoses one rejection rationale; it does not authorize relabeling the original score or assume every other rejected answer is correct.

The earlier main run's seven factual declines included both real answer omissions/contradictions and inconsistent grading. Those original findings remain in [worsened-pairs.txt](worsened-pairs.txt) and [the earlier comparison](../grpo-factual-comparison/README.md). The latest result is less negative, but the runs are too small and variable to establish that pagination improved learning.

## Next experiment, after this time window

1. Repair generic TRAINING claims into source-backed atomic facts and verify equivalence to the original question/reference. Freeze a new data version. Use independent training-derived examples to test judge behavior, including named-function attribution and combined citation support; do not tune on these selection answers.
2. Test protocol improvements separately on the existing training benchmark. Consider a narrowly versioned `action` alias or a small verified tool-use warmup. The [28 warmup candidates](warmup-candidates.json) are unreviewed candidates, not admitted training data. If SFT is used, compare SFT against SFT+GRPO so improvements are not falsely attributed to RL.
3. Once protocol reliability and reward variance are adequate, run a bounded longer GRPO stage with larger effective batches, fixed evaluation conditions, and repeated baseline measurements. Choose checkpoints using selection data, then evaluate the untouched confirmation set once. Retain both strict and factual scores, scoring coverage, contribution counts, and checkpoints. More batches alone do not guarantee improvement.

## Provenance, spending and logs

All paid experiments in this investigation have completed or failed with their outputs preserved. No run is active as of 12:24 UTC. No new paid work is planned before the 12:30 deadline.

Latest authoritative GRPO reservation ledger: **$225.63509 / $550**; baseline ledger **$257.27584 / $300**. The autoresearch increment from the recorded starting GRPO ledger is **$88.47626**. These are safety-ledger reservations, not an invoice or measured provider bill. The $1000 project cap and all allocations are unchanged.

| Run | Frozen bundle | Outcome |
|---|---|---|
| autoresearch-read-strict-v1 + autoresearch-read-paginate-v1 | b6227b5f8ceb59889d0a0fe846e357c19f65280f9cab55f3feca2fbf9e0f6094 | Both benchmarks complete |
| autoresearch-grpo-paginate-v1-seed42 | 6e09a34e1ffa6cc97916505e1d67fca9f391f254d07e8d51c13e3b1087d7aa14 | Failed before updates |
| autoresearch-grpo-paginate-v2-seed42 | 8d34ad911e03dc4931e1f4c35b8c3a66d8959507a35ad97f516acfe0acda6c94 | Complete: 10 batches, 7 updates |
| autoresearch-grpo-pagination-factual-v1 | ea0f2b2bbc5a0342c598c808d6bdc9967f5487fa2e7aab58e5d70e0ae8fec709 | Complete: 32 factual pairs |

[Benchmark receipt](submission.json), [failed training receipt](train-pagination-submission.json), [completed training receipt](train-pagination-v2-submission.json), [factual receipt](factual-v2/submission.json), [training events](train-v2-events.jsonl).

Remote artifacts: Modal volume `repository-qa-training-state-v2`, `artifacts/experiments/<run-id>/`. Campaign statuses are under `campaigns/<bundle-id>/status.json`. Local copies are under `artifacts/grader-paraphrase-validation-results/`.

At **12:30 UTC**, the investigation automation was paused successfully. No paid experiment remained active. A final integrity check verified all eight checkpoint records and the final model archive.

## Subsequent user-requested fixes

The user resumed work after the timed window. See [implemented fixes, challenges and live validation](fixes-v1/CHALLENGES.md): 78 training placeholder rubrics replaced by 316 explicit claims, optional tool-action alias and verified definition context, 523 passing local tests, and 12/12 successful training-only grading controls. Historical scores above are unchanged. No new policy training has been launched.
