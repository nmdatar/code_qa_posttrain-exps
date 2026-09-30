# Reward shaping and grader reliability — implementation and validation

**Implementation complete; no GRPO updates were run.** The final train reward is `supported-coverage-v3` and the final grader is `all-claims-v5`. Strict evaluation retains the original scoring criterion. All historical controls and unsuccessful diagnostics are retained.

## What changed

- Training credit comes only from supported required facts, with explicit penalties for contradictions, unsupported assertions, and citation defects. Scores are clipped to [0,1]. Extra correct prose earns no positive credit and cannot dilute penalties.
- Development evaluations and checkpoint selection use strict scores. Trajectories and answer tables retain strict score, training reward, and component breakdown separately.
- Overlapping evidence is compacted without dropping source lines or keys. Reference context and output budgets increased. At most one mechanical repair per judge stage is priced and logged; genuine uncertainty and valid unfavorable grades are not retried.
- Grader v5 checks extraction ranges against catalog line counts/hashes before source reads. The exact v4 failure now reaches bounded repair instead of immediately losing a whole group.
- Version 3 also audits nonempty uncited training answers against pinned source; their strict score stays zero. Empty answers and invalid citation metadata still receive deterministic zero.
- Unresolved results remain null and exclude the whole training group. The launcher rejects the retired v1 reward config.

See [reward contract and implementation details](/Users/ndatar/Documents/ChatGPT/action-interview/docs/REWARD_SHAPING.md).

## Frozen checks

480 tests passed in the full suite, plus a separately passed shared-verifier regression for uncited factual training credit. Frozen source-grounded synthetic examples rank correct=1.0, partial=0.5, incorrect=0, citation-defective=0.9, unsupported-extra=0.8, and unsupported-extra-plus-true-filler=0.8. False execution claims receive 0; unresolved judgments stay null. These checks validate reward arithmetic, not human calibration or live judge correctness.

An offline projection of the historical selection audits gave positive shaped credit to 11 of 20 strictly-zero answers with complete semantic audits. Four other zero answers had no semantic audit; unresolved answers remained null. This illustrates available feedback but does not establish on-policy group variation. No confirmation answers were used to tune the reward.

## All live runs

| Run | Grader / reward | Scored | Completed | Strict quality | Contributing groups (strict → training) |
|---|---|---:|---:|---:|---:|
| reward-shaping-v1-four-attempt-diagnostic | all-claims-v4 / supported-coverage-v1 | 16/16 | 10/16 | 6.25% | 1 → 4 (4 eligible) |
| reward-shaping-v1-selection-validation | all-claims-v4 / supported-coverage-v1 | 32/32 | 29/32 | 9.38% | selection evaluation |
| reward-shaping-v2-four-attempt-diagnostic | all-claims-v4 / supported-coverage-v2 | 15/16 | 13/16 | 0.00% | 0 → 0 (3 eligible) |
| reward-shaping-v2-grader-v5-four-attempt-diagnostic | all-claims-v5 / supported-coverage-v2 | 16/16 | 12/16 | 0.00% | 0 → 0 (4 eligible) |
| reward-shaping-v2-grader-v5-selection-validation | all-claims-v5 / supported-coverage-v2 | 29/32 | 29/32 | 15.62% | selection evaluation |
| reward-shaping-v3-grader-v5-four-attempt-diagnostic | all-claims-v5 / supported-coverage-v3 | 15/16 | 10/16 | 0.00% | 0 → 0 (3 eligible) |

The v1 prototype allowed negative scores. Its apparent 1→4 group improvement partly rewarded missing or uncited answers over false answers. It was rejected, and all its records were kept. The first nonnegative v2 check had no contributing groups and one out-of-bounds source request. That mechanical failure motivated grader v5; it was not hidden or resampled under the same identity. The next v2/v5 run still had all-zero groups: most uncited answers bypassed factual grading. Reward v3 closes that training-only bypass. Its strict control reuses the completed v5 selection run because grading and development routing are unchanged; routing regressions verify this separation.

Coverage counts both semantic audits and deterministic zero outcomes. Final training semantic audits: 9/16; final selection semantic audits: 27/32. Thus 100% scoring coverage would not imply every answer was semantically graded.

## Final decision

Small-diagnostic readiness: **not passed**.

- training_scoring_coverage: fail
- training_answer_completion: fail
- training_group_variation: fail
- selection_scoring_coverage: fail
- selection_answer_completion: pass

The final training run has 0 contributing groups, 1 excluded group(s), and 0 answers with strict zero but positive training credit. Do not launch GRPO on the assumption that shaping alone has solved the signal problem.

A shaped reward cannot create factual credit when attempts lack supported required facts. Valid cited-answer completion and sufficiently granular trustworthy references remain prerequisites. Single-claim references can still limit factual partial credit. The original full Experiment 1 throughput comparison and full fresh-control gates remain outside this bounded diagnostic.

## Remaining failures

The latest strict selection run retained two source/reference ambiguities and one invalid supported finding after its bounded repair. The final training diagnostic retained one judge-context overflow; its group was excluded. One other extraction request was successfully repaired. These are unresolved prerequisites, not scores to force into numerical rewards. Increasing context or changing reference evidence needs a newly versioned control; no such unvalidated change is silently included in the reported results.

## Cost and provenance

128 new episodes across all retained diagnostic versions; zero optimizer updates. New conservative reservations: **$23.8135**. Cumulative shared baseline ledger: **$85.4575/$300**. These are reservations, not actual invoices; the project ceiling remains $1,000. Every attempt, including failed grading and mechanical repairs, remains in the ledger.

[Final test log](/Users/ndatar/Documents/ChatGPT/action-interview/reports/reward-shaping-v3/tests-final.log) · [Results JSON](/Users/ndatar/Documents/ChatGPT/action-interview/reports/reward-shaping-v3/results.json) · [Per-attempt CSV](/Users/ndatar/Documents/ChatGPT/action-interview/reports/reward-shaping-v3/per-attempt.csv) · [Frozen fixtures](/Users/ndatar/Documents/ChatGPT/action-interview/reports/reward-shaping-v3/frozen-fixtures.json) · [Artifact checksums](/Users/ndatar/Documents/ChatGPT/action-interview/reports/reward-shaping-v3/checksums.json)

## Log links

- **reward-shaping-v1-four-attempt-diagnostic**: [W&B](https://wandb.ai/nmdatar-harvard-university/repository-qa-training/runs/reward-shaping-v1-four-attempt-diagnostic) · [controller log](/Users/ndatar/Documents/ChatGPT/action-interview/artifacts/reward-shaping-v1-results/campaigns/e558ce2772664af33a679fce127c70c77c8264ac35e5eec4dab5db6027fe7636/reward-shaping-v1-four-attempt-diagnostic.log) · [metrics](/Users/ndatar/Documents/ChatGPT/action-interview/artifacts/reward-shaping-v1-results/artifacts/experiments/reward-shaping-v1-four-attempt-diagnostic/benchmark.json) · [answers](/Users/ndatar/Documents/ChatGPT/action-interview/artifacts/reward-shaping-v1-results/artifacts/experiments/reward-shaping-v1-four-attempt-diagnostic/answers.json)
- **reward-shaping-v1-selection-validation**: [W&B](https://wandb.ai/nmdatar-harvard-university/repository-qa-training/runs/reward-shaping-v1-selection-validation-eval-7e2cd0b9) · [controller log](/Users/ndatar/Documents/ChatGPT/action-interview/artifacts/reward-shaping-v1-results/campaigns/e558ce2772664af33a679fce127c70c77c8264ac35e5eec4dab5db6027fe7636/reward-shaping-v1-selection-validation.log) · [metrics](/Users/ndatar/Documents/ChatGPT/action-interview/artifacts/reward-shaping-v1-results/artifacts/experiments/reward-shaping-v1-selection-validation/evaluations/unchanged-base-ac352a952020409fae97341ee4467775.json) · [answers](/Users/ndatar/Documents/ChatGPT/action-interview/artifacts/reward-shaping-v1-results/artifacts/experiments/reward-shaping-v1-selection-validation/answers.json)
- **reward-shaping-v2-four-attempt-diagnostic**: [W&B](https://wandb.ai/nmdatar-harvard-university/repository-qa-training/runs/reward-shaping-v2-four-attempt-diagnostic) · [controller log](/Users/ndatar/Documents/ChatGPT/action-interview/artifacts/reward-shaping-v2-results/campaigns/f0d58c7d5e5bda7784968a6341025d2e95e95174409602a675a367c033736152/reward-shaping-v2-four-attempt-diagnostic.log) · [metrics](/Users/ndatar/Documents/ChatGPT/action-interview/artifacts/reward-shaping-v2-results/artifacts/experiments/reward-shaping-v2-four-attempt-diagnostic/benchmark.json) · [answers](/Users/ndatar/Documents/ChatGPT/action-interview/artifacts/reward-shaping-v2-results/artifacts/experiments/reward-shaping-v2-four-attempt-diagnostic/answers.json)
- **reward-shaping-v2-grader-v5-four-attempt-diagnostic**: [W&B](https://wandb.ai/nmdatar-harvard-university/repository-qa-training/runs/reward-shaping-v2-grader-v5-four-attempt-diagnostic) · [controller log](/Users/ndatar/Documents/ChatGPT/action-interview/artifacts/reward-shaping-v2-results/campaigns/3287d8295cf7362a61ed01b7abdcf1810a066623604e8f89a10561e761272534/reward-shaping-v2-grader-v5-four-attempt-diagnostic.log) · [metrics](/Users/ndatar/Documents/ChatGPT/action-interview/artifacts/reward-shaping-v2-results/artifacts/experiments/reward-shaping-v2-grader-v5-four-attempt-diagnostic/benchmark.json) · [answers](/Users/ndatar/Documents/ChatGPT/action-interview/artifacts/reward-shaping-v2-results/artifacts/experiments/reward-shaping-v2-grader-v5-four-attempt-diagnostic/answers.json)
- **reward-shaping-v2-grader-v5-selection-validation**: [W&B](https://wandb.ai/nmdatar-harvard-university/repository-qa-training/runs/reward-shaping-v2-grader-v5-selection-validation-eval-9e57dc93) · [controller log](/Users/ndatar/Documents/ChatGPT/action-interview/artifacts/reward-shaping-v2-results/campaigns/3287d8295cf7362a61ed01b7abdcf1810a066623604e8f89a10561e761272534/reward-shaping-v2-grader-v5-selection-validation.log) · [metrics](/Users/ndatar/Documents/ChatGPT/action-interview/artifacts/reward-shaping-v2-results/artifacts/experiments/reward-shaping-v2-grader-v5-selection-validation/evaluations/unchanged-base-af1b8c0e248b43f4b9e0ae0776972d11.json) · [answers](/Users/ndatar/Documents/ChatGPT/action-interview/artifacts/reward-shaping-v2-results/artifacts/experiments/reward-shaping-v2-grader-v5-selection-validation/answers.json)
- **reward-shaping-v3-grader-v5-four-attempt-diagnostic**: [W&B](https://wandb.ai/nmdatar-harvard-university/repository-qa-training/runs/reward-shaping-v3-grader-v5-four-attempt-diagnostic) · [controller log](/Users/ndatar/Documents/ChatGPT/action-interview/artifacts/reward-shaping-v3-results/campaigns/362ff8c6c13dd86f4b5d414808f6edee96d70ace2d475b69d64562c933dcda91/reward-shaping-v3-grader-v5-four-attempt-diagnostic.log) · [metrics](/Users/ndatar/Documents/ChatGPT/action-interview/artifacts/reward-shaping-v3-results/artifacts/experiments/reward-shaping-v3-grader-v5-four-attempt-diagnostic/benchmark.json) · [answers](/Users/ndatar/Documents/ChatGPT/action-interview/artifacts/reward-shaping-v3-results/artifacts/experiments/reward-shaping-v3-grader-v5-four-attempt-diagnostic/answers.json)
