# Result: sampled formatting improved, but task performance did not

The four-update tool-only pilot improved sampled-action validity in a secondary fixed-context probe, but did not improve end-to-end performance over the frozen v6 baseline. Do not promote this checkpoint. This is a negative result for this small recipe, not a test of full expert-trajectory SFT or SFT followed by RL.

| Metric, same 32 selection tasks | External baseline | Within-run untrained | After SFT |
|---|---:|---:|---:|
| Strict full-credit passes | 7 | 5 | 3 |
| Mean demonstrated strict reward | 0.2188 | 0.1563 | 0.1146 |
| Resolved grades | 28 | 31 | 29 |
| Completed answers | 31 | 32 | 28 |
| Budget exhausted | 1 | 0 | 4 |
| Invalid actions / generations | 1 / 165 | 0 / 166 | 3 / 192 |
| Invalid JSON objects | 0 | 0 | 0 |
| Tool calls | 133 | 134 | 160 |
| Generated tokens | 7,741 | 7,511 | 8,138 |
| Clipped observations | 48 | 50 | 44 |

The SFT policy used all five tool calls on **every task**; the within-run control did so on 22/32. Its three rejected actions were two paths outside the pinned source catalog and one citation to an unobserved file. A plausible interpretation is that tool-only imitation overweights continuing to investigate and supplies no supervision for stopping with a supported answer. This is a hypothesis consistent with the traces, not an isolated causal mechanism established by this pilot.

The primary comparison missed every preregistered exploratory gate. After-SFT scoring coverage was 90.6%, below 95%. The within-run strict-reward delta was −0.0417; the paired-task 95% bootstrap interval was [−0.1563, +0.0417], and the repository-clustered interval was [−0.0690, 0]. For invalid-action rate the delta was +0.0156, with paired-task interval [0, +0.0365] and repository-clustered interval [0, +0.0648]. This is a small, single-seed, selection-cohort result with an uncalibrated research judge. Unresolved grades count as no demonstrated success, not proven incorrectness. The external baseline's 7 passes versus the internal control's 5 illustrate variability; their lower/higher scoring coverage also differs.

## What was trained

Base Qwen/Qwen3.5-4B; fresh rank-8 adapter; seed 42; LR 1e-4; one pass through 30 unique training-task prefixes from 19 repository families; batches 8/8/8/6. There were 69 source-verified tool actions and 1,809 supervised tokens. Prompts, tool observations and final answers received no loss. These were archived base/RL-policy actions, not stronger-teacher demonstrations. No new teacher cost or development-to-training leakage. All four optimizer calls were acknowledged and checkpointed. The best checkpoint under the existing strict selection rule remained the untrained step-zero checkpoint.

Loss values by successive (different) batches: 0.0933, 0.1794, 0.2125, 0.1832. These are not a held-out learning curve and do not independently establish improvement. Training implementation checks passed: 528 tests, native tokenizer/prompt alignment for every admitted example, context lengths, source hashes, lineage/split isolation and exact masks.

## Decision and next experiment

Keep the unchanged baseline. Do not initialize RL from this pilot's SFT checkpoint.

A better SFT test would collect a broader, balanced set of **complete verified investigations**: search/read actions, concise source-supported final answers, and correct stopping. Use training repositories only, check citation entailment and when evidence is sufficient, and avoid training on answers selected only by permissive factual reward. Start with SFT-only evaluation, then compare SFT+GRPO against direct GRPO with identical rollout budgets and fresh optimizer state. Include sampled decoding from the outset because historical RL format failures occur at temperature 1, while greedy evaluation here is near the JSON-error floor. This follow-up is a recommendation, not a launched experiment.

## Sampled-action probe: a format gain with a stopping tradeoff

The secondary probe completed **256 generations**: 64 identical public contexts (first and last action prefixes from each initial-control trajectory), two samples per context per policy, temperature 1, 512 output-token cap. The specification was frozen before inspecting the final SFT results; no further training or judging occurred.

| Probe metric | Before SFT | After SFT |
|---|---:|---:|
| Valid actions | 109/128 (85.16%) | 121/128 (94.53%) |
| Malformed/non-object JSON | 8/128 | 2/128 |
| Valid final answers on the later contexts | 47/64 | 30/64 |

Action validity rose **9.375 percentage points**; invalid actions fell from 19 to 7 (63% relative reduction). The paired-context 95% interval is [+1.56, +16.41] points, but the more conservative repository-clustered interval is [−1.56, +17.65] points and includes no gain. Interface validation checks JSON, envelopes, arguments and source binding; it does not prove an action executes successfully or an answer is correct. These are fixed histories, not complete on-policy rollouts, and sampling seeds are not controlled by the provider interface. Action validity and final-answer production are distinct metrics.

This gives a reason to continue investigating **balanced full-trajectory SFT**, while rejecting this tool-only initialization for production or RL. It does not establish that SFT+RL beats direct RL. [Raw/paired probe summary](probe-results.json), [preregistered specification](probe-spec.json), [probe submission](probe-submission.json), [raw samples](../../artifacts/sft-tool-prefix-v1-probe-results/artifacts/experiments/sft-tool-prefix-v1-probe/samples.jsonl).

## Artifacts and cost

- [Frozen protocol](README.md), [admission](admission.json), [mask audit](mask-audit.json), [paired results](results.json).
- [Training config](../../configs/experiments/sft-tool-warmup-v1/train.json), [submission receipt](submission.json), [archive checksums](checksums.json), [tests](tests-final.log).
- [W&B training run](https://wandb.ai/nmdatar-harvard-university/repository-qa-training/runs/sft-tool-prefix-v1-seed42).
- [Archived events](../../artifacts/sft-tool-prefix-v1-results/artifacts/experiments/sft-tool-prefix-v1-seed42/events.jsonl), [latest checkpoint](../../artifacts/sft-tool-prefix-v1-results/artifacts/experiments/sft-tool-prefix-v1-seed42/checkpoints/latest.json).

503 files were archived and JSON-verified; the controller exited 0. The primary run added **$24.1492 in reservations**, against a conservative $123.6906 estimate. These are ledger reservations, not provider invoices. The project ceiling remains $1,000 and this ledger cap remains $550. Historical trajectory acquisition is sunk cost; zero incremental collection expense does not imply free from-scratch data.

The secondary probe added **$0.5126** in reservations ($0.3226 model sampling plus $0.1899 controller). Combined primary + probe reservations: **$24.6617**; final shared ledger **$294.0784 / $550**. Both controllers exited 0. All 509 archived files have recorded checksums. Main preflight: 528 tests; supplemental probe validation: 6 focused tests and 11 remote-execution tests passed. [Cost reconciliation](probe-budget-after.json), [matching solver/grader source audit](matched-source-audit.json).
