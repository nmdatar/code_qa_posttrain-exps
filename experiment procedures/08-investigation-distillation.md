# 08 — Distill investigation behavior versus answer-only supervision

**Category:** model weights → SFT target / training data, with an optional matched GRPO continuation. This extends [procedure 04](04-optional-sft-before-grpo.md), which already supports full-investigation SFT but lacked a matched answer-only arm.

## Arms

| Arm | Supervised targets | Subsequent training |
|---|---|---|
| Base | None | None; reuse exact current-version baseline |
| Direct GRPO | None | Current fixed 16-batch GRPO recipe |
| Investigation SFT | Verified tool actions and final answer | None |
| Answer-only SFT | Only the final answer from the same admitted trajectory | None |
| Investigation → GRPO | Exact selected investigation SFT weights | Same direct-GRPO recipe, fresh optimizer |
| Answer-only → GRPO | Exact selected answer-only SFT weights | Same direct-GRPO recipe, fresh optimizer |

The answer-only arm **retains the same teacher tool history as context**, with zero loss on that history. It isolates whether supervising investigation actions adds value. It is not question-only answer imitation: removing teacher evidence changes the conditioning distribution and requires a separate study. Both arms verify the entire original trajectory and use the same frozen manifest, tasks, lineage deduplication and native token proofs. The renderer normalizes each example's selected target loss; report this alongside the differing target-token counts.

## Readiness and teacher data

[Configs](../configs/experiments/research-extensions-v1/README.md) are offline-validated using the existing **one-lineage** strictly passing investigation release. This supports integration smoke checks only; it cannot establish distillation effectiveness. The existing source is an archived policy-generated demonstration, not evidence of stronger-teacher transfer. The completed 30-example tool-prefix pilot is neither a matched answer-only control nor a full-investigation dataset.

For a meaningful study, collect a larger verified release on training questions only. Freeze the teacher checkpoint, prompt, tool interface, attempts per task, decoding and budgets before collection. The current-v8 `04-collect-demonstrations.json` collects base-policy demonstrations; use a separately registered teacher-model variant for stronger-model distillation and verify tokenizer/renderer compatibility. Do not assume cross-model native token proofs are interchangeable. Verify with the exact current judge/rubric and source hashes; exclude failed, truncated, stale, overlength, duplicate-lineage or ungrounded trajectories. Keep private rubrics in verification only. Report attempted/admitted counts, repository coverage, teacher identity and selection bias. Set the release size and adequacy decision before training; never repeat the one example to claim a populated study.

## Procedure

1. Freeze a single release for both SFT arms. Regenerate procedure-04 configs with `prepare_current_experiments.py` after additional collection, then pass that frozen campaign to `prepare_research_extensions.py --source <campaign-directory> --output <fresh-extension-directory>`. Freeze new versioned output/run identities; never overwrite current manifests.
2. Compare one shuffled pass over exactly the same examples, same seed/order, rank 8, LR `1e-4`, batch size eight, with an unpadded last batch and `ceil(N/8)` updates. Use `supervised.loss_scope=all-admitted-turns` versus `final-answer-only`. Prompts and observations have zero loss in both arms. Report supervised target tokens, total processed tokens and training/teacher costs; equal examples are not equal compute. Any token-matched comparison is a separate registered follow-up.
3. Evaluate SFT arms using the same strict 32-task selection protocol. Select checkpoints by the existing coverage-eligible rule. Inspect the selected step: initial base weights can win. If initial weights win, say so rather than claiming SFT helped.
4. Fork each selected SFT checkpoint with fresh optimizer into its dependent `08-*-then-grpo.json`. Run sequentially after its parent; stop if its checkpoint is absent, ineligible or expired. Compare against `08-direct-grpo.json`, or reuse a prior run only if all data, code, initialization, evaluation and recipe settings match.
5. All three RL arms have 16 attempted batches, at most 16 updates, two tasks × four attempts, LR `1e-5`, temperature 1, zero whole-group retries, and the same 393,216 worst-case rollout-output allowance. Skip equal-reward groups as prescribed. Do not equalize successful updates by granting one arm more samples.
6. Compare strict acceptance and claim coverage before/after SFT and along RL, plus action validity, useful evidence per tool call, unsupported claims, generated tokens, latency and all-attempt cost. Show RL-only and end-to-end cost curves including teacher collection, rejected demonstrations, grading and SFT. Preserve unresolved grades and report at least 95% scoring coverage for eligibility.
7. Select on selection only; repeat promising training sequences at seeds 43/44 with the same frozen data before locked 85-task confirmation. Those larger-data, additional-seed and confirmation configs must be frozen separately; the supplied seed-42 smoke bundle is not a full effectiveness study.

Archive release hashes, complete original proofs, selected-turn masks, rendered token counts, checkpoints, costs and task-paired analyses. A gain from investigation supervision on this controlled pair supports learning investigation actions; it does not by itself establish stronger-teacher transfer, question-only answer imitation, or a compute-matched advantage.
