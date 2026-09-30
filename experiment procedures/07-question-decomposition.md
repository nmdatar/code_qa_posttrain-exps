# 07 — Decompose the question before searching

**Category:** inference harness → prompt / investigation strategy. Solver weights, tools, retrieval, judge and dataset stay fixed. This tests whether a question-derived checklist improves coverage of multi-part questions.

Configs: [research-extensions-v1](../configs/experiments/research-extensions-v1/README.md), `07-control-{selection,confirmation}-r{1,2}.json` and `07-decompose-{selection,confirmation}-r{1,2}.json`.

## Intervention and controls

Control uses the current paginated source-reading prompt. Treatment adds `harness.planning=question-checklist-v1`: organize the user question into subquestions before searching, gather evidence for each, and check unresolved gaps before answering. Only the public question supplies checklist content. No reference answer, rubric, teacher hints or gold passages enter solver context.

This is a **prompt-only decomposition screen**: the checklist is internal and not separately emitted, enforced or measured. Both arms retain the action JSON schema; no extra model call, tool or free planning budget is introduced. A separately emitted/validated plan would be another harness intervention and needs its own implementation/configs. A null result here does not rule out structured planning.

## Procedure

1. Validate each config offline. Freeze source revision, config hashes, current-v8 dataset-v6 / grader-v7 identities and existing 32/85 selection/confirmation membership.
2. Run both selection arms twice with unchanged base Qwen3.5-4B, rank 8, seed 42, temperature 0. Alternate order: control/treatment for repeat 1, treatment/control for repeat 2. These are repeated inference measurements, not independent training seeds.
3. Keep the 8,192 context, six responses, five tools, 512 tokens/response, 3,072 total output and 300-second episode bounds in both arms. Count added prompt tokens in actual cost; equal caps do not imply equal realized consumption. Do not silently compensate the treatment with additional calls.
4. Compare task-paired strict acceptance, supported claim coverage, unresolved grade rate, complete cited answers, tokens, calls, latency and all-attempt cost per accepted answer. Inspect omitted requested parts, especially lifecycle / cross-file questions; define these strata from public tasks before reading arm results.
5. Report both repeats and repository-clustered uncertainty. Preserve unresolved grades and sensitivity bounds; require at least 95% scoring coverage for promotion. Do not select the best repeat per question.
6. Lock the prompt and analysis on selection before running the two confirmation pairs. Promote on a confirmed quality gain, or quality noninferiority within 0.02 with a confirmed cost saving. Otherwise retain control; report uncertainty if underpowered.

Use hard configured runtime/spend limits and the shared integrity rules. Archive prompts, trajectories, per-task judgments, costs and paired analyses. The added prompt is versioned in `training_pipeline/harness_variants.py` and captured in the frozen source bundle. Preparation does not launch paid work.
