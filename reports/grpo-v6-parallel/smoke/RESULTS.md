# v6 smoke results

The smoke completed six attempted batches and **four acknowledged optimizer updates**, with five checkpoint manifests (initial plus one for every update). It demonstrates functioning RL execution and graded reward variation, not an improvement in held-out performance.

| Metric | Initial | Final |
|---|---:|---:|
| Strict full passes | 8/32 | 4/32 |
| Strict total credit | 8/32 | 4⅓/32 |
| Strict mean, all32 denominator | .2500 | .1354 |
| Resolved strict grades | 29/32 | 29/32 |
| Completed answers | 32/32 | 31/32 |

The final evaluation includes one task with1/3 partial strict credit. Neither evaluation meets95% scoring coverage, so neither is eligible for best-checkpoint selection. The final checkpoint is saved, but is not evidence of a better policy. The parent report separately measures factual coverage by grading the saved answers; factual improvement cannot be inferred from these strict results.

Training:48/48 reward judgments resolved,5/12 groups had reward variation,7 groups had zero variation,0 groups excluded. Nineteen trajectories had nonzero normalized advantages (one trajectory in a variable group was exactly at the group mean). Four batches produced updates; two all-zero batches were correctly skipped. Batch mean rewards were .4375,0,0,.375,.475,.25; overall mean .25625 and final EMA .30341. The curve is not steadily increasing.

The frozen v6 dataset was loaded and validated. However, this small fixed-seed sample contains12 unique training tasks and **none of the78 repaired placeholder tasks**. Therefore this run exercises the complete corrected runtime configuration, but does not directly establish the effect of rewritten claims. Training tasks were not changed after observing results.

Tool use:45/48 training episodes completed;3 exhausted their budget.173 tool executions had0 execution failures, but52 outputs were truncated and20 action-validation errors remained. Six action errors attempted citations to unobserved files; other errors include malformed JSON and unsupported arguments. Tool execution is more reliable than valid action generation.

Final checkpoint: `ckpt-12b4df13c46546679d8e33894e4d2cf6`, optimizer step4. All five manifests contain sampler and optimizer-state Tinker URIs with172800-second retention. Local archive contains manifests and logs, not a permanent exported weight archive; no best-eligible extended-retention checkpoint was produced.

[W&B](https://wandb.ai/nmdatar-harvard-university/repository-qa-training/runs/grpo-v6-smoke-seed42-v1)

- Config: `configs/experiments/grpo-v6-parallel/smoke-v1.json`.
- Procedure: `reports/grpo-v6-parallel/smoke/PROCEDURE.md`.
- Machine-readable results: `reports/grpo-v6-parallel/smoke/summary.json`.
- Task/tool audit: `reports/grpo-v6-parallel/smoke/training-task-audit.json`.
- Raw archive: `artifacts/grader-paraphrase-validation-results/artifacts/experiments/grpo-v6-smoke-seed42-v1/` (877 files;112 trajectories).
- Curves and events: `reward-curve.csv` and `events.jsonl` in that archive.
- Joint campaign receipt: `reports/grpo-v6-parallel/submission.json`.

The parent reconciled authoritative GRPO reservations to $269.4167164141429, a **$39.4243686284295 combined baseline+smoke increment**. This is conservative reservation accounting, not provider billing. Parallel arm before/after cost counters overlap and must not be added. No caps were raised. No confirmation tasks were used.
