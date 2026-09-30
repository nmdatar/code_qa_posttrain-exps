# Qwen397 question subsections: seven-iteration bash GRPO

Treatment starts fresh Qwen3.5-4B with seed 42, rank 8, GRPO LR 1e-5, batch size 8 and group size 8. It runs seven attempted batches (448 training attempts), up to seven acknowledged optimizer updates. Evaluation remains at baseline, every three updates, and final: normally steps 0, 3, 6, 7. Thus there are two intermediate evaluation updates, not only two optimizer updates.

The existing `bash-correctness-grpo-15-v1` run supplies the original-prompt control at steps 0, 3 and 6. Reusing it avoids another paid control. Compare reward gains relative to each arm's own baseline; the intervention affects inference as well as learning. A single seed and a historical control provide an exploratory result, not a reliable causal improvement estimate.

Before any rollout, frozen Qwen/Qwen3.5-397B-A17B receives each of the 32 training and 32 selection questions, and only those questions. At temperature zero with a 1,024-token output cap it creates 1–6 named question subsections. The original question is preserved verbatim and the breakdown appended to the solver's user prompt. Outputs must be valid structured questions; an invalid generation fails preparation, without silently falling back to the control. Every teacher request/output is archived under `prompt-decomposition`, and the same breakdown is reused at every checkpoint. Schema validation does not prove perfect semantic preservation; inspect the archived breakdowns for omissions or invented hints.

References, source evidence and rubrics are never sent to the decomposer. Grading continues to receive the original question. Student tool and generation budgets stay unchanged, and added prompt tokens count toward the existing context/cost limits. Decomposition uses additional preprocessing calls, recorded separately from student rollouts.

Same bash tools, correctness-only reward, frozen grader settings, 32 training questions, disjoint 32 selection questions, rollout limits, temperatures, concurrency and checkpoint cadence as the control. Confirmation is unused. Current source differences from the frozen control include the inactive token-penalty option, additional efficiency metrics and descriptive checkpoint names; these do not change this run's reward or optimization.

Conservative reservation estimate including controller and decomposition: $1,924.10, under the unchanged $4,000 run ceiling. This is not actual billed cost. Exact settings and prices are in `configs/experiments/bash-prompt-decomposition-grpo-7-v1/run.json`; the submitted bundle is immutable.

57 targeted tests passed, including question-only teacher inputs, decomposition reuse, original-question grading, bash isolation, correctness, training and remote packaging tests.

Read comparative progress and baseline-adjusted reward gains with:

```sh
PYTHONPATH=. .venv-eval/bin/python scripts/compare_prompt_decomposition_grpo.py
```

Inspect scoring coverage alongside gains. Aggregate resolved-case averages can cover different tasks; compare matched task-level results before concluding the treatment improves reward.
