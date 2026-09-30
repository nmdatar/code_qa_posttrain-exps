# Research progression

Read this page in order for the research story, or jump to a family. These are
research stages, not a claim that each stage improved the model. Each linked
report records its own dataset, judge, reward, task count, and limitations.

| Stage / question | Intervention and observed result | Configs and evidence |
|---|---|---|
| 1. Can we trust the reward? | Early readiness and reward diagnostics exposed unresolved grading and compound-claim/paraphrase errors. Atomic claims separate independent facts; this changes measurement, not the underlying answer. | [Atomic-claims inputs](../configs/experiments/atomic-claims/), [judge comparison](../reports/atomic-claims-judge-comparison/README.md), [paraphrase validation](../reports/grader-paraphrase-validation/README.md), [retired diagnostics](../docs/EXPERIMENT_HISTORY.md) |
| 2. Does direct GRPO produce useful learning signal? | Retain the longer GRPO run and LR/group-size comparisons. Equal-reward groups supplied no centered advantage; five of sixteen batches in the longer run produced no update. Small pilots did not establish improvement. | [GRPO](../configs/experiments/grpo-long-v6/), [long-run report](../reports/grpo-long-v6/README.md), [LR/group sweep](../reports/lr-group-v6-sweep-v1/README.md), [baseline/control](../reports/grpo-v6-parallel/README.md) |
| 3. Does a running baseline use more of the sampled data? | The matched single-seed REINFORCE comparison used 122/128 contributing trajectories versus GRPO's 60/128. The quality difference was uncertain; this is evidence about update utilization, not a proven general winner. | [REINFORCE inputs](../configs/experiments/reinforce-v6/), [results](../reports/reinforce-v6/RESULTS.md), [paired decision evidence](../reports/experiment-decision-evidence/README.md) |
| 4. Can SFT teach investigation behavior before RL? | The 30-example tool-only SFT pilot improved sampled formatting but reduced end-to-end quality: retain this negative result. Full-investigation distillation is a different intervention; distinguish tiny admission smokes from larger teacher-collection studies. | [Tool-only SFT](../reports/sft-tool-warmup-v1/README.md), [expanded studies](../reports/expanded-studies/README.md), [distillation designs](../configs/experiments/research-extensions-v1/README.md) |
| 5. Can efficiency improve without sacrificing correctness? | Compare quality-only and efficiency objectives using matched evaluations. The earlier verifier-tier compute reward and later Bash output-token penalty are different objectives; consult each report rather than pooling their scores. | [Efficiency v1](../reports/efficiency-rl-v1/README.md), [v2 inputs](../configs/experiments/efficiency-rl-v2/), [Bash correctness inputs](../configs/experiments/bash-correctness-grpo-15-v1/), [Bash efficiency comparison](../reports/bash-efficiency-grpo-15-v1/README.md) |
| 6. How do tools and prompting affect investigation? | Restricted shell, unrestricted Bash, structured tools, and decomposition change different parts of the harness. Preserve matched controls and separate inference comparisons from post-training. | [Restricted shell](../reports/shell-only-v1/README.md), [Bash evaluation](../reports/bash-only-eval-v1/README.md), [structured GRPO](../reports/structured-correctness-grpo-15-v1/README.md), [decomposition](../reports/bash-prompt-decomposition-grpo-7-v1/README.md) |
| 7. Which failures and task differences remain? | Keep stability/autoresearch diagnostics, stronger-model controls, and full task-difficulty evidence. They explain protocol fixes and follow-up choices, rather than constituting a single combined quality result. | [Stability](../reports/rl-stability/README.md), [autoresearch](../reports/grpo-autoresearch/README.md), [strong-model validation](../reports/strong-model-validation-v1/README.md), [difficulty](../reports/task-difficulty-full-v1/README.md) |

## How to read a result

Start with the report's status and evaluation cohort. Compare correctness on
matched tasks, show scoring coverage alongside reward, and distinguish training
reward from evaluation reward. Small or single-seed results are exploratory;
failed readiness and negative findings remain part of the record. Never infer
completion from a launch receipt or “ready” document alone.

## Designs versus completed experiments

[Campaign v8](../configs/experiments/current-v8/README.md) is a versioned suite
of prepared study designs, not a universal “latest run” or proof that every arm
was executed. [Procedures](../experiment%20procedures/README.md) and the
[classification](../docs/EXPERIMENT_CLASSIFICATION.md) explain the interventions
and deferred ideas. Use individual family reports for execution evidence.

[Historical index](../docs/EXPERIMENT_HISTORY.md) preserves the retired pilot
story. [Branch integration](../docs/WORKTREE_INTEGRATION.md) records how the
original experiment worktrees were combined. Historical manifests and receipts
retain their original paths and hashes; consult the pinned pre-cleanup revision
when interpreting a path removed from the main checkout.
