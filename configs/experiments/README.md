# Experiment configuration index

Start with the [research progression](../../experiments/README.md) to choose an
experiment family. Configs here are frozen inputs: run identity, data and judge
version, reward, limits, and budget belong to that experiment. Folder names do
not imply comparative quality or authorization to launch.

- [Campaign v8](current-v8/README.md): prepared dataset-v6/judge-v7 designs for baseline, GRPO, LR/group size, SFT, efficiency, and tools/context. Read each arm's readiness and prerequisites; not all were run.
- [Research extensions](research-extensions-v1/README.md): decomposition and matched investigation-distillation designs.
- [Bash correctness](bash-correctness-grpo-15-v1/), [Bash efficiency](bash-efficiency-grpo-15-v1/), and [structured correctness](structured-correctness-grpo-15-v1/): separate 15-iteration study configurations.
- [Longer GRPO](grpo-long-v6/), [REINFORCE](reinforce-v6/), [tool-only SFT](sft-tool-warmup-v1/), and [expanded studies](expanded-studies-v1/): earlier substantive studies retained for comparison and research context.

[Training usage](../../docs/TRAINING_PIPELINE.md) and
[remote execution](../../docs/REMOTE_EXPERIMENTS.md) describe the shared machinery.
Validate a selected config offline before preparing any new versioned run.

## Historical inputs retained as dependencies

The task manifest under `reward-v1` is still hash-bound by atomic-claims studies.
The run and subset under `fast-grpo-selected-v1` are inputs to the Bash config
generator. Their presence does not make either an active study entry point.
Historical budget records also remain unchanged; removing experiment files does
not reset spending or authorize a new campaign.

Superseded launch bundles and diagnostics are indexed in the
[historical record](../../docs/EXPERIMENT_HISTORY.md), with exact archive links
and a machine-readable removal manifest.
