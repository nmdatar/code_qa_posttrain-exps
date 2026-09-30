# Worktree integration status

Integration date: 2026-09-28.

The shared working directory is `/Users/ndatar/Documents/ChatGPT/action-interview`.
Git's original `main`, `codex/agent-harness`, and `dataset-generation` branches all
pointed at the same empty initialization commit (`59930d8`). The substantive files
in those directories were untracked, so branch ancestry alone did not show whether
work was finished or available to other agents.

## Completed work integrated

- **Research harness:** imported the completed implementation from
  `/Users/ndatar/.codex/worktrees/agent-harness/action-interview`, including its
  dataset-builder dependencies, local/remote tools, coordinator, tests, guides,
  and existing verification report. The owning chat, “Plan deep research harness,”
  was idle and reported completion. Its historical live validation was synthetic;
  this integration did not launch paid remote jobs.
- **Archived training/evaluation code:** recovered missing files from archive
  snapshot `fd513de` (which includes implementation commit `7e0f801`), including
  `training_eval`, its tests/examples, and evaluation config/tracking support.
  Kept newer shared QA schema/reporting changes in this working directory.

These were file-level integrations: the harness had no implementation commits,
while the archived training branch has a separate root history. They are not
ordinary branch merges. Source worktrees and archived Git objects are preserved.

## Interfaces retained

| Interface | Implementation |
| --- | --- |
| Research loop | `agent_harness.runner.AgentRunner` |
| Current training loop | `agent_harness.training_runner.run_episode`, also exported from `agent_harness.runner` |
| Research CLI | `agent-harness` (`agent_harness.cli`) |
| Environment CLI | `qa-sandbox` (`agent_harness.sandbox_cli`) |
| Remote research coordinator | `agent-harness-remote` |
| Current training pipeline | `qa-train` / `training_pipeline` |
| Recovered training implementation | `training-eval` / `training_eval` |

Legacy environment CLI calls through `agent_harness.cli` still dispatch for
`prepare`, `validate`, `smoke`, and `run --manifest`. The research and training
implementations coexist; this integration does not claim they are one unified
training architecture.

## Work still in progress

The “Dataset creation” chat was actively implementing `posttrain` in
`/Users/ndatar/.codex/worktrees/dataset-generation/action-interview`. Its current
pipeline was not imported wholesale or marked complete. Main-folder training work
was also active and was preserved. Its current shared runtime dependencies are
recorded as a Git snapshot so the integrated code can travel with its imports;
that snapshot does not declare the active training task complete. Dataset-builder files imported here are the
completed harness dependency snapshot, not the active dataset worktree's latest
version.

Future integration should use explicit commits and reviewed file ownership.
Do not replace another checkout's entire package based solely on a matching
branch HEAD. Finish and validate a bounded change, commit it in its source branch,
then integrate it in the shared checkout. Check the actual working directory of
commands: a chat's displayed directory may differ from the worktree it edits.

## Validation

- Clean export of committed snapshot `b6932da`: all 339 offline tests passed.
  This includes 2 new worktree-integration regression tests and confirms the
  imported code does not rely on untracked source files.
- Other agents continued editing the shared training pipeline after the snapshot;
  these subsequent changes are outside this validation result.
- No new live inference, training, deployment, or paid-service validation.

Pre-integration copies of replaced files were saved at
`/tmp/action-interview-integration-backup` for this session.


## Experiment-family branch stack (September 2026)

The private GitHub repository is `nmdatar/action-interview`. Each branch below
builds on the immediately preceding branch. The first starts at `077829f`.
Shared runtime implementations are frozen in the foundation; experiment layers
add their configurations, launch/analysis scripts, and compact result reports.
These are code-history increments, not claims that later experiments use earlier
experiments' trained weights. Original worktrees and raw artifacts remain intact.

| Order | Branch | Increment |
|---|---|---|
| 1 | `codex/01-shared-runtime` | shared runtime |
| 2 | `codex/02-dataset-posttraining` | dataset posttraining |
| 3 | `codex/03-research-demo` | research demo |
| 4 | `codex/04-baseline-throughput` | baseline throughput |
| 5 | `codex/05-grader-and-reward-ablation` | grader and reward ablation |
| 6 | `codex/06-direct-grpo` | direct grpo |
| 7 | `codex/07-learning-rate-group-sweeps` | learning rate group sweeps |
| 8 | `codex/08-reinforce` | reinforce |
| 9 | `codex/09-sft-distillation` | sft distillation |
| 10 | `codex/10-quality-gated-efficiency` | quality gated efficiency |
| 11 | `codex/11-research-harnesses` | research harnesses |
| 12 | `codex/12-restricted-shell` | restricted shell |
| 13 | `codex/13-bash-only-evaluation` | bash only evaluation |
| 14 | `codex/14-bash-correctness-grpo` | bash correctness grpo |
| 15 | `codex/15-bash-efficiency-grpo` | bash efficiency grpo |
| 16 | `codex/16-structured-correctness-grpo` | structured correctness grpo |
| 17 | `codex/17-prompt-decomposition` | prompt decomposition |
| 18 | `codex/18-task-difficulty` | task difficulty |
| 19 | `codex/19-stability-autoresearch` | stability autoresearch |
| 20 | `codex/20-experiment-suite-and-results` | experiment suite and results |

The combined tree exactly matches the integration tree validated with 925 tests
(8 skipped for optional/local prerequisites), before adding this branch index.
Individual historical configurations can require local datasets or provider
credentials. Generated builds, environments, and large raw trajectory dumps
are not included. No training runs were launched during integration.

Each branch has its own managed local worktree. See
[the file-level branch manifest](EXPERIMENT_BRANCH_STACK.json) for commit IDs and
file ownership. Compare each branch with its predecessor to review one family.
