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

- Full combined offline suite: 337 tests passed.
- After the final CLI compatibility adjustment: 9 targeted tests passed,
  including 2 new worktree-integration regression tests.
- No new live inference, training, deployment, or paid-service validation.

Pre-integration copies of replaced files were saved at
`/tmp/action-interview-integration-backup` for this session.
