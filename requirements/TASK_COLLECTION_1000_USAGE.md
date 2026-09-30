# Using the repository Q&A task collection

Worktree: `/Users/ndatar/.codex/worktrees/dataset-generation/action-interview`, branch `dataset-generation`.

The collection contains source-reading benchmark imports and the original local executable tasks. See [progress and timing](TASK_COLLECTION_PROGRESS.md) for current counts. The [final release](../data/releases/repo-qa-1000-v1/DATASET_CARD.md) is published: 992 tasks, 987 grading references, and 5 quarantined references. See the [completion report](TASK_COLLECTION_1000_RESULT.md).

## What to give the solver

Give the solver the selected record from `public/tasks.jsonl`: the starting system prompt, user prompt, pinned repository, environment ID, permitted tools and budgets. Resolve the environment through `public/environments.jsonl` or the prepared bundle. Source-reading imports permit `list_files`, `search_code` and `read_file`; the local executable tasks also have their explicitly permitted probe tool.

The solver must not receive `private/`, the raw benchmark files, reviewer outputs, grading inclusion lists or authoring specifications. The repository image contains the pinned repository source, not those curator artifacts. Controller and model credentials stay outside the repository sandbox.

## Start an investigation now

Choose a task ID and its bundle from `data/generated/collection-1000-v1/<repository>-<commit>/public/tasks.jsonl`. Use a new attempt directory each time:

```sh
.venv-dataset/bin/python -m dataset_builder.investigate start \
  --bundle data/generated/collection-1000-v1/<repository>-<commit> \
  --task <task-id> \
  --output artifacts/investigations/<new-attempt-id>

.venv-dataset/bin/python -m dataset_builder.investigate tool \
  --attempt artifacts/investigations/<new-attempt-id> \
  --name list_files --args-json '{"path":"","limit":30}'
```

Continue with `search_code` and `read_file`. Each operation uses a fresh stateless sandbox; temporary files do not persist between tool calls. The runtime enforces the task's tool allowlist and bounds, records observations, and keeps private grading off the tool surface. Finish with an answer file and replay the recorded calls:

```sh
.venv-dataset/bin/python -m dataset_builder.investigate finish \
  --attempt artifacts/investigations/<new-attempt-id> --answer-file <answer.md>

.venv-dataset/bin/python -m dataset_builder.investigate replay \
  --attempt artifacts/investigations/<new-attempt-id> --output <replay.json>
```

These commands expose the model-independent investigation runtime. They do not configure a model provider or run an unattended solver loop. Existing source-tool canaries establish environment usability, not solver answer correctness.

## Private grading and selection

The final release's `private/grading.jsonl` is the grading input: it selects supported original references or independently supported corrections, plus reviewed local claims. `private/references.jsonl` retains original upstream answers and proposed corrections for provenance; do not blindly score against the original `reference_answer` field there.

Read `private/quality.jsonl` for per-task review scope and state. Unresolved references remain visible for curation but are excluded from grading. Source hash assertions check the identity of evidence; natural-language correctness still requires semantic judgment. Agent source review is not human approval, and task difficulty has not been measured across this collection.

All imported benchmark records remain development data with original source splits retained in provenance. They do not enter training or a fresh held-out test set. SFT and preference examples also require accepted solver trajectories, which this collection step does not fabricate.

## Reproduce collection checks

```sh
python3 -m dataset_builder.collection --resume
.venv-dataset/bin/python -m dataset_builder.collection_environments \
  --bundles-root data/generated/collection-1000-v1 --workers 3 \
  --output reports/task-generation-1000/environments.json
python3 -m dataset_builder.collection_review
```

The environment runner reuses matching checked images and invalidates stale bindings. Final packaging requires all first-pass and correction reviews, validates the staged release, and publishes a new directory atomically. The current version already exists; choose a new version when re-exporting:

```sh
python3 -m dataset_builder.collection_release \
  --output data/releases/<new-release-version>
python3 -m dataset_builder.collection_audit data/releases/repo-qa-1000-v1
```

Do not overwrite a frozen release. Corrected inputs require a new release version. Exact image IDs are the reproducible runtime references; rebuilding from an upstream base-image tag is not a claim of bit-identical output.
