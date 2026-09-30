# Dataset creation — runnable development batch

Latest collection: **992 runnable tasks** (980 attributed benchmark imports plus 12 local tasks), with all 41 imported source-reading environments verified on Modal. The [versioned task release](TASK_COLLECTION_1000_RESULT.md) contains 987 reviewed grading references and 5 quarantined references; runnable status alone does not establish reference correctness. See [live counts and elapsed time](TASK_COLLECTION_PROGRESS.md) and [collection quality policy](TASK_COLLECTION_1000_QUALITY.md). This document retains the first batch’s execution details.

The first selected batch is running end to end: **three pinned tasks, three ready Modal images, passing reference probes, independent investigations, and exact replay of all 29 recorded tool operations**. All 5,472 exported source-file hashes were checked inside the actual images. This implements a concrete development pass through the [dataset roadmap](DATASET_ROADMAP.md) and its [Excalidraw plan](https://app.excalidraw.com/s/8Ufs2ZMhWhu/AJrov8yM1oi).

This is **development data with draft gold**, not a human-approved training release or a representative benchmark. The remaining 11 sourcing candidates have not been built by this batch. No SFT, RL, preference, or held-out test records were admitted.

## Running tasks and evidence

| Task | Pinned behavior verified | Modal image | Independent investigation |
|---|---|---|---|
| PY-01 / `pydantic-frozen-copy-8960` | Frozen/strict construction and assignment, shallow/deep copy updates, mapping and instance revalidation | `im-XrhkLS2V0pJRfnyKrlZ4RW` | [Answer](../artifacts/investigations/pydantic-v1/answer.md), [5-operation replay](../artifacts/investigations/pydantic-v1/replay.json) |
| TQ-02 / `tanstack-hydration-fetch-freshness-8936` | Cached return at 10 seconds and 299999 ms; fetch at exactly 300000 ms; hydration timestamps and option precedence | `im-laOoTR2oIVBe3vGjZ05rP4` | [Answer](../artifacts/investigations/tanstack-query-v1/answer.md), [12-operation replay](../artifacts/investigations/tanstack-query-v1/replay.json) |
| SA-01 / `sqlalchemy-filtered-collection-7654` | Collection `[1,2]` → `[1,2]` → `[1]` → `[1,2]`; same identity-map object across queries | `im-klaBAlIS5AAdvSlwbWH75M` | [Answer](../artifacts/investigations/sqlalchemy-v1/answer.md), [12-operation replay](../artifacts/investigations/sqlalchemy-v1/replay.json) |

Exact commit IDs, source provenance, claims, assertions, and recipes are in [Pydantic](../examples/dataset_sources/pydantic.json), [TanStack Query](../examples/dataset_sources/tanstack-query.json), and [SQLAlchemy](../examples/dataset_sources/sqlalchemy.json). These authoring specifications are **private curator inputs**; they must not be supplied to a solver.

The [batch configuration](../examples/dataset_sources/development-batch.json) assigns the three repository families to development. It preserves the budget configuration and bundle paths. The [run summary](../reports/development-batch/run-summary.json) records actual observations, image references, tool counts, timings, hashes, and limitations.

## What the quality checks found

1. **Answer correctness:** three private executable matrices pass. Assertions compare actual behavior with authored expectations; stdout is not fabricated. A separate reference author wrote each task before the blind investigation.
2. **Independent investigation:** fresh-context Codex agents received only public task records and the repository tool interface. Their source access and experiments were recorded. Private references remained outside the sandbox. These are real agent investigations, not simulated fixtures or captured private reasoning.
3. **Reference and answer review:** an independent agent checked all 18 required claims, citations, and execution claims against pinned source and observed logs. See [structured review](../reports/development-batch/independent-review.json). This is machine review; human approval remains false.
4. **Replay and environment integrity:** all 29 operations reproduce their stdout, stderr, exit status, timeout status, and truncation status in fresh sandboxes. Two fresh-sandbox isolation checks per image observed nonroot execution, read-only source, blocked outbound access, absent private records, and clean temporary state. Runtime inventory and every source-file hash were inspected in each built image.
5. **Difficulty:** a [question-only control](../reports/development-batch/question-only-control.json) recovered the main behavioral answers for all three. Therefore these are useful development correctness tasks, **not demonstrated hard repository-reasoning tasks**. Exact implementation details and source citations add value: for example, Pydantic's pinned selective deep-copy path differs from the generic remembered explanation. The control used abbreviated equivalent scenarios and one response per task, so it is advisory rather than a comparable baseline score.
6. **Leakage:** these families are development-only. All training outputs are empty. The preliminary benchmark audit is not a complete repository-family or semantic-duplicate clearance. No claim is made that familiar public behaviors were absent from model pretraining.

No token counts or billed-dollar telemetry were exposed for the Codex investigators; those fields are null, not zero. Tool call and wall-time measurements are recorded. Output bytes, tool calls, per-operation resources, and attempt deadlines are enforced; output-token limits are not claimed enforced without token measurement.

## Environment architecture actually used

The controller runs outside Modal. It sends bounded tool operations into reusable immutable images. Each operation creates a new isolated sandbox and terminates it afterwards. Source is read-only; `/tmp` is fresh. State does not persist between operations, so a stateful SQLite or clock-controlled experiment runs entirely inside one bounded probe.

- **Pydantic:** Python 3.12.14; pinned core 2.49.0, typing-extensions 4.16.0, typing-inspection 0.4.4, annotated-types 0.7.0; imports the checked-out Python package from `/workspace`.
- **TanStack Query:** Node 22.16.0 plus Python 3.11.2 for the tool wrapper; pinned esbuild 0.28.2 compiles checked-out query-core into `/opt/query-core.cjs`. Its actual compiled-artifact hash is recorded. This does not substitute an npm release for the repository code.
- **SQLAlchemy:** Python 3.12.14 with typing-extensions 4.16.0; imports checked-out `/workspace/lib`; uses an in-memory SQLite fixture, no external database service.

Tracked internal file and directory symlinks are materialized from Git objects, with mappings recorded. External, cyclic, unsupported, and untracked targets are rejected. No repository code executes on the controller host. Dependency installation and repository execution occur inside remote image builds or sandboxes.

Replay uses the existing immutable image ID. Rebuilding a recipe from a base tag or OS package repository can produce a different image; its readiness, integrity, and verification checks must run again. Exact installed versions and observed source hashes are saved, without claiming bit-identical rebuilds from mutable upstream registries.

## Repeat the workflow

Run these commands from the `dataset-generation` worktree. Modal is already configured here; credentials stay on the controller and are never sent into the repository sandbox.

```sh
# Validate specs and reuse unchanged prepared bundles.
.venv-dataset/bin/python -m dataset_builder.batch \
  --config examples/dataset_sources/development-batch.json --phase prepare

# Re-run private assertions, isolation checks, and source inventory in existing images.
.venv-dataset/bin/python -m dataset_builder.batch \
  --config examples/dataset_sources/development-batch.json --phase verify

# If images need rebuilding, use --phase build first; then re-run verify.
```

Start a new agent investigation using a new output directory:

```sh
.venv-dataset/bin/python -m dataset_builder.investigate start \
  --bundle data/generated/pydantic-v1 \
  --task pydantic-frozen-copy-8960 \
  --output artifacts/investigations/pydantic-repeat-001

.venv-dataset/bin/python -m dataset_builder.investigate tool \
  --attempt artifacts/investigations/pydantic-repeat-001 \
  --name search_code \
  --args-json '{"query":"def model_copy","path":"pydantic/main.py","limit":20}'
```

Available tools are `list_files`, `search_code`, `read_file`, and `python_probe`. The last can invoke Node inside the Query image. Tool arguments can also be supplied with `--args-json @args.json`. Continue investigating, save an answer, then use `finish --attempt ... --answer-file ...` and `replay --attempt ... --output ...`. Every result is recorded in a hash-chained `trajectory.jsonl`. Existing attempts and releases cannot be silently overwritten.

The runtime is model-independent. This batch used Codex subagents as investigators. A separate provider-backed automatic model loop is not implemented here; do not confuse the tool controller with a configured unattended experiment scheduler.

## Versioned outputs and admission

The [development release](../data/releases/repo-qa-development-v1/manifest.json) contains:

- `evaluation/development/tasks.jsonl`: three solver-visible prompts, system prompts, tool permissions, budgets, repository pins, and environment IDs.
- `evaluation/development/environments.jsonl`: immutable image references and versioned source/recipe fingerprints.
- `private/`: reference claims, assertions, source provenance, verification, isolation, build manifests, and image attestations. Never include this directory in agent context.
- Empty `sft/train.jsonl`, `rl/train.jsonl`, `preferences/train.jsonl`, and `evaluation/final_test/tasks.jsonl`, because no record has been admitted to those outputs.
- A dataset card, split manifest, and deterministic artifact hashes.

The release describes runnable tasks and reference status. Investigation logs and independent reviews remain separate append-only experiment artifacts; they do not silently change the frozen task records. Package a new version after substantive prompt/reference/environment changes.

## Roadmap coverage and next collection requests

| Component | Implemented for this batch | Remaining beyond this batch |
|---|---|---|
| A: contracts/configuration | Public/private schemas, configurable budgets, strict validation | General multi-language collection configuration expansion |
| B: corpus/splits | Exact snapshots, provenance and development-family assignments | Complete benchmark/fork/semantic leakage clearance for training |
| C: environments | Three ready Modal images, source integrity and version inventory | Other shortlisted repositories and service-backed recipes |
| D: questions | Three bounded, source-motivated final prompts | Automatic broad candidate generation and coverage selection |
| E: references | 18 atomic claims, source hashes, three executable matrices | Human reference adjudication |
| F: runtime | Bounded tools, fresh sandboxes, logging, deadlines, replay | Persistent multi-step scratch state if a future task requires it |
| G: investigations | Three completed blind-context Codex investigations | Configured provider-backed unattended collection and token/cost telemetry |
| H: verification | Behavior probes, independent source review, 29-operation replay | Human acceptance and disagreement adjudication |
| I: release | Hashed development preview, no training leakage | Accepted SFT/RL/preferences and frozen test releases |
| J: improvement | Difficulty control identifies familiar behaviors | Source harder revision-specific scenarios, then repeat validation |

**Next targeted request:** prioritize VT-01's recent module-cache regression and TOK-01's scheduler/virtual-time interaction from the [sourcing shortlist](TASK_SOURCING_SHORTLIST.md). Expand only after a question-only screen and independent source verification. Also seek questions requiring a previously undocumented interaction or a genuinely ambiguous premise. Preserve the easy tasks for regression coverage; do not inflate their difficulty labels or relax reference checks to grow the dataset.
