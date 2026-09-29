# Generated tasks and isolated environments

Implementation of the repository/environment portion of [the dataset roadmap](requirements/DATASET_ROADMAP.md). This is a working-example pipeline, not a claim that all roadmap components or a training-ready release are complete.

## Included source tasks

[Click source specification](examples/dataset_sources/click.json) pins `pallets/click` at `06b2a678741131fd577ce170e23e5ca0aeba0309`. It records ten recent commits across September 10–23, 2026 (including merges, not ten independent changes), the BSD-3-Clause code license, and two newly authored questions motivated by real user reports:

- Optional values: omitted option versus bare flag versus explicit value, from [issue #3084](https://github.com/pallets/click/issues/3084).
- Negative boolean defaults versus nonboolean activation values, from [issue #3111](https://github.com/pallets/click/issues/3111).

The historical issues supply intent, not ground truth. Claims are checked against the pinned current implementation and upstream tests. Each question has expected callback behavior checked by a small executable probe. The [candidate audit](examples/dataset_sources/candidates.json) records other inspected repositories and sourcing limitations.

These are **development tasks**. Benchmark-family overlap remains unresolved; training export is disabled. Machine-verified assertions do not substitute for human reference review or semantic grading of an agent answer. Gold stays `draft`, `human_reviewed=false`.

## Reproduce

From the project root, using Python 3.11+ and Git:

```sh
python3 -m dataset_builder prepare \
  --spec examples/dataset_sources/click.json \
  --output data/generated/click-v1
```

Preparation fetches the exact commit, rejects dirty/mismatched snapshots, validates evidence ranges and hashes, and writes separate solver-visible and private files. It never executes repository code on the host. Choose a new output directory for a new release; nonempty output is deliberately preserved and rejected.

For Modal (the selected live backend):

```sh
uv venv .venv-dataset
uv pip install --python .venv-dataset/bin/python 'modal==1.5.5'
.venv-dataset/bin/modal token new
.venv-dataset/bin/python -m dataset_builder build-environment \
  --bundle data/generated/click-v1 --backend modal
.venv-dataset/bin/python -m dataset_builder verify \
  --bundle data/generated/click-v1
```

Modal credentials stay on the controller host. Only tracked repository blobs are uploaded into the image: no project directory, evaluator files, model credentials, or private assertions. Runtime networking is blocked; CPU, memory, time, process, and captured-output limits are bounded. Sandboxes terminate after each operation. Private probes are passed separately to fresh verification sandboxes over stdin, never baked into the solver image.

Docker is also supported with `--backend docker` when a local daemon is running. It uses a resolved base-image digest, read-only filesystem, fresh tmpfs, nonroot user, no capabilities/network/host mounts, and forced cleanup on timeout. Modal stores its immutable image object ID; reusing that built image is the supported replay path. A rebuild from a mutable base tag may differ.

Click needs no third-party runtime dependencies for these probes. The readiness command imports the pinned source from `/workspace/src`; each probe explicitly uses that same path. This avoids accidentally testing a preinstalled release. The environment factory supports dependency-install commands for future repositories; those execute only inside the build environment.

## Bundle layout and agent inputs

| File | Consumer / contents |
|---|---|
| `public/tasks.jsonl` | Agent: starting system prompt, user prompt, repository URL/commit/family, split, environment ID, permitted tools and budgets |
| `public/environment.json` | Controller: snapshot location and environment recipe; do not expose host filesystem tools |
| `private/tasks.jsonl` | Evaluator: existing TaskSpec-compatible claims, evidence hashes, critical errors and probe fingerprints |
| `private/assertions.jsonl` | Verifier: executable probes, expected output, user-issue provenance |
| `private/source-spec.json` | Curator: complete authored source specification and source audit |
| `manifest.json` | Controller: deterministic artifact hashes and admission state |
| `environment-build/result.json` | Controller: built image identity, readiness result and source hashes |
| `private/verification-report.json` | Curator: actual execution results, checksums, pass/failure status and review state |

The checkout is under `artifacts/repos/`. Snapshot references keep source separate from task records. Only the pinned repository enters the sandbox; merely putting gold in a different local folder would not provide isolation.

The public/private task adapter preserves the evaluator's `train`, `development`, and `final_test` split names. Private claims do not leak into the strict public task schema. Family and task-lineage split checks run before export. Source records marked `train` additionally require a completed overlap audit; this does not grant human acceptance.

## Validation and limits

```sh
python3 -m unittest discover -s tests -v
```

Tests cover schema/private-data separation, dirty and wrong-commit rejection, evidence path/symlink confinement, split leakage, deterministic preparation, immutable output preservation, sandbox command policies, timeout/output handling, and private assertion results.

Behavior probes verify reference facts, not the semantic quality of a generated answer. The source explanations still require review. This implementation does not fabricate a teacher trajectory, an SFT release, a human review, or an RL training run. Automated repository discovery, general question generation, a provider-backed agent controller, large-scale collection, and final release admission remain subsequent components. This example gives those components a concrete task/environment contract and reproducible checks.
