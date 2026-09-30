# Post-training pipeline implementation

This package implements the approved Tinker + Modal design in the `dataset-generation` worktree. Software checks, live integration, and human-calibrated experiment readiness are separate outcomes. See `POSTTRAIN_PROGRESS.md` and generated `reports/posttrain/` evidence for current counts; no synthetic run establishes model quality.

## Run commands

Use `.venv-posttrain/bin/python` for live integrations. It contains the pinned optional dependencies from `requirements-posttrain.txt`. `python -m posttrain` works without them for fake-backend runs and local reports. Installing this project registers the equivalent `posttrain` entrypoint.

```sh
.venv-posttrain/bin/python -m posttrain doctor --probe
python3 -m posttrain run --config examples/posttrain/diagnostic-fake.json
python3 -m posttrain report --run artifacts/posttrain/offline-grpo
python3 -m posttrain status --run artifacts/posttrain/offline-grpo
python3 -m posttrain resume --checkpoint artifacts/posttrain/offline-grpo
python3 -m posttrain fork --checkpoint <checkpoint-manifest> --config <new-run-config>
python3 -m posttrain evaluate --checkpoint <checkpoint-manifest> --config <new-evaluation-config>
python3 -m posttrain data validate --release <release-directory>
python3 -m posttrain calibration export --examples <predictions.jsonl> --output <private-review-directory>
python3 -m posttrain calibration import --packet <packet.json> --decisions <human-decisions.json> --output <merged-reviews.json>
python3 -m posttrain tracking sync --run <run-directory> --mode online --project repo-qa-posttrain
```

Each run ID is immutable. Re-running an existing ID is rejected; use resume or change the run ID in a new config. Resume does not add updates to a completed run: its original update budget remains fixed. Increasing it or changing tools, data, or optimizer settings requires a fork.

## Configuration and learning behavior

Examples include direct GRPO, SFT-only, and SFT→GRPO synthetic configurations. Live repository configurations must resolve an accessible model/renderer, immutable train/development releases, independently reviewed private rubrics, ready environment bundles, source roots, verifier configuration and conservative paid-call bounds. SFT requires separately admitted examples; private references are not teacher investigations.

The initial recipe uses one task group per optimizer update, four independent rollouts, synchronous policy refresh, and one importance-sampling update pass. Group advantages use population standard deviation. Equal-reward groups skip optimization; unresolved members exclude the whole group and allow one bounded group retry. User/tool context has zero loss; each generated assistant action is trained once. SFT stages start the subsequent GRPO stage with fresh optimizer state.

`eval_every` and `checkpoint_every` are configurable positive optimizer-step intervals. Development evaluation runs at step zero and at scheduled successful updates, using unchanged tasks and budgets. Evaluation pauses training. Mean reward is reported only with scoring coverage; strict correctness scores are distinct from older reference-rubric ratings.

## Recovery and observability

Every optimizer call has a durable pending/acknowledged journal entry. Unknown outcomes are never blindly retried. Resume creates a restored training client from the last committed checkpoint; uncommitted rollout/update work is discarded and retained in logs. Checkpoints include separate sampler and train/optimizer state references plus framework counters, cursor, RNG state, configuration and lineage. Remote SDK sampling is not promised bitwise reproducible.

Local events and complete episode artifacts precede optional W&B upload. W&B failures do not repeat an optimizer update. Reports include loss, reward, evaluation and artifact links. Unknown cost or time-to-first-token values remain unknown. The fake backend and its curves are explicitly synthetic. Readiness checking recognizes Tinker's official credential store and W&B's normal netrc login, without exposing secrets.

## Data and quality boundaries

Existing992 development tasks remain development-only. Schema-compatible rubrics are distinct from human-admitted gold; a matching reference or file hash alone does not establish answer correctness. New source-only environments use primary regular Git blobs, reject symlink evidence, preserve exact case, and isolate each tool operation. Persistent cross-call execution state is not supported by this recipe.

Diagnostic mode permits at most3 successful updates with at most4 concurrent episodes and may use independently machine-reviewed draft rubrics. It never modifies human-review flags. Longer training requires the unchanged human calibration gate and human-admitted training/evaluation gold. Constructed calibration examples are labeled; independent reviewers supply actual decisions. Small sample packs cannot satisfy statistical confidence gates automatically.

## Spending and remote operation

All paid Tinker, judge and Modal calls reserve costs in `artifacts/posttrain/spending.json`. The total cap is$20, with$2 withheld as a buffer; concurrent in-flight reservations count. Unknown actual charges retain conservative estimates. This dispatch guard does not impose a billing cap at the provider. An estimate violation halts further dispatch.

`posttrain.remote` provides an executable detached Modal CPU coordinator with separate run/private-grading volumes. This path has offline tests only; staging, deployment, remote restart and remote end-to-end acceptance remain unverified. Run these commands from the project checkout. Importing the module and its `plan` operation do not contact Modal.

Create a new remote configuration from the verified diagnostic, keeping its immutable data/model settings but changing these fields: `run_id="remote-qwen4b-01"`, `artifacts_root="/runs/experiments"`, `budget_ledger="/runs/budgets/remote-qwen4b-01.json"`, `budget_cap_usd=3.0`, and `budget_reserve_usd=0.25`. Save it as `artifacts/posttrain/remote-config.json`. This is a fresh run; it does not migrate the completed local run.

```sh
.venv-posttrain/bin/python -m posttrain.remote plan \
  --config artifacts/posttrain/remote-config.json \
  --remote-config /private-grading/configs/remote-qwen4b-01.json \
  --envelope-usd 4 --coordinator-upper-usd 1 \
  --original-workspace "$PWD"
```

Before launch, explicitly stage the selected release directories, every configured environment bundle, repository checkouts including `.git`, and referenced calibration files under `/private-grading/workspace`, preserving their workspace-relative paths. Private references remain on this coordinator-only volume. The original-workspace alias preserves existing absolute paths without rewriting hashed manifests. Do not place Python packages that shadow the trusted image modules in the staging root. Do not upload the campaign spending ledger, `.env`, API-key files or local credential stores.

The following commands illustrate creating volumes and uploading one release; repeat the upload with each exact required directory, then upload the configuration. These commands have not been executed for remote coordination.

```sh
.venv-posttrain/bin/modal volume create repo-qa-posttrain-runs
.venv-posttrain/bin/modal volume create repo-qa-posttrain-private
.venv-posttrain/bin/modal volume put repo-qa-posttrain-private \
  data/releases/repo-qa-training-smoke-v1 /workspace/data/releases/repo-qa-training-smoke-v1
.venv-posttrain/bin/modal volume put repo-qa-posttrain-private \
  artifacts/posttrain/remote-config.json /configs/remote-qwen4b-01.json
```

Configure a named Modal secret `posttrain-services` containing the Tinker and judge credentials required by the config, and optional W&B credentials. Local authentication is not automatically available remotely. Repository tool sandboxes receive neither the private volume nor these coordinator secrets.

```sh
.venv-posttrain/bin/python -m posttrain.remote launch \
  --config artifacts/posttrain/remote-config.json \
  --remote-config /private-grading/configs/remote-qwen4b-01.json \
  --parent-ledger artifacts/posttrain/spending.json \
  --envelope-usd 4 --coordinator-upper-usd 1 \
  --original-workspace "$PWD" --secret posttrain-services
```

Launch reserves the entire $4 envelope in the existing campaign ledger before dispatch; the child ledger permits at most $3 of service calls. These are illustrative estimates and must fit the remaining campaign budget. Never reset or copy the campaign ledger to make room. The parent retains its reservation after failed launches or unknown billing; it does not infer a refund. The coordinator/image estimate is not a provider-enforced billing limit.

Use one coordinator per run and child ledger. For recovery of a previously staged remote run, pass `--checkpoint /runs/experiments/<run>/checkpoints/<checkpoint>/manifest.json` with the unchanged config and existing child ledger. This conservatively reserves another full envelope. Local-to-remote live-state migration is not implemented. Normal longer experiments still require human calibration and admission gates.

## Verified Qwen4B run

`examples/posttrain/repository-qwen4b-diagnostic-v4.json` produced a real GRPO optimizer update and reloadable checkpoint. Its run ID already exists and cannot be reused. The full100-training/20-development cohort configuration is `examples/posttrain/repository-qwen4b-full-cohort.json`; it retains the same shared campaign ledger and diagnostic/human-review restrictions. It has not been executed.

```sh
.venv-posttrain/bin/python -m posttrain status --run artifacts/posttrain/live-repository-qwen4b-004
.venv-posttrain/bin/python -m posttrain report --run artifacts/posttrain/live-repository-qwen4b-004
.venv-posttrain/bin/python -m posttrain doctor --config examples/posttrain/repository-qwen4b-full-cohort.json
```

To continue from weights, use `posttrain fork --checkpoint <manifest> --config <new-run-config>` with a new run ID. To recover an interrupted run without changing training semantics, use `posttrain resume --checkpoint <manifest>`. Resuming the completed diagnostic does not authorize extra optimizer updates. See `POSTTRAIN_RL_ACCEPTANCE.md` for exact live results and remaining quality gates.

Live SFT requires manifest-bound `training/sft.jsonl` and `private/sft_reviews.jsonl`, with exact example digests, independent supported review, and `provenance.collection_mode=blind_solver`. No private reference answer is automatically turned into a solver investigation.
