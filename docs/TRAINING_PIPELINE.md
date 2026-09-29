# Training pipeline

`training_pipeline` orchestrates synchronous SFT and GRPO using Tinker. Tinker runs
forward/backward and optimizer operations remotely; this package owns input
validation, tool episodes, rewards, loss alignment, checkpoint boundaries, and
local experiment records. Dataset collection and judge calibration are separate.

## Install and commands

```bash
python3 -m pip install -e '.[training,modal,tracking]'
qa-train validate --config examples/training-toy.json
qa-train validate --config examples/training-toy.json --remote
qa-train run --config examples/training-toy.json
qa-train evaluate --checkpoint /absolute/path/to/checkpoint.json
qa-train resume --checkpoint /absolute/path/to/checkpoint.json
qa-train fork --checkpoint /absolute/path/to/checkpoint.json --config new-run.json
qa-train smoke --output artifacts/training-smoke
```

`python3 -m training_pipeline` exposes the same commands without installing the
entry point. `validate` checks configuration and data without creating a trainer;
`--remote` also resolves model capabilities and its tokenizer. Tinker credentials
use the SDK's existing authentication. No credentials belong in configuration.

Training dependencies are optional. Existing baseline and strict evaluation
commands keep their existing behavior. Episode metrics now permit unknown cost
as `null`; old numeric cost records remain valid, and aggregate cost remains
unknown if any contributing episode's cost is unavailable.

## Configuration and data boundary

The executable configuration example is `examples/training-toy.json`. Set a unique
output directory and run ID. Stages may be omitted or repeated, but the stage list
must be nonempty. Each stage specifies its update limit, attempted-batch limit,
batch size, and learning rate. GRPO additionally requires group size of at least
two and temperature 1. Unsupported algorithms fail before trainer allocation.

The toy environment builds eight tiny accepted conversations and deterministic
lookup tasks in memory. They are explicitly synthetic test fixtures. Their fixed
reward assigns 0.8 to correct lookup and up to 0.2 to a simple letter score in the
answer's tag; this gives the fixture an output-dependent reward without inventing
per-attempt labels. It measures plumbing, not useful repository behavior.

For real repository input, set `environment.kind` to `repository`, and provide:

- `release`: absolute directory containing the immutable release.
- `manifest_sha256`: SHA-256 of `manifest.json`.
- `calibration`: absolute path to a human calibration report.
- `calibration_sha256`: SHA-256 of that report; it must have status `passed` and
  a nonzero reviewed-task count.

The release manifest must set `training_eligible: true` and contain SHA-256 hashes
under `artifacts` for `sft/train.jsonl`, `rl/train.jsonl`, and
`evaluation/development/tasks.jsonl`. Unused training exports may be empty; an
active stage's export may not. Development evaluation input must be nonempty.

Every row has `id`, `split`, `family_id`, and `lineage_id`. IDs are unique within an
export; families and lineages cannot cross train/development splits. No release
is generated or relabeled by this pipeline.

SFT rows additionally contain `status: "accepted"` and `messages` (role/content
objects). Optional `assistant_turns` selects message indices to supervise;
defaults select all assistant turns. User and tool observations are context.
Each assistant turn is rendered with its actual prefix, and prefix compatibility
is checked rather than guessing token offsets. Empty targets and overflow fail.

Repository RL/evaluation rows additionally embed `task` (the existing private
`TaskSpec`), `experiment` (the existing frozen `ExperimentConfig`), `environment`
(a ready manifest from `agent_harness.images.prepare_environment`), and
`snapshot_root` (an absolute pristine host checkout used only for evidence
verification). Embedded identities, splits, commits, allowed tools, frozen task
hashes, and independent judge families are validated. Runtime judges use the
existing configured `judge_command`; their endpoint credentials remain on the host.
The generated baseline worktree's development exports are not implicitly
compatible training releases.

## Episode and update semantics

The shared `agent_harness.runner` records exact per-generation prompt tokens,
sampled tokens, behavior log probabilities, raw text, and policy identity before
tool execution. Each repository episode receives a fresh Modal sandbox with no
private references mounted. Tools are `list_files`, literal `search_code`, bounded
`read_file`, and `python_probe`; observations are bounded and commands execute
without host-shell interpolation. A public tracked-file catalog comes from the host
snapshot, so archive-based Modal images do not need `.git` metadata. Tool images
require `python3`; the catalog is capped at 1 MB and split into bounded argv chunks.
Evidence grading reads a separate pristine
snapshot. Tool command failures are observations; transport failures are unresolved.

Only the public question, repository identity, tools, and budgets enter the prompt.
The strict final response is an `AnswerSubmission` object in an `answer` envelope.
Trusted `EpisodeRecorder` telemetry is authenticated before grading. Full grading
records live in the host's `private/` directory, outside policy observations.
The reference-claim baseline remains a separate evaluation command and score.

SFT uses Tinker's built-in cross-entropy with mean assistant-token loss within an
example, then mean across examples. GRPO uses population-standardized terminal
advantages and Tinker's built-in importance-sampling loss. Because the SDK sums
loss contributions, rows carry explicit normalization: mean over generated tokens
within a contributing trajectory, then mean across contributing trajectories.
Earlier assistant turns used as context are never trained again.

A group must share task, policy, environment, experiment, split, and reward version.
Any unresolved member excludes the whole group. Infrastructure failures permit
one whole-group retry; all attempts remain logged. Zero-variance groups contribute
zero. An entirely zero-contribution batch advances the task cursor without an
optimizer step. There is no clipping, KL penalty, replay, or multi-epoch update.

## Checkpoints and recovery

The first checkpoint records the initial trainer before updates. Each GRPO update
commits training and sampling artifacts before the next rollout. SFT honors the
configured checkpoint cadence and always saves at stage completion. Stage changes
load the preceding weights into a fresh optimizer.

Local manifests publish only after both remote artifacts exist and match the
model and adaptation identity. `.pending.json` files expose incomplete saves;
`checkpoints/latest.json` continues pointing to the last fully committed state.
Manifests include configuration, dataset identity, stage/step/batch counters,
shuffle order/cursor, client RNG state, and checkpoint lineage.

- Evaluate loads only the immutable sampler, without a trainer.
- Fork creates a new run with compatible weights and a fresh optimizer/counters.
- Resume restores optimizer and framework state; semantic configuration changes
  are rejected. Output and tracking destinations may change.

Ambiguous forward/backward or optimizer failures poison that client. Restore the
last committed checkpoint rather than retrying a possibly applied update. Partial
rollouts are discarded on recovery. Remote generations need not reproduce bit for
bit. Each run records a source snapshot. A process lock prevents concurrent training
writers, and shared spending reservations use a separate file lock. Checkpoint TTL is explicit; expired artifacts fail load preflight.

Periodic and standalone evaluation use the shared runner and bind each result to
the exact checkpoint, configuration, and data identity. Missing results remain
visible in coverage. Automatic evaluation is development-only; this training CLI
does not expose final-test selection or automatic best-checkpoint promotion.

## Artifacts, spending, and validation

Events and trajectories are flushed locally before optional W&B logging. Checkpoint
manifests, evaluation reports, and a stable sample of public traces are uploaded as
artifacts when tracking is enabled. Private grading records are excluded. Upload
failures are recorded without replaying training. Local artifacts are the recovery
source. W&B modes are `disabled`, `offline`, and `online` (explicit configuration).

The spend ledger reserves uncached input/output token bounds, training tokens,
and checkpoint storage before provider operations. Failed or ambiguous calls keep
their reservations. Unknown actual billing remains `null`. For the smoke command,
all invocations in the same output root share one $5 ledger, including debugging
attempts. Pricing is fetched from Tinker's documented JSON endpoint and storage
rate page; preflight stops if either cannot be verified. Checkpoints expire after
24 hours. The storage bound intentionally allows 32 bytes per base-model parameter
for each pair of training/sampler artifacts, far exceeding rank-8 adapter state.
General repository runs may also incur Modal and judge costs; this ledger bounds
Tinker charges only. The toy smoke uses local tools and no paid judge.

The smoke performs one SFT update and at most two GRPO updates, using at most four
predefined RL batches of two tasks and four attempts. It interrupts at a committed
boundary, resumes with a distinct client, independently evaluates the final
sampler, and verifies a fresh-optimizer fork. At least one nonzero-contribution RL
update is required to pass. Quality improvement is not required or claimed.

```bash
python3 -m unittest discover -s tests -v
```

Tests cover numerical reduction/alignment, real SDK tensor construction, grouped
failure semantics, stage transitions, checkpoint save failures, fork/resume,
private-reference separation, strict grading with synthetic judgment fixtures,
unknown costs, and tracking outages. The live smoke report records actual update
acknowledgments and checkpoint references separately from offline test results.

Live results and their limits are recorded in [the smoke report](../reports/training-pipeline-smoke.md).

## Automated task admission for experimental rollouts

Human review is not required to decide whether an existing collection task can
produce an experimental rollout. Run:

```sh
qa-train export-rollouts --source /path/to/collection-release --output data/releases/new-rollout-release
```

This offline operation verifies source artifact hashes, selected reference review
status (including independently reviewed corrections), question/reference hash
bindings, environment image/commit attestations, and cited source blobs and line
ranges. It excludes unsupported references and incompatible execution requirements.
It preserves public/private separation, retains source provenance and attribution,
and assigns whole repository families to train or development. The original
collection is unchanged. Reusing benchmark tasks for training means they are no
longer an untouched benchmark; the export makes no final-test claim.

`data/releases/repo-qa-automated-rollouts-v1` contains 866 training tasks across 33
families and 121 development tasks across 8 families. Five unsupported references
were excluded. The previously exercised Pydantic, TanStack Query and SQLAlchemy
families remain development-only. Admission reuses recorded live environment checks;
it does not claim a new live check or calibrated reward quality. Private references
retain their original draft status and `human_reviewed: false`.

The export uses `experimental-rollout-release-1.0`, preserving collection-native
Modal environments and reference-comparison grading. It is a task export, not a
completed RL run. `rollout_eligible` and `experimental_rl_task_eligible` are true;
`training_eligible` remains false for the existing strict training-loader contract.
The current strict `qa-train run` adapter cannot consume this format: the remaining
integration is a collection-native rollout adapter and provisional reference reward.
Do not relabel draft references as human-approved gold to satisfy that adapter.
No new SFT examples are manufactured and no provider spending is incurred by export.
