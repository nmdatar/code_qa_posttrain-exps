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

### Configurable experiments and W&B names

Copy [`examples/training-experiment.json`](../examples/training-experiment.json)
for a collection GRPO experiment. All settings below are read from JSON; no code
edits or per-run command-line hyperparameter overrides are required:

| JSON field | What it controls |
|---|---|
| `model.base_model` | Tinker solver/training model |
| `model.rank` | LoRA rank |
| `judge.base_model` | Independent frozen Tinker judge; never updated by training |
| `judge.renderer` | `hf-chat-no-thinking-v1`, or an installed Tinker cookbook renderer name |
| `judge.context_tokens`, `max_tokens`, `temperature`, `provider_timeout_seconds` | Judge-only context, generation and request settings |
| `stages[].learning_rate` | Learning rate for that stage |
| `stages[].batch_size` | Tasks per GRPO batch, or examples per SFT batch |
| `stages[].group_size` | Attempts per task; nominal GRPO episodes per batch = batch size × group size |
| `stages[].max_updates`, `max_batches` | Successful-update and attempted-batch limits |
| `stages[].optimizer` | Optional `beta1`, `beta2`, `eps`, `weight_decay`, `grad_clip_norm` passed to Tinker Adam |
| `seed` | Task-order RNG and trainer initialization seed |
| `group_retries` | Whole-group retries for retryable infrastructure failures; default 1 |
| `limits` | Generation, output/context token, tool, latency and provider limits |
| `evaluation.every`, `max_tasks`, `temperature` | Scheduled development evaluation cadence, prefix size and sampling temperature |
| `model.checkpoint_ttl_seconds`, `checkpoint_every` | Retention and SFT checkpoint cadence; GRPO still commits each update |
| `spend.cap_usd`, `ledger`, `prices` | Persistent reservation ceiling, ledger location and solver prices |

Omitted optimizer fields retain SDK defaults. GRPO temperature remains 1 because
the current importance-sampling objective uses that behavior-policy contract;
judge/evaluation temperatures may be any finite nonnegative value. No PPO, KL,
learning-rate scheduler or parallel-rollout setting is implied by these controls.
Evaluation also runs at stage completion (and before collection training), even
when `evaluation.every` is zero. Task budgets can narrow run limits.

Set `judge.prices: null` to fetch and pin that model's current sampling prices at
run setup, or supply a model-bound price object with `model`, `prefill`, `sample`,
`source`, and `checked_at`. `spend.prices: null` resolves solver pricing when a new
ledger is created. Changing a model requires its matching prices; use a fresh
ledger for a different solver. The template uses separate solver/judge models
and automatic prices; availability and the spending estimate are checked at launch.
Plain `validate` does not fetch prices or contact providers. `validate --remote`
checks model capabilities/renderers, but is not a complete spending preflight.

`run_id: "auto"` creates a new ID for every new run/fork. Use `{run_id}` in `output`
and `spend.ledger` to keep artifacts and reservations separate. Explicit run IDs
and paths remain supported. Relative paths are relative to the working directory.
The resolved config is saved as `config.json` and embedded in checkpoints.

W&B uses the resolved run ID as its stable ID, and generates a readable name such
as `Qwen3.5-4B__judge-Qwen3.5-9B__grpo-lr1e-05-b1-g4__r8-s42__<run-id>`.
Multi-stage names include each stage. The full run config is logged, and update
events include learning rate, batch size, group size and explicit optimizer settings.
The tracking config also accepts:

```json
{
  "mode": "online",
  "project": "repository-qa-training",
  "entity": "your-team",
  "experiment_id": "lr-group-size-sweep",
  "run_name": "optional-readable-condition",
  "tags": ["grpo", "independent-judge"],
  "notes": "Describe the hypothesis and changed settings."
}
```

`experiment_id` becomes the W&B group. `run_name` overrides the generated condition
label; the unique run ID is still appended. Omit optional fields rather than using
empty strings. Tracking identity is persisted in `tracking.json`, including when
W&B is disabled/unavailable. Resume keeps that identity; standalone evaluation gets
a separate W&B ID and `evaluation` job type. The example defaults to disabled
tracking; set `tracking.mode` to `online` after configuring your account/project.

```bash
python3 -m training_pipeline validate --config examples/training-experiment.json
python3 -m training_pipeline run --config examples/training-experiment.json
# Resume the exact saved experiment, not the template with run_id: auto:
python3 -m training_pipeline resume --checkpoint /absolute/path/to/checkpoint.json
```

Changing judge, optimizer or other training semantics requires a new run/fork.
A fork keeps weights with a fresh optimizer and requires matching solver model/rank;
changing the base model requires a fresh run. Old configs without `judge` retain
their legacy same-base-model judge behavior. Independent judge settings apply to
collection mode; strict repository mode continues using each task's frozen judge
contract. Neither mode's grading is made calibrated merely by selecting a larger model.

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
Use `environment.kind: collection` with `qa-train run` for this format. The
collection adapter uses the shared episode runner, isolated `/workspace` Modal
images, and a frozen base-model reference judge. Its provisional reward is labeled
`experimental-reference-v1`; it does not enter the strict calibrated evaluation path.
Do not relabel draft references as human-approved gold to satisfy that adapter.
No new SFT examples are manufactured and no provider spending is incurred by export.


## First repository GRPO pilot

`examples/training-repository-pilot.json` starts Qwen/Qwen3.5-4B directly with
rank-8 GRPO, learning rate 1e-5, groups of four, one task per batch, at most two
batches and one optimizer update. SFT is absent. The first checkpoint provides
the baseline evaluation; the committed final checkpoint uses the same two
development tasks. Equal-reward batches skip the optimizer as usual.

The collection loader admits source-reading tasks only: 858 training and 117
development records. It checks release hashes, private reference and pinned source
bindings, and family splits. Executable tasks are not silently downgraded to
source-only tasks. Public questions/tools enter the sandbox; private references
and grading remain on the host and in judge requests. AnswerSubmission and trusted
EpisodeRecorder telemetry are retained. Invalid/missing answers or citations score
zero; unknown grading or infrastructure outcomes remain unresolved. The reference
judge uses a frozen Qwen/Qwen3.5-4B base sampler, never the updated policy. This is
an inexpensive exploratory reward, not independent calibrated quality evidence.

The $5 ledger covers policy sampling, judge sampling, updates, checkpoint storage,
and full-lifetime Modal reservations at resource limits with a 2x rate allowance.
No images are built. The pre-dispatch estimate includes whole-group retries and
conservative evaluation/checkpoint counts. Reservations persist before calls;
ambiguous calls retain them. Provider billing is distinct and may be unavailable.
Pricing evidence is saved alongside `launch-estimate.json` and `spend.json`.

```sh
qa-train validate --config examples/training-repository-pilot.json
qa-train validate --remote --config examples/training-repository-pilot.json
qa-train run --config examples/training-repository-pilot.json
```

The run directory is exclusive; for a new experiment choose a new run ID/output
and budget ledger. For recovery, use the last committed checkpoint with `resume`;
never blindly reissue an ambiguous optimizer call. A config copied to another
machine must point to the exported release and its pinned source checkouts.

### Qwen pilot protocol correction

Use `examples/training-repository-qwen-v2.json` for the corrected pilot. It pins
`environment.protocol_version: experimental-reference-v2`. The earlier pilot
exposed a literal `TASK_ID` placeholder in the example and rejected Markdown-fenced
JSON. V2 inserts the actual task ID, accepts a single JSON fence, and supplies only
missing envelope metadata (`task_id`, schema version, optional diagram). It does
not repair answer text, citation paths, citation ranges, or source hashes. Both raw
generations and parsed actions are persisted. This change uses a new run/fork,
not silent continuation of training under changed protocol semantics.

Collection images use a persistent Python command server over Modal stdin/stdout,
with sequential commands, bounded outputs and deadlines, and sandbox teardown.
The initial Modal exec transport timed out on these prebuilt images; a live
command-stream readiness/listing probe passed before the corrected run.

The v2 pilot shares the original $5 ledger, including reservations from failed
attempts. Its estimate plus prior reservations was $4.9183 before dispatch. It
uses one GRPO group (four attempts), at most one update, and one fixed development
task before and after. These small samples can verify training integration but
cannot establish reliable quality improvement.


### Independent judge for Qwen experiments

`examples/training-repository-qwen-nemotron.json` keeps the policy at
`Qwen/Qwen3.5-4B` and selects a frozen
`nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` judge, matching the training judge
family in the separate `posttrain` configuration. The cookbook renderer is
`nemotron3_disable_thinking`; judging uses temperature 0, a 16,384-token total
context limit, and at most 1,024 output tokens. The same frozen judge grades
training and development to keep their scale consistent. This is a candidate
judge upgrade, not evidence of improved grading accuracy or calibration.

Install optional dependencies in the environment used for this runner:

```sh
python -m pip install -e '.[training,tinker,remote]'
qa-train validate --config examples/training-repository-qwen-nemotron.json
qa-train validate --config examples/training-repository-qwen-nemotron.json --remote
```

Remote validation checks sampling capability and renderer construction without
sampling tokens or creating a training client. Independent judges do not need
training capability. Missing dependencies/models fail explicitly; there is no
fallback to Qwen. Legacy configs without `judge` retain their original frozen
policy-family judge for historical checkpoint evaluation.

Judge settings, prompt identity, and pricing participate in the experiment/reward
identity. Changing them requires a new run or fork, not resume. Reports retain
raw judge tokens, conditioning tokens, stop reason, and model/renderer identity.
Truncated or malformed grades remain unresolved. No judge optimizer is created.

The launch estimate and each request reserve the judge's own published token
prices against the same total ledger as policy, Modal, and checkpoint charges.
The current example uses the project's $650 direct-GRPO allocation and 48-hour
checkpoint retention. The total project ceiling is $1,000, including all other
allocations; see `configs/experiments/project-budget.json`. No experiment may be
launched yet. Historical pilot configs and ledgers remain unchanged.

Before comparing policies, re-evaluate the unchanged Qwen base and candidates
using this frozen judge and identical cohorts. Do not compare Nemotron scores
against the earlier Qwen-judged pilot as evidence of model improvement. Check
agreement on audited, correct, incorrect, partial, and insufficient-evidence
answers; until calibrated, all quality conclusions remain exploratory. The
completed one-update pilot established integration, not quality improvement.


Current launch configurations and budget allocation are indexed in
`configs/experiments/README.md`. The single live Nemotron check is recorded in
`reports/nemotron-single-rollout-check.md`; it establishes integration, not calibration.


Bounded collection concurrency and the no-training throughput benchmark are
implemented; see [the concurrency guide](CONCURRENCY.md). Paid benchmark execution
remains paused.
