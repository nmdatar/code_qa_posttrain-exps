# Training and evaluation pipeline

`training_eval` implements synchronous SFT and GRPO, Tinker/model loading, immutable checkpoints, standalone/periodic evaluation, and local-first W&B logging. Dataset generation, repository tools, sandbox lifecycle and the research-agent loop remain external, as shown in the [dataset](https://app.excalidraw.com/s/8Ufs2ZMhWhu/AJrov8yM1oi) and [harness](https://app.excalidraw.com/s/8Ufs2ZMhWhu/7us72y2PT2q) drawings. Only synthetic environment/verifier adapters are supplied here.

## Run the offline pipeline

Requires Python 3.11+. From this checkout:

```bash
python3 -m training_eval demo --output /tmp/training-eval-demo
python3 -m unittest discover -s tests -v
```

The demo uses a stateful two-token mock model. It exercises SFT → GRPO → checkpoint → development evaluation, creates real local artifacts, and explicitly reports synthetic rewards. It is not evidence of language-model quality. `result.json` contains the final manifest path. Every manifest records distinct training and sampling artifacts plus client/RNG progress.

Install optional integrations only when needed:

```bash
python3 -m pip install -e '.[training,tracking]'
```

Tinker credentials come from the SDK's runtime environment (`TINKER_API_KEY`); W&B credentials come from W&B's normal environment/login. Never put them in config JSON or task payloads. The adapter targets Tinker 0.30.x; the default installation constrains that minor version.

## Online W&B logging

Install the tracking extra and authenticate privately with `wandb login --verify`. Keep API keys out of config files. The [online preview config](examples/training-wandb.json) streams synthetic SFT/GRPO metrics, sampled rollout tables, trace artifacts and checkpoint-linked evaluations to project `repository-qa-eval`:

```bash
python3 -m training_eval run --config examples/training-wandb.json \
  --plugin training_eval.mocks:demo_inputs --output artifacts/runs/synthetic-wandb-preview
```

Add `tracking.entity` to select a W&B team; when omitted, W&B uses its configured default account. Assign a new `run.run_id` and output directory for each fresh experiment. Online logging retains local events and artifacts during outages. Checkpoint manifests stay local; W&B receives checkpoint IDs. This preview still uses synthetic models and tasks. Real baseline evaluation requires the external model and harness configuration.

To upload a previously recorded W&B offline run, authenticate and run `wandb sync --project repository-qa-eval /absolute/path/to/offline-run-directory`. Plain local JSON logs require replay through `WandbSink`; they are not W&B sync files.

## Organizing experiments

Pass organization metadata in the config's `tracking` object for both `run` and `evaluate`:

```json
{
  "mode": "online",
  "project": "repository-qa-eval",
  "experiment_id": "grpo-reward-ablation-v1",
  "run_name": "evidence-reward-seed42",
  "tags": ["grpo", "evidence-reward"],
  "notes": "Compare evidence rewards against the baseline",
  "metadata": {"variant": "evidence-reward"}
}
```

`experiment_id` maps to the W&B run group; `run_name` is the display name. `run.run_id` remains the unique run identity, so use a different ID for each seed, variant, or standalone evaluation. `job_type` defaults to `training` or `evaluation` according to the command, and may be overridden. For standalone evaluation, retain the experiment ID and add `source_run_id` pointing to the training run; `model.checkpoint` selects the evaluated checkpoint. Periodic evaluations remain within their training run.

The metadata is recorded in local `run_metadata` events even when W&B is disabled and in W&B run configuration when enabled. Extra descriptive fields belong in `tracking.metadata`; keep credentials out of all metadata. On resume, reuse the original organization metadata and run ID. When asked to run an experiment, the operator can populate these fields before invoking the normal CLI; no source edits are needed.

## Interfaces for every diagram node

The [editable overview](https://app.excalidraw.com/s/8Ufs2ZMhWhu/3eHAQBP8Lwr) now names these interfaces. Protocols and neutral records live in [contracts.py](training_eval/contracts.py); concrete implementations are small adapters around them.

| Diagram node | Contract / implementation | Ownership |
|---|---|---|
| Base model / checkpoint | `ModelRef` | Exactly one base ID or manifest reference |
| Model factory | `ModelFactory.resolve` → `ModelBundle` | Train a base model, evaluate, fork, or resume |
| SFT dataset | `SFTDataset`, `SFTExample`; in-memory/JSONL adapters | Consume already rendered token IDs and target masks |
| Optional SFT | `TrainingStrategy`; `SFTStrategy` | Cross-entropy with selected-token mean reduction |
| GRPO | `TrainingStrategy`; `GRPOStrategy` | Group-standardized terminal rewards, importance-sampling loss |
| Environment / tasks / tools | `Environment.run`, `TaskSource`, `RolloutRequest` | External harness; mock only here |
| Verifier | `Verifier.verify` → `Verification` | External private grader; mock only here |
| Checkpoint registry | `CheckpointStore`; `LocalCheckpointStore` | Atomic manifest publication and lineage |
| Evaluation | `Evaluation`; `Evaluator` | Fixed-policy development or explicitly requested final-test evaluation |
| Held-out tasks | `TaskSource` | Dataset agent supplies fixed split/family identity |
| Tinker | `Backend`, `SamplingPolicy`; `TinkerBackendFactory` | SDK sampling, updates, saving/loading |
| Run records | `Tracker`; `JsonTracker` | Durable local events and full trace artifacts |
| W&B | `TrackingSink`; `WandbSink` | Best-effort live metrics and rollout tables |
| Future algorithms | `TrainingStrategy`, `StrategyRegistry`, input-adapter registry | New requirements, input kinds, losses and client state |

`Pipeline` owns stage order, counters, checkpoint cadence, retries and periodic evaluation. It does not parse tools, execute repository code, or construct grading rubrics. Existing `qa_eval` and `eval_pipeline` code is preserved because it supplies separate domain-specific grading/reporting behavior; there was no competing trainer to remove.

## Dataset and harness handoff

Export SFT JSONL rows with `example_id`, `tokens`, `loss_mask`, `split="train"`, and optional `task_id` / `family_id`. Tokens and binary mask have the same full-sequence length; mask index 0 is zero and at least one subsequent assistant token is selected. The trainer performs next-token shifting. Supply the exact tokenizer and renderer/version IDs when creating `JsonlSFTDataset`. The Tinker adapter uses the base-model ID as its tokenizer identity and the explicit `renderer` config as its rendering identity.

The data/harness adapter owns model-specific conversation and tool rendering, including special tokens and stop sequences. The backend accepts exact token prompts; it does not guess a chat template. Tinker factory `stop` accepts explicit renderer stop strings or token IDs. Include the full rendering configuration in the versioned identity. `context_limit` is required and must match the chosen model; local checks enforce it, but the catalog API does not establish that caller-supplied number.

`Task` contains a public payload, task ID, split and family ID. Private answers/rubrics stay inside the verifier implementation; they never belong in `Task.payload`. SFT and RL inputs must be reviewed train-split releases. The pipeline rejects known task/family overlap with periodic evaluation. Supply SFT task/family metadata for these checks; external release auditing remains responsible for semantic duplicates and missing lineage metadata.

An external environment implements:

```python
class MyEnvironment:
    version = "harness-and-tool-config-v1"

    def run(self, task, policy, request):
        # Delegate to the shared harness. It owns isolated state and cleanup.
        # policy.sample(prompt_tokens, max_tokens=..., temperature=..., seed=...)
        # returns Sample with the ORIGINAL token IDs and behavior log probabilities.
        # Return Trajectory with the requested task/episode/policy identities.
        ...
```

Return each actual model generation as its own `Sample`, including its exact conditioning tokens. Never reconstruct training tokens by retokenizing decoded answers. Earlier assistant turns appearing as later prompt context receive no second loss. Honor turn/token limits, enforce task-specific tool/time budgets in the harness, and clean up on every terminal path. Raise `InfrastructureError` for recoverable infrastructure failures; malformed contracts fail the run. The verifier receives `role="training"` or `"evaluation"`; the external adapter must select the correct private judge/context.

## Configured runs and custom inputs

An input plugin is an importable `module:function`. It receives config JSON and returns the keyword adapters for `Pipeline`: `sft_dataset`, `tasks`, `environment`, `verifier`, and `evaluation_tasks` as needed. No plugin code is fetched or installed automatically.

The runnable [demo config](examples/training-demo.json) demonstrates this boundary:

```bash
python3 -m training_eval run --config examples/training-demo.json \
  --plugin training_eval.mocks:demo_inputs --output /tmp/training-configured
```

For a real run, replace the mock plugin with your dataset/harness integration, use `backend.kind="tinker"`, and set `renderer`, `context_limit`, `rank`, optional `stop`, and a currently supported `model.base_model`. Set W&B `tracking.mode="online"` and `tracking.project` to stream results. Numerical settings in the demo are mock-test values, not recommended model hyperparameters.

Each stage's `steps` is an **attempted-batch cap**, including zero-signal or quarantined batches; `optimizer_step` counts only successful updates. SFT `batch_size` counts examples; GRPO `batch_size` counts task groups and `group_size` counts episodes per task. `checkpoint_every` counts attempted batches; `eval_every` counts successful optimizer updates. Set `eval_every=0` for SFT-only runs without an environment. Each stage transition starts from saved weights with a fresh optimizer.

The GRPO recipe is explicit: one synchronous on-policy update per batch, population-standardized group advantages, no initial KL/clipping/replay, equal trajectory weighting and assistant-token-only loss. Equal-reward groups contribute no update. An unresolved member excludes the entire group; retry once by default, then quarantine. Sampling and optimization do not overlap.

Future strategies implement `name`, `input_kind`, `requirements`, `configuration()`, `build_batch()`, `state_dict()` and `load_state_dict()`. `configuration()` must declare the version and every loss-affecting setting; it is bound into resume compatibility. Register strategies with `StrategyRegistry`. Supply new input kinds through `input_adapters[kind](BatchContext) -> BatchInput` plus immutable `input_bindings[kind]`. Built-in input kinds are `sft` and `rollouts`. The backend remains explicit about supported losses; a critic-based PPO implementation requires additional capabilities rather than silently reusing an incompatible API.

## Evaluate, fork and resume

```bash
# Graceful interruption saves a resumable boundary.
python3 -m training_eval run --config examples/training-demo.json \
  --plugin training_eval.mocks:demo_inputs --output /tmp/training-paused --stop-after-batches 3

# Use the checkpoint path from result.json and the unchanged config.
python3 -m training_eval run --config examples/training-demo.json \
  --plugin training_eval.mocks:demo_inputs --output /tmp/training-paused \
  --resume /absolute/path/to/checkpoint.json
```

For standalone evaluation, set config `model` to `{"checkpoint":"/absolute/manifest.json"}` or a base model, then run `training_eval evaluate` with the same config/plugin/output flags. The plugin supplies `evaluation_tasks`. Final-test splits require `--final-test`; they cannot enter periodic evaluation. Reports include coverage, unresolved counts, immutable policy identity and generation settings. Transient grader/harness failures produce incomplete reports; identity/configuration errors stop execution.

For a fresh training fork, set `model.checkpoint`, choose a **new run_id**, and use `run` without `--resume`. Fork preserves weights and lineage but resets optimizer/client progress. Resume restores the original run configuration, strategy configuration, data/version bindings, cursor and RNG state. Logging output locations can change. Partial saves do not publish manifests. An ambiguous Tinker update poisons that client: restore a known checkpoint into a new backend; never blindly retry the request.

Checkpoints use durable local manifests referencing provider artifacts; multi-host artifact stores can implement `CheckpointStore`. Live tool processes and bit-identical remote sampling are not recoverable. Full manifests stay local; W&B receives opaque checkpoint IDs. This boundary follows the automatic approval restriction encountered while implementing remote checkpoint metadata uploads.

## Observability and validation limits

`logs/events.jsonl` and content-addressed JSON traces are authoritative. W&B receives scalar metrics, nested component/usage metrics, trace artifacts, and up to eight rollout table rows per optimizer step by default. Failed or unselected examples remain in complete local records. Stable event IDs and receipts support replay; ambiguous remote acknowledgments can still yield transport duplicates. `JsonTracker.replay()` retries delivery, and CLI results report external logging warnings without repeating training updates.

Backend tests exercise real SDK request shapes through fixture clients; pipeline tests cover offline SFT/GRPO, immutable evaluation snapshots, fork/resume, stage-boundary optimizer reset, zero-signal groups, retries, isolation boundaries, split checks and extension registration. Existing evaluator tests remain intact. The real Tinker 0.30.4 serialization types and W&B 0.30.0 offline SDK have also been exercised without remote training. A paid live Tinker run and real environment/grader quality remain unverified until those external adapters and credentials are supplied.
