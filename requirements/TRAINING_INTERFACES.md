# Proposed training interfaces

These are the original design contracts. The implementation now lives in `training_pipeline` and `agent_harness.runner`; see [the executable configuration and input contracts](../docs/TRAINING_PIPELINE.md). Record names here remain conceptual rather than literal API names. They define the minimum responsibilities for the next implementation. Record names below are conceptual; adapters preserve current evaluator JSON schemas. See [architecture](TRAINING_ARCHITECTURE.md) for flows and defaults.

## Run, model factory and backend

`RunSpec` contains a version, run ID, ordered stages, input model reference, immutable data/environment/verifier configurations, output location, seed, tracking settings, checkpoint/evaluation schedules and explicit numeric training budgets. Validate it before allocating a trainer. A stage declares `sft` or `grpo` in v1; unimplemented strategy names fail before sampling. Stages can be omitted or repeated.

| Interface | Input → output | Required behavior |
|---|---|---|
| `ModelFactory.resolve` | model reference + purpose + strategy requirements → model bundle | Purpose is evaluate, fork or resume; reject unsupported capabilities and incompatible artifacts |
| `ModelRef` | base-model ID or checkpoint ID | Resolve mutable aliases once, persist the concrete identity |
| `ModelBundle` | resolved identity + capabilities + tokenizer + renderer + sampling/training handles | Evaluation need not allocate a training handle |
| `Backend.capabilities` | model/adaptation config → capability descriptor | Include sampling, token log probabilities, supported losses, optimizer state save/load and context limit |
| `Backend.sample` | immutable policy handle + rendered prompt + generation settings → generated tokens/log probabilities + usage | Preserve model/tokenizer identity and actual stop reason |
| `Backend.update` | validated loss batch + optimizer config → acknowledged step result | Serialize stateful updates; an unknown outcome is not safe to repeat |
| `Backend.save` | completed step + unique artifact names → state and sampler references | Await completion; do not advertise pending references |
| `Backend.load` | artifact + evaluate/fork/resume purpose → appropriate handles | Restore optimizer only for resume |

Tinker is the initial adapter. Arbitrary Hugging Face weights are not promised to load through it. Query the supported model catalog and validate requested adaptation configuration. Renderer selection must match the resolved tokenizer and model family; missing tool rendering support is a capability error. [Supported Tinker models](https://tinker-docs.thinkingmachines.ai/tinker/models/)

Do not require critic/value-head support from every backend. Future strategies declare it if needed; capability mismatch is an actionable error that names the missing requirement.

## Data and training strategies

| Record/interface | Minimum information | Invariants |
|---|---|---|
| `DatasetAdapter` | release manifest, split, stable example IDs, deterministic cursor/order | Only training split enters training; artifact and split identity are fixed for a run |
| `SFTExample` | ID, visible conversation or final-answer example, supervision selection, provenance | Observations are context; accepted examples only; no private references injected as policy context |
| `RenderedExample` | input tokens, next-token targets, aligned target weights, renderer/tokenizer versions | Weights zero outside selected assistant targets; nonempty targets; bounded length |
| `TaskSource` | public task view, private verifier reference, task hash, split/family identity | Private reference never serialized into policy-visible task view |
| `PreferencePair` (future) | same-task chosen/rejected examples, review rationale, provenance | Explicit reviewed preference, not inferred automatically from two candidates |
| `TrainingStrategy` | requirements, batch construction, loss config, state serialization | Own algorithm semantics; backend owns tensor/provider translation |
| `StrategyState` | stage, optimizer step, algorithm-specific schedules and auxiliary policy IDs | Included in every resumable checkpoint |

SFT uses weighted cross-entropy on intended assistant outputs. GRPO consumes complete groups of terminal verifier results and corresponding policy-token data; it does not consume SFT examples as if they were rewards. The initial GRPO variant is synchronous, one update pass, standardized terminal group advantages and an importance-sampling objective; record the exact reduction and loss configuration with the run. Use mean over contributing assistant tokens within each trajectory, then mean over contributing trajectories, so long answers do not receive extra weight solely from token count.

DPO later consumes preference pairs plus frozen-reference log probabilities. PPO later declares policy and any critic/value-loss requirements. The common strategy interface must not imply these algorithms have interchangeable datasets or state. Tinker exposes loss primitives rather than supplying this entire orchestration. [Tinker losses](https://tinker-docs.thinkingmachines.ai/tinker/losses/)

## Environment, runner and verification

`EnvironmentPackage` binds `TaskSource`, `EpisodeFactory`, action schema, versioned tool configuration, limits and verifier. Per-task limits can narrow run limits; effective limits are the stricter values and are persisted. Different budgets create a different experiment identity.

| Operation | Input → output | Required behavior |
|---|---|---|
| `EpisodeFactory.create` | public task + seed + episode ID + effective budgets → isolated episode | New mutable state per rollout; shared immutable snapshots are allowed |
| `Episode.reset` | none → initial policy-visible observation | No reference answers, private claims or grading rubrics |
| `Episode.step` | parsed policy action → observation + termination + usage | Enforce permitted tools and budgets on the host; validate arguments |
| `Episode.close` | terminal or failed episode → cleanup result | Always release tool processes and temporary state |
| `RolloutRunner.run` | policy handle + episode + generation config → trajectory | Same machinery for evaluation and RL; no optimizer mutation |
| `Verifier.verify` | terminal trajectory + trusted metrics + private grading context → verification result | Resolved scalar reward or explicit unresolved status with diagnostics |

`Trajectory` contains run/stage/task/group/episode IDs; immutable policy version; environment, renderer and tokenizer versions; visible observations/actions; per-generation conditioning tokens, generated tokens, token log probabilities and assistant-target masks; final submission; termination; and measured resource usage. Large observations may be content-addressed artifact references. Store the full trajectory before adding it to a training batch.

`VerificationResult` contains verifier/reward version, status (`resolved` or `unresolved`), reward (finite scalar or null), component metrics, reasons and diagnostic artifact references. Generic rewards need not be in `[0,1]`; the strict repository adapter preserves its existing `[0,1]` contract. Preserve its accepted/partial/failed tiers as adapter diagnostics. Infrastructure errors, unavailable evidence and incomplete judge responses are unresolved; they are not valid negative examples.

For the strict adapter, construct existing `AnswerSubmission` and trusted `EpisodeMetrics`, invoke evidence grading, and adapt `GradeReport`. Preserve task/experiment/reward hashes and training/evaluation judge role. An adapter may not accept policy-authored accounting as trusted metrics. Existing reference-rubric scores remain available through a separate evaluation adapter and carry a distinct verifier identity.

Invalid model tool arguments are policy outcomes and can produce a structured tool-error observation within budget. A tool-service outage is infrastructure failure. Deterministic, task-relevant command failure is an observation. Timeout classification depends on whether the policy exhausted the documented task budget or infrastructure failed; persist the classification. Never repeatedly execute a side-effectful tool call just because its response was lost.

### Group and token invariants

- A group contains at least two rollouts of the same task, immutable behavior policy, experiment, reward/verifier version and effective budgets, with independently initialized episodes.
- Before calculating advantages, validate every member. Any unresolved member excludes the whole group; one full-group retry is the proposed default, then quarantine. Log all attempts.
- Use population standard deviation and return zero advantages for zero variance, matching the existing reward bridge. No optimizer update for an entirely zero-contribution batch.
- Every trained action token has a matching behavior log probability under its actual conditioning context. Missing/nonfinite probabilities are invalid training data.
- User and tool tokens have zero policy-loss contribution. For multi-turn episodes, previous assistant turns included as later context are not trained a second time.
- Token IDs, targets, masks, log probabilities and advantages are aligned after next-token shifting. Reject malformed lengths rather than trimming to fit.
- Each optimizer batch uses one pinned behavior policy. Refresh the sampler after an acknowledged update; asynchronous/off-policy collection is deferred.

## Checkpoints and recovery

| Operation | Required artifact | State semantics |
|---|---|---|
| Evaluate | sampling weights plus model/renderer metadata | No optimizer; fixed immutable model identity |
| Fork | compatible train-loadable weights plus metadata | New run and fresh optimizer/counters; retain parent lineage |
| Resume | training state with optimizer plus committed framework manifest | Restore exact stored configuration, cursor, counters and algorithm/client state |

A sampler-only artifact does not imply train-loadable weights or optimizer state. The registry checks purpose-specific availability instead of treating all checkpoint paths as interchangeable.

`CheckpointManifest` contains schema version; immutable checkpoint/run/stage IDs; parent model/checkpoint; backend and model/adaptation configuration; training and sampler artifact references and verified availability; tokenizer/renderer versions; run/strategy configuration; optimizer step and attempted-batch cursor; dataset manifest/hash, order/shuffle state and next cursor; environment/verifier/reward versions; client random state; strategy/auxiliary-policy state; and creation time. Later evaluations are separate immutable records indexed by checkpoint ID, so adding evaluation links does not mutate a committed manifest. Resume must restore local random generators and deterministic data order, not merely an integer seed.

Commit protocol: stop at an acknowledged optimizer boundary → freeze updates → save training state and sampler snapshot → persist framework state → verify required artifacts → atomically publish manifest → expose immutable checkpoint ID → continue. A failure leaves an uncommitted artifact record and the prior committed checkpoint intact. Do not delete backend artifacts automatically in v1. Retention and access must support the declared recovery window; detect expired/inaccessible artifacts during preflight.

Tinker `save_state`/`save_weights_for_sampler` serve different artifact purposes; `load_state_with_optimizer` is required for optimizer recovery. Store provider references without assuming they can be copied between model architectures. [TrainingClient save/load APIs](https://tinker-docs.thinkingmachines.ai/tinker/api-reference/trainingclient/)

Resume rejects changed algorithm, optimizer settings, data identity, renderer, environment or verifier configuration. Intentional changes create a fork. Logging destination and output-directory changes may be allowed and recorded because they do not affect training semantics. An interrupted group is discarded; external tool processes are not resumed. Uncertain remote-update status requires restoring the last known committed checkpoint before replaying work.

## Evaluation and tracking

`EvaluationSpec` binds model/checkpoint, task release and selected split, verifier/judge role, generation settings, seed and effective budgets. `EvaluationReport` records the same identity, task-level results, expected/attempted/resolved counts, aggregate metrics, failure categories and artifact references. Missing evaluations stay missing; never improve a mean by silently dropping unresolved examples without coverage reporting.

The first periodic scheduler pauses at configured optimizer steps, saves an immutable snapshot, evaluates development data, persists the report, then returns control to training. Expected transient evaluation failures produce an incomplete report and permit training to continue; configuration errors or snapshot identity mismatches stop the run. Standalone evaluation uses the same runner. Final-test runs are explicitly requested and excluded from automatic training decisions. Existing rubric and strict reports remain clearly labeled and are not aggregated as one score.

`Tracker` offers run-start, scalar-event, rollout-table, artifact-reference and run-finish operations. Each event carries run/stage, event ID, attempted-batch count, optimizer step and policy/checkpoint ID where applicable. Persist local events before upload; event IDs enable deduplication during reconnection. An upload failure must not replay an optimizer step or mark a checkpoint incomplete.

Proposed W&B views: loss/reward curves; reward component and advantage distributions; exclusion/zero-variance counts; completion and termination rates; tokens, tools, duration and measured cost; checkpoint evaluation curves; sampled rollout table with visible messages, final answer, verifier reasons and links to full artifacts. Default table sampling is deterministic by episode ID with a configurable per-update cap, and full artifacts retain all outcomes including failures. References and private verifier context belong in verifier-only storage, never policy-visible tables or mounts.

Online W&B is opt-in; offline and disabled modes preserve the same local run record. Existing evaluator end-of-run logging stays compatible. Unknown usage/cost fields are nullable with availability flags, not fabricated zeros. Credentials are runtime inputs and are never serialized into configuration, trajectories or checkpoint manifests.
