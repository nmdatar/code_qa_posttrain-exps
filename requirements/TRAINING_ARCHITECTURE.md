# Training and evaluation architecture

Implementation status: the SFT/GRPO machinery is now implemented; see [usage and current boundaries](../docs/TRAINING_PIPELINE.md). The design below records the original requirements. See [interface contracts](TRAINING_INTERFACES.md), [dataset roadmap](DATASET_ROADMAP.md), and the [requirements index](README.md) for diagrams and review decisions.

## Goal and first implementation boundary

Build a reusable pipeline in which a model can start from a supported base model or compatible checkpoint, optionally undergo supervised fine-tuning (SFT), undergo one or more post-training stages, and be evaluated at immutable snapshots. The first implementation will provide SFT and GRPO on Tinker, repository Q&A as the first real environment, and a tiny deterministic environment for contract tests. The user subsequently authorized machinery implementation; current behavior and validation are documented separately.

“Any model,” “any dataset,” and “any environment” mean extensible adapters with validated contracts. They do not promise that Tinker can train arbitrary external weights, that arbitrary JSON is already a valid training dataset, or that every algorithm is supported by every backend. Tinker model availability must be resolved at run creation, not embedded as a frozen catalog in this design. [Tinker models](https://tinker-docs.thinkingmachines.ai/tinker/models/)

DPO and full PPO are extension points, not first-release implementations. DPO consumes reviewed preference pairs and a frozen reference policy; it is not an environment rollout loop. PPO may additionally need a critic and value-training capability. Availability of a clipped PPO policy loss does not by itself establish a complete actor–critic trainer. Validate algorithm requirements explicitly. [Tinker loss functions](https://tinker-docs.thinkingmachines.ai/tinker/losses/)

## Repository baseline and compatibility

The repository already contains two distinct evaluation paths:

- `eval_pipeline` runs synthetic demos, retrieval baselines or imported answers, produces reference-based rubric scores, and uploads optional W&B reports after evaluation. Those scores are not independent source verification and must not silently become strict RL rewards.
- `qa_eval` provides evidence-aware schemas and grading, trusted episode telemetry, independent comparisons, and `prepare_group`, a minimal GRPO reward bridge. It does not update model weights. The bridge excludes an entire group when any reward is unresolved, accepts only training-role/train-split reports, checks task/experiment/reward compatibility, and uses population-standard-deviation advantages with zero advantages for equal rewards.

Preserve both existing command-line paths and report semantics. Introduce adapters around their public behavior instead of rewriting their schemas to fit a generic environment. The stricter evidence verifier is the proposed repository-Q&A training default, subject to its existing calibration prerequisites; synthetic fixtures cannot count as measured model quality.

The dataset roadmap is a proposed collection plan, not an available training release. Its `dev` and `pilot_test` names differ from the strict evaluator's `development` and `final_test`. Adapters map `dev` to `development`; keep pilot-test provenance explicit and never silently relabel a previously inspected pilot set as an untouched final test. Require explicit evaluation-role assignment for a release. Public benchmark tasks remain evaluation-only unless a separately reviewed data policy changes that decision.

## Components and ownership

| Component | Responsibility | Boundary |
|---|---|---|
| Run orchestrator | Validate a stage sequence, own counters, coordinate training, checkpointing and evaluation | No reward formulas or provider tensors |
| Model factory | Resolve model/checkpoint, tokenizer, renderer and capability requirements | No assumption that all models share a chat template |
| Backend adapter | Sampling, token log probabilities, forward/backward, optimization, state and sampler saves | Tinker SDK details stay here |
| Data adapter | Produce validated SFT examples, task streams, or future preference pairs | Split and provenance checks before consumption |
| Training strategy | Declare inputs/capabilities, assemble losses, manage strategy state | SFT and GRPO first; registered extensions later |
| Environment package | Supply tasks, isolated episodes, tools, budgets and a private verifier | Environment never receives optimizer access |
| Shared rollout runner | Drive model/action/observation cycles and trusted telemetry | Reused by RL and evaluation |
| Verifier adapter | Return resolved rewards or explicit unresolved results and diagnostics | Private references excluded from policy context |
| Checkpoint registry | Commit manifests, lineage and artifact references | Consumers resolve immutable checkpoint IDs |
| Evaluation runner | Evaluate fixed model identities on fixed task manifests | No weight updates or automatic final-test selection |
| Tracker and artifact store | Live metrics, sampled tables, complete local artifacts | W&B is a presentation sink, not recovery state |

The environment package is a composition root: task source + episode factory + action/tool schema + verifier + versioned configuration. This allows a repository environment and a deterministic toy environment to share a runner without sharing task-specific fields.

## End-to-end flows

### Model loading and SFT

Resolve the input reference and selected algorithm before allocating expensive resources. Validate base-model architecture, adaptation configuration, tokenizer and renderer compatibility, context limits, and required capabilities. A checkpoint reference must identify the intended operation: evaluate, fork, or resume.

For SFT, the data adapter reads an immutable training release. The model renderer converts accepted answer or tool-trajectory examples into aligned tokens and target weights. User messages, repository content and tool observations are context; only selected assistant outputs are supervised. Reject empty-target examples and overlength examples with recorded reasons; do not silently truncate away answers, tool boundaries or evidence. Repeated or optional stages are allowed, including base → GRPO and SFT → checkpoint → GRPO.

### GRPO update boundary

1. Pin an immutable sampling policy version for the update. Select training tasks and instantiate independent episodes with the same task, environment/verifier versions and budgets within each group.
2. Collect a configured number of independent trajectories per task. Every assistant generation records the actual conditioning tokens, generated tokens and behavior-policy log probabilities. Execute permitted tools in isolated episode state and append their observations as context.
3. Verify terminal trajectories with private grading inputs. A valid but wrong answer is a scored outcome; unavailable infrastructure or incomplete grading is unresolved, not a zero reward.
4. Exclude the entire group if any member is unresolved. Default to one bounded full-group retry, then quarantine the group and log the reason. Never resample repeatedly until a favorable group appears.
5. For complete groups, use `A_i = (r_i - mean(r)) / population_std(r)`. Equal rewards produce zero advantages. Retain these groups in diagnostics; skip their zero-contribution update data. If no contributing groups remain, perform no optimizer step and still advance/log the attempted task cursor.
6. Assemble assistant-token-only training rows, preserving behavior log probabilities and prompt contexts. Proposed first GRPO variant uses one on-policy update pass per collected batch and Tinker's importance-sampling policy objective, with no reference KL term initially. Record that variant in the configuration; clipping, KL regularization, replay and multiple epochs require explicit strategy changes.
7. Await successful optimization, increment the committed optimizer-step counter, and create a new sampling snapshot before collecting the next batch. No asynchronous stale-policy rollouts in the first implementation.

Multi-turn data must not count earlier assistant tokens repeatedly when they appear in later turn prompts. Every generated token contributes once; tool observations never become policy actions. A context overflow is an explicit termination, not permission to substitute a different prompt and retain incompatible log probabilities.

### Evaluation and checkpoint reuse

Standalone evaluation accepts a base model or immutable sampling checkpoint. Periodic evaluation runs on development data at a configured committed step: pause training, commit a snapshot, evaluate it, persist results, then continue. A failed evaluation records incomplete coverage and does not become a passing score. The default is no automatic early stopping or best-checkpoint selection; those policies need explicit configuration and may use development data only.

Evaluation records the model/checkpoint ID, task manifest, environment/verifier versions, budgets, generation settings, coverage and failure counts. Comparisons require compatible settings and report differences when they are intentionally varied. Final-test tasks do not enter training, routine periodic evaluation or checkpoint selection.

## Recovery, tracking and failures

A resumable checkpoint binds backend training state, sampling weights and framework state from the same completed optimizer boundary. Freeze updates while these artifacts are saved. Publish the manifest last; failed or incomplete saves cannot replace the last committed checkpoint. A sampler-only snapshot can be used for evaluation but is not advertised as resumable.

Evaluate loads sampling weights. Fork loads compatible training weights with a fresh optimizer and new run ID. Resume restores optimizer state, strategy state, counters, data order/cursor and client random state under the original immutable configuration. Tinker distinguishes weights-only `load_state` from `load_state_with_optimizer`; the backend must use the optimizer-aware operation for resume. [Tinker TrainingClient](https://tinker-docs.thinkingmachines.ai/tinker/api-reference/trainingclient/)

Recovery starts at the last fully committed boundary. In-flight tools and partial rollouts are discarded and may be recollected; remote sampling is not promised to be bit-identical. An ambiguous optimizer-request failure stops the run and requires restoring a known checkpoint rather than blindly retrying an update that may already have happened.

Log scalars during training, not just after a run: loss, reward/component distributions, group exclusions, zero-variance groups, completion rates, token/tool/time usage and measured costs. Stream sampled rollout tables with answers, visible tool actions/observations, verifier explanations and policy/checkpoint identity. Keep all complete trajectories and manifests in durable artifacts. Unknown costs remain unknown, not zero. Host-side telemetry stays authoritative.

W&B may run online, offline or disabled. Write local events/artifacts first and retain them on W&B outages; record upload failures and reconnect without repeating optimization. Never expose private reference/rubric fields to the policy through logs or tool mounts. Verifier-only diagnostic artifacts may contain grading evidence but require separation from policy-accessible storage. Do not collect hidden reasoning traces or credentials.

## Proposed defaults and implementation gate

Defaults for review: synchronous sampling/update/evaluation; terminal scalar rewards; train-only SFT/GRPO; strict repository verifier; population-standardized group advantages; no reference KL, replay or multi-epoch GRPO initially; one full-group retry; local durable artifacts; explicit online W&B opt-in; immutable checkpoints without automatic deletion.

Require run configuration for base model, SFT learning rate, GRPO learning rate, batch/group sizes (group size at least two), token/tool/time budgets, maximum steps, sampling settings, checkpoint cadence and evaluation cadence. There is no meaningful universal numeric setting before the smoke test. Validate these values before training rather than inventing silent production defaults.

Before trainer implementation, review the system overview, then rollout and checkpoint diagrams; resolve or explicitly accept the decisions tracked in this package. Acceptance scenarios must exercise the real adapters and deterministic fixtures before any paid live run is considered successful.
