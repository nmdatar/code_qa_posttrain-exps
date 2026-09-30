# Training design decision log

Baseline: v0.1, 2026-09-28. Status labels distinguish user choices from technical proposals. No outstanding item prevents reviewing the diagrams.

## Confirmed scope

| ID | Decision | Basis |
|---|---|---|
| D01 | Reusable framework; repository Q&A is the first real adapter | User selected reusable framework and repository Q&A |
| D02 | Tinker-supported models and compatible checkpoints initially; backend interface permits later providers | User selected Tinker-first model scope |
| D03 | Implement SFT and GRPO first; allow future algorithms after SFT without rewriting orchestration | User explicitly narrowed initial algorithms |
| D04 | Multi-turn tool environments plus single-response tasks | User selected multi-turn support |
| D05 | Checkpoint evaluation, fresh training from weights, and optimizer/client-state resume | User selected all three operations |
| D06 | Periodic and standalone evaluation; periodic evaluation pauses training | User selected both and synchronous evaluation |
| D07 | Overview plus focused GRPO and checkpoint views | User selected multiple coordinated views |
| D08 | Requirements and diagram iteration precede training implementation | Explicit deliverable boundary |

## Proposed technical defaults for review

| ID | Proposal | Reason / consequence |
|---|---|---|
| P01 | Algorithm plugins declare data, backend and auxiliary-model requirements | DPO need not pretend to consume online RL episodes; full PPO can declare critic requirements |
| P02 | Synchronous on-policy GRPO; one update pass; importance-sampling objective; no initial reference KL or replay | Smallest explicit initial variant; this is a design proposal, not a claim that it is the best learning recipe |
| P03 | Population-standardized within-task advantages; equal rewards contribute zero | Matches the existing strict reward bridge; track zero-variance groups |
| P04 | Mean loss across assistant tokens within a trajectory, then across contributing trajectories | Prevent accidental length weighting and duplicate training of historical context |
| P05 | Unresolved verifier member excludes its whole group; one full-group retry, then quarantine | Preserve group comparability; avoid treating infrastructure failure as model error |
| P06 | Strict evidence-aware repository verifier for training, subject to calibration prerequisites | Existing reference-rubric evaluation remains a distinct scoring method |
| P07 | Durable local artifacts and events first; configurable W&B online/offline/disabled mode | Logging outages cannot lose recovery state or cause repeated optimization; online mode requires configured credentials |
| P08 | Immutable manifests published only when required artifacts exist; no automatic checkpoint deletion | Readers never consume incomplete checkpoints; retention policy deferred |
| P09 | No automatic best-checkpoint selection or early stopping | Development evaluation informs review; final test remains outside training decisions |
| P10 | Explicit run parameters rather than universal numeric training defaults | Model, learning rates, group/batch sizes, budgets and cadence require a measured smoke test |
| P11 | Expected transient evaluation failures yield an incomplete report and allow training to continue; invalid configuration or snapshot mismatch stops the run | Preserve coverage and attribution without treating missing results as passing scores |

## Review questions before trainer implementation

1. Accept or revise P02–P04: exact GRPO variant, optional reference KL, and loss reduction. A recipe change must be named and versioned rather than silently called the same experiment.
2. Confirm the repository verifier's calibration evidence and judge/provider configuration. The existing strict contract is reusable, but a design document does not establish live grader quality.
3. Choose the first available model, approved training release, tool budgets, numeric run settings, and W&B project. These are experiment inputs; no credentials belong in this package.
4. Confirm artifact storage and recovery-window requirements before a paid run. Local storage is the initial design default; multi-host/cloud storage and retention are future decisions.

## External feasibility constraints

- Tinker supports its hosted model catalog and adaptation capabilities, not arbitrary model architectures supplied by this factory: [model catalog](https://tinker-docs.thinkingmachines.ai/tinker/models/).
- Tinker's PPO loss accepts caller-provided advantages; full learned-critic PPO requires a separate capability investigation: [losses](https://tinker-docs.thinkingmachines.ai/tinker/losses/).
- Fork and resume require different loading behavior: [TrainingClient APIs](https://tinker-docs.thinkingmachines.ai/tinker/api-reference/trainingclient/).
- Environment abstractions can support separate episodes and grouped rollouts: [environment documentation](https://tinker-docs.thinkingmachines.ai/tutorials/cookbook-abstractions/env-and-envgroupbuilder/).
- W&B metric logging does not itself supply complete rollout inspection; provide explicit tables and artifact links: [rollout logging](https://tinker-docs.thinkingmachines.ai/cookbook/rl/rl-logging/).

Recheck provider capabilities when implementation begins. Current interfaces describe required behavior, not a frozen SDK method signature guarantee.

## Revisions

| Version | Change | Status |
|---|---|---|
| v0.1 | Initial architecture, contracts, acceptance scenarios and three editable diagrams; aligned with the existing dataset roadmap | Proposed for detailed review |
