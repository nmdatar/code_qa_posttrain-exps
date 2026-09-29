# Correctness-first evaluation, version 1.0

This package implements the project's agreed evaluation and RL-reward contract.
The code uses Python 3.11+ and Git, with no third-party Python dependencies.
The existing `eval_pipeline` reference-answer baseline is a separate experiment;
its five-axis scores are **not** interchangeable with these accepted/partial/failed
labels or rewards.

## What is implemented and what remains empirical

Implemented: versioned JSON schemas; pinned-source/citation/symbol validation;
two-stage semantic-judge interface; graph normalization/rendering; authenticated
host telemetry; strict quality gates and terminal rewards; split audits and frozen
task/budget manifests; calibration analysis; independent checkpoint scorecards;
repository-clustered comparisons; an adversarial regression suite.

Not claimed: human-reviewed calibration data, demonstrated live-judge accuracy,
a measured baseline, a trained RL checkpoint, or a proven injection-resistant LLM.
`reports/` explicitly records these as pending. Synthetic fixture tests validate
the scoring mechanics, not the truthfulness of a real judge. No generated task
is automatically declared human reviewed.

## Quick start

From the project directory:

```bash
python3 -m unittest discover -s tests -v
python3 -m qa_eval demo --out reports/synthetic-smoke-test.json
python3 -m qa_eval schemas --out schemas
python3 -m qa_eval --help
```

Installing with `pip install -e .` additionally exposes `qa-eval`. The module form
does not require installation. The smoke test creates and removes a temporary Git
repository and evaluates explicit mock judgments. It never calls a model.

## Trust boundary and records

`TaskSpec` contains question, immutable commit, family/lineage/split, atomic required
claims, weight 3 for essential facts or 1 for supporting facts, evidence spans and
hashes, answerability, graph criteria, runtime fixtures and budgets. Only reviewed,
accepted gold can receive a numeric training reward. Duplicate rubric IDs and
identical normalized claim texts are rejected. Reviewers must also merge semantic
duplicates that normalization cannot identify.

`AnswerSubmission` contains text, citations and optional graph/Mermaid. It cannot
carry latency, tool histories, rewards, or trusted execution results. Extra fields
are rejected. Its hash binds the full submission; oversize answers are invalid,
never graded on a truncated prefix. Empty answers fail.

`EpisodeMetrics` is measured by the trusted host runner. Use `EpisodeRecorder` to
time dispatch-to-answer, record provider token usage, time actual tool operations,
and retain retries, costs and trajectory references. All generation/retry usage
must be recorded; do not put grader usage into answer metrics. Cost uses one
currency per experiment. Execution/probe attestations come only from the trusted
runner, not the model. The signing key is never mounted in the agent environment.

`GradeReport` records deterministic findings, complete semantic findings, quality
tier, bounded reward, raw trusted metrics, provenance and planned task cohort.
Unresolved judgments return `reward: null`; never coerce this to zero in RL.
Drop/retry the whole GRPO group when unresolved grading makes its relative reward
comparison incomplete. Log excluded groups and causes.

`qa_eval.rl.prepare_group(reports)` implements this trainer handoff and returns
rewards plus standardized group advantages. It rejects held-out or mixed-task
groups and returns zero advantages for equal rewards. A PPO/GRPO optimizer,
rollout model and compute allocation remain external integrations; this utility
does not train weights. Mask tool observations in the actual optimizer loss.

Authoritative artifacts are HMAC-SHA256 envelopes bound to task/submission hashes.
The `attest` CLI is a supervisor utility, never an agent tool. A valid signature
proves provenance from the key holder, **not** that an LLM or human is correct.
Protect that key and the trusted judge code with process/container/filesystem
isolation. The CLI checks that grader inputs, outputs and key are outside the
agent repository. These checks are not a general OS sandbox.

## Evaluation order

1. Validate contracts, signatures, frozen task membership and dataset role.
2. Check clean pinned Git snapshot, gold evidence, paths, hashes, line ranges,
   Python symbols, allowed tools, hard limits, probes and graph rendering.
3. Extract every assertion from prose and graph with an independent model pass.
4. Assess required claims and all extracted assertions against supplied source
   evidence. Validate ID coverage and evidence references programmatically.
5. Assign tier before reading any efficiency values for reward calculation.
6. Apply the accepted-only efficiency bonus, then emit a signed report.

Repository and gold defects are unresolved. Missing/unverifiable source evidence
is distinct from proof of a false assertion. An invalid candidate citation is an
output defect; a verified material contradiction is a failure. Unsupported,
nonmaterial or uncited additions block acceptance even when all required points
are present. Additional facts never earn completeness credit. A submitted citation
must participate in a supported claim link; repeated supported citations add no
reward. Every extracted assertion needs an assessed supporting citation.

Actual snapshot discrepancies quarantine the task. An answer about the wrong
version, against an otherwise valid environment, fails if the judge identifies
a material falsehood. Reference probes that disagree with gold quarantine the
task instead of blaming the model.

Python AST symbol checks are optional and exact; other language symbol analysis
is not implemented. Omit `symbol` for other languages and use spans plus semantic
assessment. Evidence must be tracked in the pinned source tree. Symlink evidence,
submodules, dirty trees, ignored/untracked source additions and unsupported graph
syntax are rejected or quarantined explicitly. Isolate runtime work in another
checkout and keep the grader's evidence checkout pristine.

## Semantic adapter and data limits

The external judge command reads one JSON request from stdin and writes one JSON
response to stdout. `ClaimExtraction.json` and `JudgeOutput.json` define the two
stages. Prompts mark candidate text and source content as untrusted data. The
judge sees no performance metrics or candidate model identity. It sees execution
facts needed to assess claims such as “tests passed.”

The extractor receives a tracked-file catalog and can request alternative hashed
evidence spans. The assessor receives required evidence, candidate citations and
those alternative spans. This is a bounded two-stage adapter, not an unrestricted
repository-search agent. If more investigation is needed, return `needs_review`
and retry with an expanded, explicitly reviewed evidence package. Oversize source
catalogs/evidence are unresolved rather than silently truncated.

The optional HTTP adapter supports chat-completions-compatible endpoints:

```bash
# Configure values locally; do not commit secrets.
export QA_TRAIN_JUDGE_BASE_URL=https://YOUR_TRAIN_PROVIDER/v1
export QA_TRAIN_JUDGE_MODEL=YOUR_TRAIN_JUDGE
export QA_EVAL_JUDGE_BASE_URL=https://YOUR_OTHER_PROVIDER/v1
export QA_EVAL_JUDGE_MODEL=YOUR_EVAL_JUDGE
# QA_TRAIN_JUDGE_API_KEY and QA_EVAL_JUDGE_API_KEY are optional env secrets.
```

Use exact versions and genuinely different model families. Family names in config
are operator attestations; the software cannot infer model lineage from a provider
alias. The HTTP adapter routes roles to separate environment settings. It makes
two calls per completed judgment; live costs are external to answer efficiency.
No live endpoint has been exercised as part of the included smoke test.

## Diagrams

Canonical graph nodes: `id`, `entity`, `label`, `citations`. Edges: `source`,
`target`, `relation`, `condition`, `citations`. Allowed relations are calls,
contains, inherits, data_flow, precedes and signals. Graph citations refer to
submission citation IDs. Stable entity IDs carry meaning; labels carry display
text. Unknown endpoints and malformed structures are rejected.

Supported Mermaid subset, one declaration/edge per line:

```text
flowchart TD
Executor["Executor"]
Worker["Worker"]
Executor -->|signals;when cancellation is requested;src1| Worker
```

`flowchart LR` is also accepted. Node IDs become canonical entities. Edge labels
must have `relation;condition;comma-separated-citation-IDs`; empty conditions are
allowed. Sequence diagrams, subgraphs, styles, HTML, links and implicit nodes are
not supported. Use canonical JSON for more precise entities or citations.

Rendering uses deterministic source → labeled relationship → target rows, with
stable ordering, deduplicated edges and escaped labels. Repeated boxes carry the
same entity identity. The judge evaluates graph semantics and abstraction rules;
the renderer does not infer correctness from graph equality. An optional diagram
can lower quality if wrong, but cannot raise correctness or earn a format bonus.
Raster-image answers and general diagram layout optimization are out of scope.

## Reward contract

| Tier | Rule | Reward |
|---|---|---|
| Failed | Invalid/no usable answer, verified integrity violation, or material falsehood | 0 |
| Partial | Missing facts, grounding or output defects without established material falsehood | 0.2 × C |
| Accepted | All required facts and substantive additions supported; citations/diagrams valid; no uncertainty | 0.9 + 0.1 × E |
| Unresolved | Bad/ambiguous gold, judge failure/disagreement, untrusted metrics or infrastructure failure | null |

`C` is weighted supported coverage. Partial required-claim coverage earns 0.5 only
when separable subparts were preregistered; otherwise it earns zero. Generic refusal
on an answerable question earns zero. Proper abstention/clarification requires its
own gold rubric and evidence.

Compute units = input tokens × input rate + output tokens × output rate + tool
seconds × tool rate. Rates are fixed in the experiment; examples are illustrative,
not measured economic estimates. Include all retries. Hard tool/output-token
limits are separate from soft latency/compute normalization budgets.

Efficiency(x) = max(0, 1 − x/budget). With trustworthy timing, E averages compute
and latency efficiency; otherwise E uses compute only for the entire experiment.
Tool count and time-to-first-token are diagnostics, never rewards. Tool limits must
be enforced by the runner; validation also detects exceeded limits after the fact.

Partial reward is never above 0.2; accepted reward is never below 0.9. This guarantee
is conditional on correct grading. Average reward is not a checkpoint promotion
criterion, because a policy could trade some successful answers for aggregate cost.

## Calibration, split audit, and frozen experiments

Start with at least 100 tasks across 10 repository families. To generate a draft
queue from existing clean Python checkouts, provide a JSON list of `path`, `url`,
and `family_id` (at least 10 unique families):

```bash
python3 -m qa_eval seed-calibration --repositories repositories.json --out draft-tasks.json
python3 -m qa_eval audit-splits --tasks all-splits.json
```

This extracts 10 top-level functions per family and seeds localization, flow/diagram,
boundary, and answerability investigations. It deliberately marks all tasks draft
and unreviewed. Behavior/intent tasks contain an explicit review-required criterion
that must be replaced with atomic evidence-backed facts. It is a review-queue
generator, not a source of automatically certified gold. Select nontrivial symbols,
edit poor questions and attach execution fixtures as needed.

Independently generate candidate answers and seed wrong variants; double-review a
separate calibration sample. Review rows have `task_id`, `family_id`,
`predicted_tier`, and two `reviews` with distinct `reviewer` names and a `label`
of `accepted`, `partial`, or `material_error`. Labels must agree or be adjudicated
before inclusion. Report disputed rows; do not silently drop them.

```bash
python3 -m qa_eval calibration --reviews reviewed-examples.json --out calibration.json
```

Gates use 95% Wilson bounds: detection lower bound ≥95%; false-acceptance upper
bound ≤2%; false-rejection upper bound ≤5%. A small perfect sample is insufficient.
Variants of the same task cannot be counted independently within a calibration
stratum. These item intervals do not eliminate repository correlation: maintain
broad family coverage and inspect family-level failure patterns. Expect to expand
beyond 100 examples, particularly to establish a ≤2% false-acceptance rate.

Choose category budgets using baseline host measurements, then freeze one split:

```bash
python3 -m qa_eval freeze --config examples/experiment-config.json \
  --tasks reviewed-tasks.json --baseline baseline-measurements.json --out frozen.json
```

Set actual endpoints, accounting and environment identity in the config first.
The freeze operation hashes the baseline artifact, category budgets and full task
specifications; it does not invent measurements or validate their scientific quality.
Any task/budget change requires refreezing. Semantic/rule changes require a new
reward version and fresh calibration; do not reuse old raw rewards.

## Trusted grading and isolated runtime hooks

Generate a random 32-byte key outside agent mounts and restrict it to mode 0600.
Use `EpisodeRecorder.finish` on the host to emit an authenticated metrics envelope.
For existing trusted runners, `attest EpisodeMetrics` validates/signs their record.
Never sign an agent-provided metrics object.

Runtime probes must execute in a separate externally managed sandbox. This package
only consumes authenticated attestations: fixture ID, commit, isolated flag,
completion status and stdout SHA-256. Supply matching expected hashes in TaskSpec.
No repository code or shell probe runs on the grader host. A sandbox executor is
an integration prerequisite for runtime tasks, not something a boolean creates.

```bash
python3 -m qa_eval grade --task task.json --submission answer.json \
  --metrics metrics.signed.json --experiment frozen.json --repo /path/to/clean/repo \
  --key /trusted/key --role evaluation --out grade.signed.json --svg answer.svg
```

Use `--semantic` for authenticated replay of an earlier assessment with identical
task, answer, experiment, judge identity and snapshot bindings. Training rejects
development/final-test tasks. The grading process must not expose gold, keys or
judge artifacts to the policy, including through any shared tool server.

## Independent scorecards and promotion

```bash
python3 -m qa_eval scorecard --reports signed-grade-array.json --key /trusted/key --out scorecard.json
python3 -m qa_eval compare --baseline baseline-grades.json --candidate candidate-grades.json \
  --key /trusted/key --out comparison.json
```

Compare identical task cohorts, repeat counts, scoring contracts, environment and
evaluation-judge versions. Missing preregistered tasks remain visible and block
promotion even when both checkpoints omitted them. Unresolved episodes, missing
telemetry, fewer than 10 families, non-final-test data and unregistered candidates
also block promotion. Neither command deploys anything.

Report correctness, completeness, citation support, unsupported assertions,
answerability, diagram quality, latency p50/p95, compute and all-attempt cost per
accepted answer. Unsupported/citation diagnostics use assessed subsets and explicitly
report their coverage; do not mistake unknowns for zero errors.

Bootstrap paired per-task differences by repository family (5,000 resamples).
Quality eligibility requires a positive lower 95% bound on accepted-rate gain,
no observed material-error increase, and an upper bound ≤0.5pp on material-error
increase. This is a conservative implementation of “no supported regression.”
Efficiency eligibility requires accepted-rate lower bound ≥−1pp and material-error
upper bound ≤+0.5pp, plus a negative upper bound for latency or compute delta on
tasks both checkpoints answer correctly on every repeat. Latency comparisons are
disabled when timing is unreliable. Costs of failures stay in total cost.

An eligible comparison is explicitly `eligible_pending_human_audit`, never an
automatic approval. Require passed independent calibration, stratified human review,
and a frozen release-candidate decision before promotion. Use development data for
selection, record final-test accesses, and rotate exposed final suites; access policy
and real reviewer identity remain governance responsibilities.
