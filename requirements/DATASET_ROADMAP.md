# Dataset Generation — End-to-End Roadmap

[Editable Excalidraw roadmap](https://app.excalidraw.com/s/8Ufs2ZMhWhu/AJrov8yM1oi)

This document specifies how to generate our own repository Q&A dataset, grounded in the [dataset research](../DATASET_RESEARCH.md) and [project description](../PROJECT_DESCRIPTION.md). It covers source-reading and executable tasks, reproducible environments, independent reference construction, teacher investigations, verification, and versioned releases. It describes planned components, not an implemented or executed pipeline.

The unit of data is a question tied to a repository revision and an environment, with private grading evidence and optional verified investigations. Collection volume, language coverage, split proportions, model choices, attempts per task, review sampling, and resource budgets are versioned configuration, not fixed milestones. Public benchmarks are reserved for evaluation, not used as training question seeds.

## 1. Repository and environment preparation

### A. Shared dataset contract

**Input:** project goals and intended question categories. **Output:** versioned schemas, tool interfaces, fixtures, and collection configuration.

Define categories for localization, cross-file flow, configuration, API behavior, architecture, tests/invariants, runtime behavior, and insufficient evidence. Distinguish lookup from deep investigation and source-reading from execution-required tasks. Capture teacher/generator/verifier identities and versions; record token, tool-call, time, execution-resource, retry, and spending limits for every run. Define review policy and collection coverage targets before launching a batch.

**Complete when:** valid and invalid example records exercise all interfaces; each component can develop against fixtures; configurations identify the rules used to produce every record.

### B. Repository selection, snapshots, and splits

**Input:** candidate repositories and A's coverage rules. **Output:** immutable repository snapshots, provenance records, and a family-level split manifest.

Select repositories, pin full commit hashes, preserve source notices and provenance, and record file hashes, submodules, and dependency manifests. Group forks, mirrors, renamed repositories, and closely related packages. Assign `train`, `dev`, or `test` before question generation. Check overlap with reserved benchmarks and inspect near-duplicate code or documentation across split boundaries. Propagate exclusions into all derived records.

**Complete when:** every admitted snapshot is reproducible, its family has exactly one split, provenance is recorded, and unresolved source or overlap issues are excluded or quarantined.

### C. Environment factory and readiness

**Input:** B's snapshots and A's environment/tool contract. **Output:** immutable environment artifacts and readiness manifests.

Support two capabilities:

- **Source reading:** immutable code and documentation with listing, search, and file-reading tools. Application dependencies are not required unless a task needs execution.
- **Execution:** pinned code plus the language runtime, installed dependencies, fixtures, and required service configuration. Build or adapt an image; verify that it contains the intended revision and can execute representative commands. Record which tests or commands actually passed rather than claiming the entire repository works.

Build flow: **snapshot → environment recipe → build/adapt image → readiness checks → immutable artifact reference**. An upstream environment can be reused only after its revision, dependencies, and readiness are checked. Do not assume a repository dataset supplies an image.

The environment manifest ties an `environment_id` to repository commit, capability, snapshot/image digest, build recipe, dependency/environment details, tool version, readiness status, and build/check logs. Reuse artifacts across compatible questions; rebuild and issue a new environment version when inputs change. Service-backed tasks must include reproducible service initialization and reset instructions.

**Complete when:** a clean instance can be started from the manifest and repeat its required readiness checks. Failed builds and unavailable required services remain environment failures. Quarantine affected executable tasks; never silently downgrade them to source-reading tasks.

## 2. Task and trajectory generation

### D. Candidate question generation

**Input:** B's pinned, split-assigned snapshots, category targets, and C's capability information. **Output:** candidate questions with provenance, category, difficulty, answerability, and environment requirements.

Generate questions from code structure, documentation, tests, and source discussions. Discussions suggest questions; their answers are not automatically ground truth. Resolve historical questions to explicit revisions or exclude them. Avoid giving away answer locations unnecessarily. Keep generation evidence plans private. Candidates may be drafted while executable environments build, but execution-required investigations must wait for readiness.

**Complete when:** each candidate has a stable ID, pinned revision, split, source trail, and explicit environment requirements. Duplicate or underspecified questions are revised or rejected before acceptance.

### E. Independent reference construction

**Input:** candidate questions and the corresponding code/environment. **Output:** private reference answers, atomic claims, supporting evidence, answerability labels, and grading rubrics.

Independently investigate the question and verify each essential claim against pinned code or recorded execution. Store evidence paths, line ranges, hashes, and command results. Distinguish supported behavior from unverified design intent. Resolve ambiguous premises through revision or an explicit insufficient-evidence label. Freeze the reference/rubric version used for each validation run.

**Complete when:** required claims and their support are explicit and checked. Reference material remains outside solver-accessible files, indexes, prompts, and tool responses.

### F. Shared investigation runtime

**Input:** A's tool contract and C's environment manifests. **Output:** sandbox lifecycle APIs, tool execution, observation logs, and replay validation.

The controller and model endpoint run outside the repository sandbox. The controller sends the task and visible history to the model, receives tool calls, executes them in the sandbox, and returns observations until an answer or budget termination. Model weights do not need to be loaded into each repository environment.

Per-attempt lifecycle:

1. Resolve a ready environment version and allocate a clean sandbox.
2. Load its snapshot/image and initialize task fixtures or services.
3. Execute listing, search, reading, and permitted bounded commands.
4. Capture arguments, observations, exit codes, artifact hashes, timings, and costs.
5. Save the trajectory and termination reason, then destroy or reset the sandbox, including service state.

Images can be shared, but mutable files, processes, and service state cannot leak between attempts. Gold records stay outside the sandbox. Repository contents are untrusted data, not collection instructions. Network and execution capabilities are explicit configuration. A container image packages the environment; the infrastructure may run it in a container, VM-backed sandbox, or managed equivalent. No provider or fresh-VM-per-question requirement is imposed here.

**Complete when:** attempts start clean, budgets are enforced, tools cannot reach private grading records, logs survive failures, and replay checks distinguish environment drift from answer errors.

### G. Teacher investigations

**Input:** question, ready environment, shared runtime, and collection configuration. **Output:** candidate trajectories, answers, termination reasons, and cost measurements.

Run a stronger teacher through the same tool interface intended for training and deployment. Give it the question, repository environment, and permitted tools only—not reference answers, generator evidence plans, or hidden rubrics. Collect a configurable number of attempts; retain useful failures and recoveries. Log visible actions and observations without requesting private reasoning traces.

**Complete when:** every attempt is attributable to a task/environment/configuration version and its outcome is recorded, including tool failures and budget exhaustion. A fluent answer is not accepted until H verifies it.

## 3. Verification

### H. Evidence checks, replay, and review

**Input:** E's private grading records, G's candidate investigations, and F's replay support. **Output:** accepted records, a review queue, rejected records, and reasons.

Check snapshot and citation integrity, claim support, factual completeness, tool-schema validity, recorded observations, and required runtime behavior. A valid path alone is insufficient evidence. Compare candidates to private references with a verifier distinct from the solving role. Replay deterministic operations; explicitly report differences from nondeterministic or unavailable dependencies.

Use `accepted`, `needs_review`, and `rejected` states. Human reviewers resolve verifier disagreements, ambiguous references, and consequential claims, and audit a stratified sample under A's versioned review policy. Review all held-out reference tasks before freezing evaluation releases. Measure reviewer agreement and adjudicate disagreements. Corrected questions/references become new versions and their dependent checks must rerun.

Separate infrastructure failures from answer-quality failures. Route question/reference issues to D/E, environment failures to C/F, and collection failures to G. Keep failed attempts for analysis or reviewed preference comparisons; they must not enter SFT simply because they are useful negatives. A preference pair is accepted only with an evidence-backed ranking of attempts on the same task and snapshot under compatible budgets.

**Complete when:** release candidates have explicit decisions, reasons and review provenance; required checks pass; pending records are excluded from accepted exports.

## 4. Release and improvement

### I. Leakage audit and versioned exporters

**Input:** accepted records, environment artifacts, split assignments, and verification history. **Output:** immutable releases with schema/tool versions, manifests, hashes, dataset cards, and quality reports.

Export four distinct products:

| Output | Contents | Boundary |
|---|---|---|
| SFT | Accepted training investigations: task, visible assistant actions, tool observations, and supported answer | Training split only; observations provide context rather than targets to fabricate |
| RL | Training questions, reproducible environment references, and private scorer records | Policy never receives gold answers; fresh rollouts are generated during RL |
| Preferences | Reviewed chosen/rejected investigations with rationale | Training split, same task/snapshot, compatible budgets; rejected attempt is explicitly the negative |
| Evaluation | Separate dev and frozen test tasks, environments, and private references | No export into training; demonstrated solver trajectories are not model inputs |

Audit family separation, near-duplicates, benchmark overlap, private-data visibility, and record references before release. Identical inputs and configuration must reproduce identical exported record content and hashes. Preserve provenance, source notices, exclusions, and deletion propagation. Keep large snapshots and observations in referenced artifacts instead of duplicating them per question.

**Complete when:** manifests resolve, all split and quality gates pass, accepted counts match exports, and the release can be reconstructed from its recorded inputs.

### J. Coverage analysis and targeted collection

**Input:** collection and verification metrics, release reports, and development findings. **Output:** targeted requests for new repositories, environments, question categories, or investigations.

Track accepted/rejected counts by family/category, rejection reasons, readiness/replay failures, reviewer agreement, supported-claim measures, and token/tool/time/cost distributions. Measure cost per accepted example. Use these findings to address missing coverage and recurring failure modes in the next version. Do not lower verification requirements to hit volume targets. Test data remains frozen; if its results guide changes, disclose that use and establish a new untouched test set for final claims.

**Complete when:** each expansion request identifies the observed gap, affected component, proposed collection change, and the development measure used to judge improvement.

## Shared records and parallel ownership

Use JSONL metadata with stable IDs and content-addressed references for large artifacts. Minimum records:

| Record | Required connections |
|---|---|
| Repository / split | Repository and family IDs, source, commit/hash, provenance, split/version |
| Environment | Environment ID, repository commit, capability, snapshot/image digest, recipe, dependency/service setup, tool version, readiness/check logs |
| Task | Task/version ID, repository/environment references, question, category, difficulty, answerability, provenance, split |
| Private reference | Task version, answer, claims, evidence, rubric/version, reviewer history |
| Trajectory | Attempt/task/environment IDs, model/prompt/tool/configuration versions, visible calls/observations, answer, termination reason, cost counters |
| Verification / preference | Linked record versions, checks, decision/reasons; chosen/rejected IDs and ranking rationale when applicable |
| Release | Version, artifact hashes, split manifest, schemas/tool versions, configuration, counts and quality report |

A establishes contracts first. B and F can then develop in parallel; C follows repository snapshots while F develops against fixtures. D follows pinned split assignments. E and G work independently after D; executable attempts additionally require C's readiness and F's runtime. H joins their results, I packages accepted outputs, and J feeds future collection. Exporters and validators can develop early against A's fixtures. A–J denote future agent ownership, not agents started by this document.

## Acceptance scenarios for the future implementation

- A source-reading question succeeds without installing the application's dependencies.
- An executable task starts from a recorded image, initializes fixtures/services, and reproduces its required check.
- A broken environment is quarantined without being counted as an incorrect answer or downgraded silently.
- Two attempts share an immutable image but cannot see each other's changed files or service state.
- The controller/model endpoint is external; attempted access to private references through tools fails.
- Family/near-duplicate overlap is detected before release, and no dev/test records reach training exports.
- Unsupported claims, invalid citations, missing observations, and pending reviews block acceptance; valid recovery from a failed search remains admissible.
- A corrected task/reference invalidates dependent verification until rerun; rejected attempts can appear only as explicitly reviewed preference negatives, never accepted SFT examples.
- Re-exporting the same release inputs yields the same record content and hashes.

Collection sizes and budgets remain configuration. Infrastructure provider, model, and training framework remain architectural choices for their implementation plans. This roadmap defines the complete generation process without implementing it.
