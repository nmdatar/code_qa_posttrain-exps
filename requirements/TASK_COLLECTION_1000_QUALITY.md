# Quality gates for the 1,000-task collection

This is the collection and release policy for scaling repository Q&A beyond the first 12 development tasks. The user permits benchmark reuse. An imported row is not automatically a verified answer, a runnable environment, an accepted training example, or a newly generated task. Report those counts separately.

## Collection target and admission

Target approximately 1,000 unique, source-reading repository Q&A tasks with preserved original prompts, pinned repository snapshots, private references, and working bounded search/read tools. Source-reading tasks are legitimate when their original question can be answered by source inspection. Keep execution-required tasks in a separate queue until their required runtime is ready; never quietly change their task type to meet a count.

Use existing benchmark questions only under recorded source terms. Retain their original benchmark IDs and splits in provenance. Our family split is an additional field, not a claim to preserve the benchmark's experimental protocol. Benchmark-derived tasks are evaluation/development-only by default because benchmark contamination is intentional and known. New training material requires separate admission and new repository families. Existing Pydantic, TanStack Query and SQLAlchemy families remain development.

For each source preserve its dataset/repository revision, fetched artifact SHA-256, source URL, upstream row ID, original prompt and answer, license/terms evidence, upstream repository URL and commit, original split, transformation version, and lineage. Preserve a raw immutable source artifact before normalization. Do not reinterpret the latest repository commit as the commit that an old reference answer described.

## Per-record gates

| Gate | Pass condition | Failure disposition |
|---|---|---|
| Provenance | Stable source record, revision and content hash; recorded source terms | Quarantine missing or unclear records |
| Question | Nonempty complete original question, no placeholder, answer dump, or missing attachment; task scope recorded | Review ambiguity, quarantine unrecoverable omissions |
| Repository | Full commit exists, canonical family assigned before import, clean immutable snapshot | Quarantine unavailable or incompatible revision |
| Reference | Nonempty original reference preserved outside solver files; factual claims not fabricated from question | Quarantine missing references or keep explicitly ungraded candidates outside verified count |
| Evidence | Every cited path and line range exists at that commit; file hashes bind evidence to exact source bytes | Reject invalid citations; distinguish integrity from semantic support |
| Duplicates | Original ID and normalized exact-prompt hashes unique; cross-source near-duplicates grouped under shared lineage | Select one representative; preserve duplicate mapping |
| Split | No repository family or lineage spans train/development/test in joint release audit | Stop release; never resolve by silently moving established families |
| Environment | Immutable source snapshot, bounded read-only list/search/read interface, readiness and snapshot attestation | Quarantine unavailable tasks; no execution capability implied |
| Privacy | Public task allowlist excludes answer, gold assertions, grading hints and reference metadata; source environment contains no private imports | Stop release on exposure |
| Review | Per-task semantic and difficulty status recorded independently of mechanical checks | Unreviewed is explicit; never infer approval from sampled siblings |

Original references without citations are not upgraded to citation-verified gold by adding a plausible path. A source-reference span that merely exists is `evidence_integrity=passed`; its support for the answer is a separate `semantic_review` finding. References imported as one whole claim remain coarse grading drafts until decomposed and reviewed.

Use deterministic source assertions for source-reading tasks: exact commit, nonempty tracked source inventory, cited file SHA-256, valid line bounds, and expected cited text hashes when available. Name these `source_integrity_assertions`; they do not replace executable behavior probes or prove every sentence in a reference answer.

## Quality state and count vocabulary

Store private sidecar records keyed by task ID and canonical public/private task hashes. Suggested independent fields:

```json
{
  "task_id": "upstream-task-id",
  "source_type": "benchmark_import",
  "task_kind": "source_reading",
  "provenance_status": "passed",
  "schema_status": "passed",
  "snapshot_status": "passed",
  "evidence_integrity_status": "passed",
  "environment_status": "ready",
  "semantic_review_status": "not_reviewed",
  "difficulty_status": "not_measured",
  "human_reviewed": false,
  "training_eligible": false,
  "gold_status": "draft"
}
```

Count `imported`, `mechanically_validated`, `runnable`, `semantically_reviewed_supported`, `needs_revision`, `quarantined`, `human_accepted`, and `training_eligible` separately. A runnable release can contain mechanically validated reference drafts, but its dataset card must not call all records independently verified or high-quality gold. Do not inflate counts using paraphrases, repeated prompts at multiple revisions, or one record per subclaim.

## Scalable semantic review

Perform cheap gates over every record. Review a deterministic stratified sample across repository families, upstream sources, question categories, reference lengths and evidence structures. Use task-hash sorting within strata so the sample is reproducible and cannot be selected only for successful examples. Start with at least one task per represented family plus targeted edge cases, then expand high-risk strata and all detected failure clusters. A larger random sample can estimate aggregate error rate only when selection probabilities and labels are recorded; a purposive sample cannot justify a population error-rate claim.

Independent reviewers see original prompt, private reference and actual cited source. They record answerability, question/reference consistency, material unsupported claims, obsolete APIs, misleading context, and answer leakage. Preserve reviewer identity/type, exact task and reference hashes, findings, and revisions. Agent review is not human review. If any systematic defect appears, stop promotion for its entire source/transformation stratum, repair the importer or affected tasks, and rerun all mechanical gates before resampling.

Sample question-only controls separately to assess whether repository access adds value. Missing citations alone are not evidence of difficulty. Preserve exact prompts, model identity when available and unknown usage values as null. An easy question may be useful learning material, but should not acquire a hard-reasoning label. Do not spend 1,000 solver investigations before knowing whether task acquisition and gold quality are adequate.

## Minimal implementation changes

Existing `dataset_builder.build.validate_spec` requires executable assertions, and `prepare` always permits `python_probe`. Keep that behavior for executable tasks. Add an explicit source-reading import path rather than manufacturing empty or tautological executable probes. Existing `qa_eval.schema.TASK` accepts `probes=[]`, and `dataset_builder.contracts.PUBLIC_TASK` already supports a restricted permitted-tools allowlist.

A minimal pipeline can expose `import -> validate -> prepare-environments -> audit -> package` commands. Inputs are a pinned source artifact, immutable split manifest and selected task IDs. Outputs are public tasks/environments, private references/provenance/quality records, rejection and duplicate ledgers, snapshot inventories, and a deterministic release manifest. Validation should be resumable by artifact hash; environment work should be keyed by repository commit rather than task. Prepare each unique snapshot once and reuse its attestation for all bound tasks. Preserve upstream answer text rather than inventing a detailed rubric at import time.

For source-reading execution, mount only immutable repository content, keep the controller and grading outside the sandbox, and permit only list/search/read. A common tool image can serve many snapshots if the per-attempt snapshot digest is verified before tools run. If using one image per repository, build concurrently with bounded fan-out and collect live readiness evidence per unique environment. The existing investigation runtime requires a built immutable image; a new local read-only snapshot runner must not be described as Modal sandbox execution unless it actually ran there.

## Throughput measurements and release audit

Append each optimization to `TASK_GENERATION_OPTIMIZATIONS.md`: bottleneck, change, invariant preserved, before/after measurement where comparable, and adverse findings. Track unique tasks per minute, unique environment builds, acquisition failures, invalid evidence, deduplicated records, semantic sample support/revision rates and review corrections. Record tokens and monetary cost only when measured.

Before release, independently recount JSONL records and unique IDs/prompts; verify every artifact hash; resolve every environment reference; jointly audit families and lineages; verify public/private separation; test representative list/search/read calls and replay; and ensure advertised quality counts equal actual per-record status counts. Package public and private exports separately. SFT and preferences remain empty without accepted investigation trajectories or reviewed pairs; RL training exports remain empty when benchmark tasks are evaluation-only. Freeze releases and create a new version for corrections.

## Expanded review decision

The initial purposeful sample exposed material errors in upstream references. Review therefore expanded to every imported task, grouped by pinned repository and distributed across three workers. Each first-pass review binds the exact prompt and upstream answer hashes and records source-supported central claims, contradictions and proposed corrections. This scope is static central-claim review, not an exhaustive human audit or measured task difficulty.

Proposed corrections are passed without the original reviewer’s rationale to a different worker, who checks truth and whether the correction adequately answers the question. The release keeps both originals and proposed corrections. Only supported original references or independently supported corrections enter `private/grading.jsonl`; remaining references stay in the review quarantine. Agent review never sets human approval.

Every cited review span is independently checked for membership in the solver-visible snapshot, exact Git-blob hash and valid line bounds. This mechanical audit validates evidence identity; the separate reviewer judges semantic support. The frozen-release audit independently recounts task IDs and grading records and verifies artifact hashes, environment bindings, source lineage, split boundaries and reported quality counts.

Repository activity is recorded separately in `reports/task-generation-1000/repository-activity.json`. Benchmark commits remain historical pins. A ten-commit sample is only a selection indicator; repositories that do not meet the recent-activity indicator or cannot be verified are labeled explicitly, rather than being represented as recently active.
