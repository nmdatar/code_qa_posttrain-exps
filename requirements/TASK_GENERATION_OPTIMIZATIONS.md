# Task-generation optimization log

## Collection round: 2026-09-28, generation-v2

Objective: increase the throughput of independently checkable task creation. Quantity alone is not acceptance. Preserve the existing repository-family splits; this round adds development tasks, not cleared training examples.

| Change | Rationale and quality guard | Observation |
|---|---|---|
| Three repository authors in parallel | Disjoint files; authors inspect pinned source and create contrasting behavior cases. Independent review happens after authors finish. | Three author workstreams launched. Measured end-to-end results appended below. |
| Multiple mechanisms per existing image | Amortize environment build cost without turning one question into superficial paraphrases. | Target three new mechanisms per repository, nine new tasks total. |
| Exact immutable environment reuse | Require identical repository, commit, recipe, snapshot and environment manifest; validate source hashes and bound isolation/attestation evidence. | `dataset_builder.reuse` copies environment evidence only. Every new task still executes its own assertions. |
| Cheap source and prompt screening before investigation | Reject underspecified setups, answer leakage, trivial paraphrases and unsupported claims before spending solver calls. | Author reports record rationale and screening; difficulty remains unproven until controlled solves. |
| Deterministic contrastive fixtures | Cover positive/negative cases or interventions; no real sleeps or wall-clock thresholds where explicit clocks/events suffice. | Assertions run in fresh Modal sandboxes; failed output prompts source reinspection, not automatic expectation replacement. |
| Separate runnable from accepted | A passing executable fixture checks behavior, not language-answer correctness or training admission. | Draft gold, human review false and training eligibility false remain explicit. |

## Measurements and limits

Record wall time, authored / executable / independently reviewed task counts, first-run failures, image builds avoided, and review corrections. Do not claim a speedup multiplier without a comparable serial baseline. Provider token usage and billed cost are unavailable and must remain unknown rather than zero.

Previous round's question-only control recovered the main behavior of all three tasks. Therefore citations alone are not a difficulty criterion. Favor interacting mechanisms and precise counterfactuals; label new difficulty estimates as hypotheses. Existing Pydantic, TanStack Query and SQLAlchemy families stay in development permanently for this split manifest. A training collection requires additional families and a completed leakage audit.

## Observed results and decisions

- **Nine new runnable tasks; 49 supported reference claims.** Three authors worked by repository, then different agents reviewed their work. The final versions passed 18 fresh-sandbox assertion runs (two per task). No new image builds were required; three existing immutable images were reused.
- **Independent review stays mandatory.** Initial behavior checks all passed, yet review found Query answer hints and two SQLAlchemy prompt/fixture inconsistencies. These were corrected in new versions with preserved draft/review history. Faster generation must not skip this stage.
- **Batch Git blob reads replaced per-file subprocesses.** One local Query snapshot measurement fell from 18.09 seconds to 0.087 seconds for 3,581 files. Exact source hashes match the existing image for all three repositories. This is a single component measurement, not a 207× whole-pipeline claim. [Measurement](../reports/task-generation/snapshot-optimization.json).
- **Observed Query prepare/reuse/two-verification time:** 76.37 seconds before batched reads; 17.35 seconds for the revised prompt version afterward. Same image and assertion code, separate runs; scheduling and Modal provisioning can vary. Treat this as operational evidence, not a controlled performance benchmark.
- **Expected-output corrections are explicit.** Query author revised an initial assumption after source inspection showed retry-delay evaluation precedes the retry-eligibility check. Pydantic author recorded an extra-storage inspection revision. SQLAlchemy recorded a launcher failure before sandbox creation and later fixture corrections. The author reports retain evidence rather than erasing unsuccessful assumptions.
- **Exact prompts replace abbreviated difficulty controls.** The new fresh-context control receives the same final user prompts as the exported tasks. No source or runtime tools are allowed; missing citations are not, by themselves, evidence of repository reasoning difficulty.
- **Implementation checks:** 136 tests pass, including binary Git-object framing, internal symlink handling and rejection of stale/mismatched environment reuse. This verifies collection code, not semantic task difficulty.

## What to optimize next

1. Screen question-only answers before expensive solver trajectory collection; prefer tasks whose concrete outcomes or mechanisms are not reliably reconstructed from general knowledge.
2. Expand into additional preassigned repository families for eventual training coverage. Reusing three development families is efficient now but cannot produce an independent training/test split by itself.
3. Keep generation and review queues separate and bounded. Dispatch new work as a repository author finishes; reuse images and source navigation, not reference judgments.
4. Track rejection/revision rate and coverage gained per repository, not just raw task count. Preserve cheap documented cases for regression coverage, and distinguish them from challenging investigation tasks.

This round covers task generation and reference validation. Nine new blind solver trajectories, human admission and a training release are not claimed complete.

## Difficulty-control outcome: change the next collection strategy

The exact-prompt question-only control recovered all **49 substantive claims across all nine tasks**, including revision-sensitive behavior. This result used one fresh-context agent response per prompt, unknown model version, and no matched tool-enabled comparison; it is advisory rather than a benchmark score. Missing citations and lack of execution were deliberately not counted as answer errors.

**Decision:** retain these as development/regression tasks, and do not spend this round collecting nine expensive tool-assisted trajectories merely to confirm already predictable answers. Passing precise fixtures and independent review establishes a useful correctness reference, not a difficult research problem.

For the next generation round, screen less-guided problem statements and less-familiar repository mechanisms before writing extensive rubrics. Avoid telling the solver which interacting implementation mechanisms to compare when the real user problem would require discovering those mechanisms. Prefer novel source-grounded integration behavior over increasingly elaborate permutations of widely documented APIs. Compare a target learner as well as a strong teacher before deciding that an easy-for-teacher task has no training value.

References: [exact inputs](../reports/task-generation/question-only-inputs.json), [answers](../reports/task-generation/question-only-answers.json), [claim-level assessment](../reports/task-generation/question-only-assessment.json). This finding does not authorize changing the three development families into training families.

## Collection of approximately 1,000 tasks — 2026-09-28

Progress and timing are recorded in [TASK_COLLECTION_PROGRESS.md](TASK_COLLECTION_PROGRESS.md) and the append-only `reports/task-generation-1000/progress.jsonl`. Elapsed time covers acquisition, preparation, environment verification and review since this collection started; it is not a claim that imported benchmark questions were newly authored.

- Imported 980 distinct attributed Q&A records from pinned SWE-QA and SWE-QA-Pro artifacts after excluding unpinned trajectory sources. Existing 12 local tasks bring the runnable target to 992.
- Acquired snapshots with four workers; assigned repository families to development before importing. Benchmark test provenance is preserved and no benchmark records enter training exports.
- Reused one immutable Modal image per repository commit and ran environment checks with three workers. All 41 imported environments passed source-tool readiness, isolation, source-hash attestation and replay canaries in approximately 30 minutes from collection start.
- Recovered three source snapshots using exact Git blobs and Linux extraction. This avoids macOS case collisions and preserves code bytes. Symlink targets and unavailable submodule pins are explicit metadata; these environments grant source reading only.
- Used three independent source-review workers, grouped by pinned repository to reuse navigation while reviewing every task separately. An initial purposeful review found material upstream answer errors, so expanded review to every imported reference instead of treating benchmark provenance as sufficient quality assurance.
- Preserve original answers, reviewer verdicts, proposed corrections and rejection reasons separately. Audit review bindings and every cited file/span against immutable source hashes. Runnable status is never used as evidence of answer correctness.
- Keep tool observations and grading information separate. Source-tool canaries are runtime checks, not solver trajectories or completed SFT examples.

### Findings that change subsequent task generation

The source reviewers repeatedly found questions or references that assume unmeasured performance bottlenecks, infer behavior from comments rather than assertions, overstate caching/atomicity guarantees, or conflate neighboring APIs. These are observed failure classes in the review records, not population error-rate estimates.

1. Reject a performance premise at generation time unless a reproducible measurement supports it. Prefer a source-grounded mechanism question when no execution measurement exists; preserve the original imported prompt and label its faulty premise rather than silently rewriting benchmark history.
2. Build references from executable statements and assertions. Check whether a purported assertion actually tests a condition; a nonempty string in `assert` is not an output-membership check.
3. Require explicit paths through cache invalidation, errors, ownership and state reset before accepting concurrency or atomicity claims. Avoid inferring guarantees from names, comments or intended design.
4. Review every imported reference after systematic errors appeared, and send corrections to a different worker. Reuse repository navigation and pinned images, but preserve independent judgments.
5. Record current repository activity independently of historical task pins. The public metadata check is advisory and retains unknown results; do not change the reference commit to a recent one merely to meet an activity preference.
6. Stage exports and run a separate cross-file audit before publishing a versioned release. A single private grading export selects supported references and excludes unresolved originals, preventing consumers from accidentally scoring against superseded answers.

### Final measured outcome

The collection reached 992 runnable task records (980 benchmark imports plus the original 12 local tasks) at approximately 30 minutes. Final review, correction checks, deterministic packaging and completion documentation finished at approximately 77 minutes of wall-clock elapsed time, including the usage-limit interruption.

All 980 imported references were reviewed. Of 411 proposed corrections, a different reviewer supported 406 and kept five incomplete. The final private grading export contains 987 references/claim records. The release has 44 unique Modal environments and 305 hashed artifacts; two fresh exports were identical. All 172 implementation tests passed.

To shorten the final review phase, reassigned 20 unstarted correction batches when reviewers became available, recording each reassignment and ensuring no author reviewed their own correction. This avoided leaving a finished reviewer idle; no controlled before/after speedup factor is claimed. Immutable artifacts, exact prompts/pins, source hash checks and independent correction review were preserved throughout.
