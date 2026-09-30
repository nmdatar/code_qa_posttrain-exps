# Repository Q&A collection: completed task release

**992 runnable task records; 987 reviewed grading references; 5 unresolved references excluded from grading.**

The versioned release is [repo-qa-1000-v1](../data/releases/repo-qa-1000-v1/DATASET_CARD.md). It contains 980 attributed benchmark imports and the original 12 locally generated executable tasks, spanning 41 repository families and 44 immutable Modal environments. The imported tasks are source-reading tasks; application execution is not implied.

## What is included

- [Public tasks](../data/releases/repo-qa-1000-v1/public/tasks.jsonl): original user prompts, starting system prompts, repository URLs and full commit pins, environment IDs, allowed tools, budgets and development split.
- [Public environments](../data/releases/repo-qa-1000-v1/public/environments.jsonl): immutable Modal image references, snapshot/recipe hashes, tool versions and source capabilities.
- [Private grading](../data/releases/repo-qa-1000-v1/private/grading.jsonl): 987 selected references or reviewed local claim records. Keep this file outside solver access.
- [Private quality records](../data/releases/repo-qa-1000-v1/private/quality.jsonl), original references, exact raw source records, source/behavior assertions, independent reviews, correction reviews, recipes, snapshot inventories, runtime evidence and source notices.
- [Usage guide](TASK_COLLECTION_1000_USAGE.md), [progress and timing](TASK_COLLECTION_PROGRESS.md), and [optimization log](TASK_GENERATION_OPTIMIZATIONS.md).

## Quality results

All 980 imported references received a central-claim review against pinned source. 569 originals were supported; 411 required correction. A different reviewer checked every proposed correction: 406 were supported and 5 remained incomplete. Combined with the 12 previously reviewed local tasks, this yields 987 grading references.

These are agent-reviewed development references, not human-approved gold. Review focuses on central claims and question coverage; it is not an exhaustive human sentence-by-sentence audit. Difficulty has not been measured over this collection. Citation parser diagnostics for original upstream answers are retained separately from correction/evidence checks.

Original benchmark splits and provenance are preserved. All records remain development-only; no imported benchmark enters training or a new held-out test set. SFT, RL-training and preference exports are empty. This task collection does not claim 992 teacher trajectories or 992 solved investigations.

## Verification evidence

- All 41 imported source environments passed readiness, per-attempt isolation checks, source-hash attestation, and nonempty list/search/read calls with exact replay. All original local environments retain their earlier executable checks.
- All 980 imported prompts and original answers match the exact immutable upstream file and row; commits and family assignments match their manifests.
- Every recorded review evidence span binds a real solver-visible file and exact Git-blob hash. The 411 corrected answers also passed recognized citation checks, including seven explicitly documented shorthand/context resolutions.
- Exact normalized-prompt uniqueness and a separate lexical near-duplicate screen passed. The latter is a heuristic, not proof that all questions are semantically unrelated.
- [Frozen-release audit](../reports/task-generation-1000/release-audit.json) independently recounts 992 tasks, 987 grading records, 44 environments and 41 families; all 305 artifact hashes and task/source/environment bindings pass.
- [Reproducibility audit](../reports/task-generation-1000/reproducibility-audit.json): two fresh exports produced identical manifests and every artifact hash.
- 172 implementation tests passed. A separate agent inspected the export/audit code and actual inputs and found no concrete release blockers.

## Quarantined grading references

These five records remain available for curation but do not appear in the private grading export. Their repository tools can run; the incomplete references must not be used to score answers.

| Task | Repository | Remaining issue |
|---|---|---|
| `import-2f3c8028c5a3f4f11f316e5c` | psf/requests | Auth stripping explanation is accurate, but original question explicitly asks coordination with get_redirect_target; correction entirely omits its Location extraction and redirect-loop role. Add that link before acceptance. |
| `import-8993b4bacfe092b77bd92be9` | matplotlib/matplotlib | The corrected facts are accurate, but the actual question asks where matplotlib_inline bootstraps and how it chooses by IPython version. This external behavior remains unavailable, so registration-only explanation cannot substantively complete that question. |
| `import-de308ef09c161ab9ef8b94b3` | streamlink/streamlink | Question explicitly asks structure and data types needed to avoid parsing errors; answer only names converters, omitting that requestId is string-like and timestamp must support float conversion. |
| `import-f148ffd42684f22f2be2db0d` | ethereum/web3.py | Account assignment and RPC documentation are accurate, but the question specifically asks typed-data architecture with domain-separator validation. The corrected answer explicitly leaves that external behavior unverified and does not adequately answer it. |
| `import-ff6619e1f10412384936a947` | streamlink/streamlink | Question requests a prefetching mechanism integrated with segment data hierarchy; answer substitutes a playlist-test extension and leaves production fetching, optional metadata and ordinary segment compatibility design unanswered. |

## Coverage and next collection decisions

The sources are SWE-QA (720 questions, Apache-2.0) and SWE-QA-Pro Bench (260 questions, MIT). Their notices, revisions and raw artifacts are included. These historical benchmark pins should not be described as questions newly generated from current repository heads.

Current repository activity is a separate advisory record: 19 of 38 imported families met the latest-ten-commits indicator (at least five sampled commits within 30 days), 8 did not, and 11 could not be verified. Unknown activity is not treated as recent activity. Use this metadata when selecting families for a new generation round.

For new training collection, choose new preassigned repository families, generate fresh questions and private references, reject unmeasured performance premises, and collect blind teacher trajectories. Preserve the current development partition. The present task-generation objective is complete; the full training-data roadmap remains broader.

The progress log records wall-clock elapsed time, including the usage-limit interruption. Monetary cost and per-model token billing were not measured; no cost claim is made.
