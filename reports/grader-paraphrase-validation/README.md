# Qwen grader fix, validation, and baseline refresh

The primary training-reward validation checks passed after the fixes. Baseline rescoring is complete, but the strict confirmation scoring-coverage gate remains below its required threshold.
All earlier attempts and mismatches remain archived; no GRPO updates have run.

## What changed

The Qwen judge remains `Qwen/Qwen3.5-397B-A17B`. Training uses `all-claims-v7` with an
independent factual-coverage pass, version `independent-factual-coverage-v3`.
The numeric reward formula remains `positive-coverage-v4`: complete required facts earn
1, partially covered facts 0.5, missing/contradicted facts 0, followed by weighted averaging.
Citations and unrelated extra assertions cannot affect this separate judgment. They remain
part of the strict audit. An unresolved strict audit is recorded separately; independently
verified training coverage can still be usable. Genuine uncertainty in factual coverage
remains unresolved, never silently zero.

The separate reward pass receives pinned reference source plus verified source gathered
by the strict evidence reader or valid candidate citations, without seeing strict verdicts
or citation judgments. The source is useful evidence; the citation itself earns no credit.
Every source range is checked against the pinned file hash. The output requires exact
candidate span IDs and source evidence IDs before software computes the reward.

The prompt also accepts semantic paraphrases and prevents the broader question from adding
requirements to an individual scored claim. The PyBryt reference explicitly describes the
context manager's collection, wrapping, and reference-checking operations. Existing prose
references supplied another 555 valid source ranges across 291 tasks; 43 invalid/unresolvable
citations were recorded rather than trusted. Questions, policy inputs, splits and original
reference prose are unchanged. The final data release is
[repo-qa-atomic-claims-v4](../../data/releases/repo-qa-atomic-claims-v4/manifest.json).

The judge context budget increased from 32,768 to 65,536 tokens to accommodate verified
source; policy rollout budgets are unchanged. All retries remain bounded to one mechanical
repair, never a retry just because a valid score is unfavorable. Launch cost estimates now
include the third training judge call and its possible repair. Strict evaluation uses its
existing two-stage audit; the extra reward call applies only to training routing.

## Validation design

The same 92 answer cases were used throughout:

- 25 training tasks from 25 repository families, with all reference facts, one reference
  fact, and an abstention: 75 controlled answers.
- Seven previously examined regression cases, including correct/partial/mixed/wrong
  uncited answers and the PyBryt response.
- Two extra grades of each of five fixed cases: ten repeatability attempts.

These are source-grounded rubric-coverage controls, not independently human-authored gold
answers or held-out benchmark accuracy. Task selection is deterministic and requires multiple
usable claims; it is not a representative random sample of the whole release. No new
confirmation answers were used to tune the training reward. The full response texts and
expected ranges were frozen in the fixture. All attempts, including failures, remain in
reports and raw logs. Repeats do not select a best score.

[Engineering checks](predeclared-checks.json) were specified before inspecting the first
run's results: at least 95% usable rewards, at least 24/25 full-fact answers at full credit,
25/25 zero-credit abstentions, at least 24/25 positive one-fact answers, at least 20/25 strict
full-over-partial rankings, and all known regressions within their frozen ranges. Repeat
spreads are inspected separately. These are practical engineering checks, not confidence
bounds on general accuracy.

## Preserved iteration results

| Run | Usable rewards | Full-fact controls at 1 | Positive one-fact controls | Zero abstentions | Full > one fact | Regressions within range |
|---|---:|---:|---:|---:|---:|---:|
| Prompt clarification | 85/92 | 11/25 | 21/25 | 24/25 | 15/25 | 5/7 |
| Claim-scope + source enrichment | 89/92 | 20/25 | 24/25 | 25/25 | 15/25 | 6/7 |
| Independent reward pass | 92/92 | 25/25 | 25/25 | 25/25 | 25/25 | 6/7 |

The first runs showed that a combined strict/citation audit sometimes demanded unscored
extra detail or withheld factual credit from uncited answers, despite prompt instructions.
Some references also lacked necessary source context. The independent pass eliminated
those failures on all 25 training-task groups. Its remaining PyBryt failure exposed an
implementation omission: verified helper code read during the strict audit was not reaching
the reward pass. The final run fixes that evidence handoff.

[Run 1](summary-v1.json) · [Run 2](summary-v2.json) · [Run 3](summary-v3.json)

## Completed baseline and comparability

The archived 149 episodes contain 32 selection tasks, 85 confirmation tasks, and a second
32-task selection pass. Generation compatibility was checked against the original frozen
bundle: model, prompts, tool implementations, shared rollout loop, episode methods, public
task inputs and limits are unchanged. Only grading and reference evidence need refreshing.
[Compatibility checks](baseline-reuse-checks.json).

The completed rescore preserves development/evaluation routing. Of 149 episodes, 24 receive
a deterministic strict zero under the unchanged missing/invalid submission gate; 125 need
live strict grading. Cohorts and the repeated selection pass are reported separately below.
This is a baseline **rescore**, not a new measurement of rollout throughput, policy generation
variance, or runtime. Its evaluation-mode `training_feedback` is not a separately measured
training-mode reward and must not be presented as one. No GRPO optimizer updates are run.

## Files and logs

- [Revised release manifest](../../data/releases/repo-qa-atomic-claims-v4/manifest.json)
- [Evidence enrichment summary](evidence-enrichment-summary.json)
- [Final affected-case validation config](../../configs/experiments/paraphrase-validation/v6/validation.json)
- [Frozen cases and expectations](fixture-v2.json)
- [Initial tests](tests.log), [final full suite: 509 passed](final-full-tests.log), [focused integration tests](final-focused-tests.log)
- [Raw validation and baseline run directories](../../artifacts/grader-paraphrase-validation-results/artifacts/experiments)

## Narrow follow-ups after the broad run

The source-transfer fix preserved all 25/25 full > partial > abstention rankings, but
PyBryt's earlier clarification had introduced an incidental caller-name requirement.
The final release corrects just that one task, retaining collection, footprint wrapping,
and checking against references as three separate facts. A targeted run then gave the
saved partial answer exactly 1/3 on all three repetitions. Its new uncited-helper control
exposed a remaining retrieval gap: no helper source was requested by the strict extractor.

The final implementation therefore reads definitions of explicitly named functions in
already-known, pinned source files. It considers snake_case identifiers and identifiers
quoted as code, avoiding ordinary prose words such as "name" or "values". It does not
execute candidate text or search outside the pinned snapshot. Existing source spans are
reused. The prompt/source impact audit identifies affected earlier controls; those plus
all regressions and the PyBryt positive/negative controls are rerun. Unchanged cases are
retained from the broad run, not cherry-picked from repeated attempts. Final validation
is a broad run plus explicit affected-case verification, not one newly run 92-case batch.

[Source/prompt impact audit](prompt-impact.json) · [Final release integrity](final-release-validation.json)

## Final validation decision

The final, explicitly assembled 92-case suite has 92 usable rewards, 25/25 full-fact
controls at 1, 25/25 positive one-fact controls, 25/25 abstentions at 0, and 25/25
full > partial > abstention orderings. All seven original regressions fall within their
frozen expected ranges. All five repeated cases have zero observed reward spread over
three grades each. Twelve affected cases were replaced by mandatory rerun results;
the other eighty retain their unchanged broad-run results. This is not best-of selection.

Of five additional targeted controls, four exactly match expectations. The terse uncited
PyBryt paraphrase receives **1/6 rather than the expected 1/3**. It now earns positive
credit, and wrong/abstaining/full-answer ordering is intact, but some semantic
under-crediting remains. This mismatch is retained; expectations were not widened after
seeing the result. The primary gates passed, so the decision is to freeze the grader,
refresh the baseline, and consider a small GRPO pilot rather than continue open-ended
prompt tuning. This is not a claim of perfect grading or human-calibrated accuracy.

[Final validation summary](summary-final.json) · [Final case table](case-results-final.csv)
· [All final and retained judgments](live-results-final.json) · [Baseline decision](baseline-decision.json)


## Final baseline results and remaining blocker

| Cohort | Attempts | Resolved strict grades | Strict passes | Unresolved/error |
|---|---:|---:|---:|---:|
| Selection | 32 | 32 | 7 | 0 |
| Confirmation | 85 | 78 | 10 | 7 |
| Selection repeat | 32 | 32 | 3 | 0 |
| All, including repeat | 149 | 142 | 20 | 7 |

The primary selection + confirmation total is 17 strict passes among 110 resolved grades
from 117 attempts. The repeated selection pass is not an independent confirmation set.
All 149 archived episodes were processed: 125 live grades and 24 deterministic strict zeros
for missing/invalid submissions. Of the 142 resolved results, 122 are strict failures.
There were 138 completed answers and 11 budget-exhausted trajectories in the original runs.

Twelve requests were unconditionally rescored after a catalog audit found source files
available to the original grader but omitted from the initial rescore input. Selection of
these corrections used archived source catalogs, not score outcomes. All twelve resolved;
the final results retain every correction, including lower scores. Original outputs remain
archived. See [catalog audit](baseline-source-catalog-audit.json),
[corrected results](baseline-context-results.json), and [assembled results](baseline-final-results.json).

Confirmation scoring coverage is **78/85 = 91.8%**, below the required 95% (at least 81/85).
The aggregate 142/149 coverage does not override this per-cohort gate. The answer-completion
gate passes. No optimizer updates or new policy rollouts were run. These score changes
measure changed grading/references, not policy improvement.

The seven remaining confirmation cases comprise five semantic adjudications and two invalid
judge outputs after the bounded repair. The semantic cases concern an unsupported quadratic
complexity premise, child-process/tracing evidence, ambiguous “Missing sentinel” wording,
a performance-bottleneck assertion, and the connection between typed attributes and validation.
The mechanical failures are an invalid coverage enum and a supported assertion missing verified
evidence keys. They remain unresolved, not zero. Full details: [unresolved cases](baseline-unresolved-cases.json).
A source-backed adjudication of these narrowly identified cases is the remaining strict-evaluation
work before clearing the full GRPO readiness gate; confirmation results were not used to tune
the training reward. The extra terse PyBryt control also retains its documented under-credit.

- [Baseline summary](baseline-summary.json)
- [Every baseline episode and its result](baseline-episode-results.csv)
- [Main baseline raw logs](../../artifacts/grader-paraphrase-validation-results/artifacts/experiments/qwen-atomic-paraphrase-baseline-rescore-v3/)
- [Catalog correction raw logs](../../artifacts/grader-paraphrase-validation-results/artifacts/experiments/qwen-atomic-baseline-context-restore-v1/)
- [Final targeted reward-validation raw logs](../../artifacts/grader-paraphrase-validation-results/artifacts/experiments/qwen-paraphrase-validation-v6/)

## Budget

The six reward-validation runs reserved $75.300027 for model calls. Baseline rescoring and
its source-catalog correction reserved $20.362431 for model calls. Including controller
reservations, this work increased the shared ledger by $100.220922, from $157.054915 to
$257.275837 of the $300 cap. These are conservative reservations, not confirmed invoiced cost.
The final controller and all model runs completed, and raw logs and the authoritative shared
ledger were downloaded. [Shared budget ledger](../../artifacts/project-budget/01-baseline.json).
