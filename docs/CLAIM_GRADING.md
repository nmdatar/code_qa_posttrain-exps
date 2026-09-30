# Collection claim grading

New collection runs use `source-claims-v2`. The judge returns judgments, not a scalar reward. Software computes:

`score = sum(weight × credit) / sum(weight)`

| Verdict | Credit |
|---|---:|
| supported | 1 |
| partial | 0.5 |
| missing | 0 |
| contradicted | 0 |
| unresolved | Entire episode remains unresolved, with null reward |

A wrong candidate is a resolved failure, not an unresolved grading result. Unresolved is reserved for insufficient/conflicting evidence or invalid model output. Missing or invalid submissions/citations still receive zero through the existing deterministic checks.

## Rubric construction and scope

Before collecting answers, the collection loader constructs a private rubric from **every** `reviewed_claims` entry in the pinned release. Claims have stable IDs, equal weights, bindings to verified source excerpts, and a rubric hash. Qualified references retain their qualification. Missing claims/evidence bindings stop input loading; there is no prose-only scoring fallback. Rubrics and private references are not sent to the solver.

The current release has 975 eligible tasks and 1,141 recorded claims: 853 tasks have one claim, 84 have two, 32 have three, and six have four. These were source-reviewed automatically, not independently human-calibrated. Most rubrics are broad. A complete assessment of these records is **not** proof that every detail in the prose reference has been covered. `complete_prose_coverage_verified` remains false. Expanding the rubrics requires a new reviewed release; do not derive expected claims from candidate answers or change them after viewing scores.

This metric measures reference-claim coverage. It does not systematically penalize every extraneous false assertion unrelated to any recorded claim, and it is not the accepted/partial/failed efficiency-reward verifier.

## Response validation

The judge must return exactly one judgment for every claim ID. Missing, duplicate, unknown IDs, unsupported verdicts, duplicate JSON keys, and model-provided overall scores are rejected. Supported, partial, and contradicted verdicts require nonempty `answer_span_ids` and known source evidence IDs. Software splits the candidate into numbered passages at sentence/newline boundaries, retaining exact character offsets. The judge selects passage IDs instead of copying text. Unknown/duplicate IDs and missing-claim judgments with nonempty passage lists are rejected. Accepted records include exact `answer_quotes` and offsets reconstructed from the candidate. No punctuation normalization, ellipsis expansion, or model-written quote repair is performed. The code verifies passage provenance, not semantic entailment; that remains the model's responsibility.

The prompt separates the candidate from reference claims and evidence. The old prose reference is not included in the grading prompt. Embedded instructions are explicitly treated as data. This improves auditability but does not establish prompt-injection resistance.

Raw generations are saved before output validation. Accepted artifacts include all per-claim verdicts, reasons, weights, credit, rubric identity, and the code-calculated score. Episode verification diagnostics retain the claim records.

Resolved run configs pin `environment.grading_version: source-claims-v2`; reward identities include the new prompt and judge config. Legacy scalar checkpoints cannot silently resume under the new semantics. Use a new run/fork and reevaluate controls. The legacy scalar parser remains only for inspecting historical recovery artifacts.

## Validation result

Unit/integration tests cover weighted aggregation, every-claim completeness, invalid weights/evidence, invented quotes, scalar injection rejection, raw-output retention, and concurrent grading. All 975 admitted task rubrics loaded successfully.

The 15-case live Nemotron rerun produced differentiated scores, but still overcredited wrong answers and sometimes emitted verdicts inconsistent with its own explanations. See [diagnostic report](../reports/grader-diagnostic-claims-v2.md). The aggregation implementation is verified; Nemotron remains unsuitable for trusted training rewards. Prepared experiment configs now select Qwen397, which rejected the six wrong/injection cases, with three formatting failures under the old quotation protocol. The new passage-ID protocol removes that copying requirement and is covered by offline regression tests. The earlier diagnostic is sufficient for exploratory screening; a further calibration campaign is not a prerequisite. The new protocol has not yet had a live model run. See [Qwen397 validation](../reports/grader-diagnostic-qwen397-v3.md).
