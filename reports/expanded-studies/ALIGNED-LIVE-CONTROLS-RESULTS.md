# Aligned reward live controls: failed prerequisite

All 56 fixed controls completed, but only **48/56 (85.7%)** were eligible; the frozen requirement was at least 54/56. All eight unsupported-extra controls were excluded. Every task therefore lacked an eligible unsupported-extra variant. **The aligned REINFORCE arm remains blocked.** The direct three-seed arm is complete; the SFT arm is separately blocked and this gate does not validate its data or resolve that block. No model training occurred in this campaign. Monitoring is paused and no campaigns remain active.

| Variant | Eligible | Mean reward | Reward range | Mean factual coverage |
|---|---:|---:|---:|---:|
| Correct | 8/8 | 1.00000 | 1–1 | 1.000 |
| Partial | 8/8 | 0.35625 | 0–0.5 | 0.500 |
| Wrong | 8/8 | 0.00000 | 0–0 | 0.000 |
| Mixed contradiction | 8/8 | 0.03750 | 0–0.15 | 0.625 |
| Unsupported extra | 0/8 | unavailable | unavailable | 1.000 |
| Uncited | 8/8 | 0.76250 | 0.7–0.9 | 1.000 |
| Invalid citation | 8/8 | 0.78125 | 0.65–0.85 | 1.000 |

The official gate reports the coverage failure and four missing-variant failures. Its implementation skips further per-task checks when a variant has no eligible samples. Consequently, its failure list does not enumerate the additional Sphinx partial-answer failure diagnosed below. This reporting limitation does not change the failed outcome; no gate logic or thresholds were changed after results.

## Unsupported assertions: judge behavior meets an incompatible reward mapping

All unsupported controls appended the same deliberately unverified claim: measured benchmarks show this implementation is exactly 3.7 times faster. The pinned source contains no such benchmark evidence. Case IDs are `8d120bf1-unsupported_extra-r1/r2`, `73d1da82-unsupported_extra-r1/r2`, `92caf412-unsupported_extra-r1/r2`, and `eb1ef45d-unsupported_extra-r1/r2`.

The judge marked these additions as material errors and false execution claims, with empty evidence keys on the negative finding. Some verdicts were `insufficient`; others were `contradicted`. For example, `8d120bf1-unsupported_extra-r2` correctly describes missing benchmark evidence in its explanation and labels the claim insufficient, yet also sets material_error and false_execution_claim. `73d1da82-unsupported_extra-r1` labels the unsupported benchmark contradicted without a source contradiction witness. The claim explicitly asserts measured results, so treating fabricated empirical evidence as a false-execution claim may be reasonable; it does not establish a source-backed contradiction or that the candidate claimed to run the benchmark itself.

The production adapter explicitly includes every material-error finding in its negative-evidence guard. Empty evidence keys then deterministically cause `negative_finding_without_source_evidence`, before it can compute the intended smaller unsupported-claim penalty. The adapter also counts material_error as contradiction and excludes such findings from the unsupported count. This is a **mismatch between the frozen desired unsupported penalty and the existing conservative reward contract**, exposed by real judge outputs. The guard operates as implemented: it intentionally rejects source-unbound material flags. These observations do not establish an implementation bug; they establish that this contract and judge behavior fail the desired live gate. Simply removing the evidence guard would convert ambiguous unsupported findings into severe contradiction penalties and would not repair the contract.

## Partial Sphinx answers: a demonstrated semantic false penalty

`92caf412-partial-r1` and `92caf412-partial-r2` state only that the matching built-in LaTeX theme takes precedence. The pinned `sphinx/builders/latex/theming.py` lines 115–123 select `self.themes[name]` before looking for a user theme, then call `theme.update(self.config)`. The partial answer states the precedence fact and omits the update; it does not deny the update.

Both independent coverage passes assign the expected **0.5**. Both full semantic assessments nevertheless mark required claim c2 as contradicted/material_error, even though the extracted additional assertion is supported. The explanations themselves acknowledge absence rather than denial; r1 even concludes that material_error should be false, contradicting its structured fields. The adapter's required-claim fallback then sets contradicted=1 and deducts 0.75, clipping both rewards to **0**. This is principally a **judge semantic/structured-output error**, amplified by a reward mapping that accepts any source-linked required-claim material-error flag as a contradiction. Source-key presence establishes provenance, not that the negative judgment is correct. Both partial rewards equal the wrong-answer reward, violating the intended strict ordering.

## What worked, and what this does not establish

All eight complete correct answers had reward 1, coverage 1 and zero penalties; all eight wrong answers had reward 0. Citation removal and invalid hashes produced deductions. No nonfinite values or infrastructure execution failure were observed: the controller exited 0, all cases completed, and exclusions were semantic eligibility decisions. The 341 archived files were independently rehashed successfully with zero JSON parse errors.

Four training tasks, seven variants and two replicates are a directional prerequisite, not a general accuracy estimate. Citation deductions vary with claim extraction granularity, as illustrated by different invalid-citation replicate rewards; this should be measured before treating the reward as a stable scalar objective. Neither passing subgroups nor successful infrastructure justify bypassing the failed prerequisite. Confirmation and selection were not used.

## Cost and provenance

The authoritative ledger reserves **$15.971949**: **$14.452461** for judge calls and **$1.519488** for the controller, within the $100 private allocation and $79.966080 conservative full-campaign bound. These are reserved amounts, not invoices; actual billing remains unavailable. The campaign made zero policy-generation or optimizer calls and produced no training data.

- Run sandbox: `sb-yK6BrRCUZsd6QGLvDATowg`.
- Fixture hash: `3013c87268b0b155b29fe147e34ba31c888b92f3259768afc6954f0b1fc0b648`.
- Bundle: `2568f97f39ec5e31085ba0da3c2160ceb1c1431ac4ea705caa61cef905931890`.
- Archive: `artifacts/aligned-live-controls-v1-results`.
- Audit: `reports/expanded-studies/aligned-live-controls-archive.json`.
- Companion JSON contains all 56 case IDs, rewards, components, required/additional findings and exclusions.

## Concrete next decisions

1. Preserve this failed campaign. Do not lower its eligibility threshold, relabel its cases, or count exclusions as zero rewards.
2. Define a new versioned reward contract distinguishing absent support, explicit source contradiction, and asserted execution evidence. Decide separately how unverifiable benchmark assertions should be treated; the current material_error boolean cannot represent those distinctions reliably.
3. Require a candidate answer span that asserts the contradicted proposition for a contradiction penalty. An omitted rubric claim should lower coverage, not create a contradiction. Audit both required and extracted findings, including disagreement between structured fields and explanatory rationale.
4. Build regression cases from these failures, then freeze a separate set of previously ungraded training-only controls across additional source tasks. A repaired gate must demonstrate the distinction on those fresh cases before aligned RL. Success on these now-observed fixtures alone would be development validation.
5. Keep a new grader/reward version explicit in any resumed experiment and paired comparison. Resolve the SFT arm's separate admission failure independently. No fix, retuning, new grade, or launch is included in this report.
