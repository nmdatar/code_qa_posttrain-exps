# Positive reward: live validation

**Calibration failed: 4/7 expected scores matched, 6/7 cases resolved.** The
controller exited normally and correctly blocked the four-attempt diagnostic.
No new policy rollouts, optimizer updates, or checkpoints were produced.

| Case | Expected training reward | Actual | Result |
|---|---:|---:|---|
| Saved correct answer | 1 | 0.5 | Under-scored |
| Same answer without citations | 1 | 0.5 | Under-scored |
| Correct portion only, uncited | 0.5 | 0.5 | Passed |
| Correct portion plus contradiction, uncited | 0.5 | 0.5 | Passed |
| Wholly wrong, uncited | 0 | 0 | Passed |
| Saved partly correct answer | 0.5 | 0.5 | Passed |
| Saved `finds_no_match` answer | 0 | Unresolved | Invalid judgment |

The first, sixth, and seventh cases are unchanged saved answers. The others are
explicitly controlled variants. Expectations were frozen before paid grading,
based on pinned source inspection. Selection examples were routed through the
training grader only for calibration; this does not replace baseline evaluation,
add examples to training, or use confirmation answers. Four variants share the
same source example, so these seven cases are not independent task measurements.

## Findings

The desired partial-credit behavior works on the controlled cases: the judge
recognizes a correct part of a bundled claim even with no subparts or citations.
The mixed answer retains 0.5 despite a material contradiction. Wrong content alone
receives zero. All resolved cases have strict score zero in this training audit;
strict development evaluation was not rerun.

However, the judge misreads the saved correct answer's phrase `level_names is
None` as an assertion that `level_names` is a function parameter. The answer never
says that; the pinned source checks the instance property `self.level_names`.
The judge also lets that supposed extra error reduce required factual coverage,
although its own findings verify both parts of the required reference claim.
Removing arithmetic penalties alone does not prevent penalties hidden inside the
judge's coverage classification.

For `finds_no_match`, extraction no longer copies assertions from the question,
but invents a metaclaim: “The candidate answer is 'finds_no_match'.” Assessment
then labels that metaclaim supported without source evidence. The verifier rejects
it, including after one bounded repair, as `Supported assertion lacks verified
evidence keys`. The unresolved case has not been converted to zero.

Next repair: keep required-claim coverage scoped to that claim, interpret
contextually equivalent paraphrases without inventing unstated API assertions,
and exclude metaclaims about the answer text from factual extraction. Revalidate
these frozen cases before a new rollout diagnostic. Do not launch GRPO yet.

## Cost and logs

Calibration took 153.1 seconds. Judge reservations: **$1.0913**;
including controller reservation: **$1.8510**.
Shared ledger is **$87.3085/$300**. Reservations are conservative
bounds, not actual provider invoices. No W&B run was created because calibration
stopped the campaign before the rollout benchmark's tracker started.

- [Raw results](/Users/ndatar/Documents/ChatGPT/action-interview/artifacts/reward-shaping-v4-results/campaigns/a5e8ad24a330b77aae84762bb546198f564bcdc43b48e0443285572084998324/reward-validation/results.json)
- [Per-case CSV](/Users/ndatar/Documents/ChatGPT/action-interview/reports/reward-shaping-v4/live-validation.csv)
- [Summary JSON](/Users/ndatar/Documents/ChatGPT/action-interview/reports/reward-shaping-v4/live-validation-summary.json)
- [Controller gate status](/Users/ndatar/Documents/ChatGPT/action-interview/artifacts/reward-shaping-v4-results/campaigns/a5e8ad24a330b77aae84762bb546198f564bcdc43b48e0443285572084998324/status.json)
- [Correct-answer judge log](/Users/ndatar/Documents/ChatGPT/action-interview/artifacts/reward-shaping-v4-results/campaigns/a5e8ad24a330b77aae84762bb546198f564bcdc43b48e0443285572084998324/reward-validation/private/saved-correct.assess.judge-raw.json)
- [Mixed-answer judge log](/Users/ndatar/Documents/ChatGPT/action-interview/artifacts/reward-shaping-v4-results/campaigns/a5e8ad24a330b77aae84762bb546198f564bcdc43b48e0443285572084998324/reward-validation/private/mixed-uncited.assess.judge-raw.json)
- [Failed abstention repair](/Users/ndatar/Documents/ChatGPT/action-interview/artifacts/reward-shaping-v4-results/campaigns/a5e8ad24a330b77aae84762bb546198f564bcdc43b48e0443285572084998324/reward-validation/private/saved-no-match.assess.repair-1.judge-raw.json)
- [Frozen fixtures](/Users/ndatar/Documents/ChatGPT/action-interview/artifacts/reward-shaping-v4-bundle/reward-validation.json)
- [Artifact hashes](/Users/ndatar/Documents/ChatGPT/action-interview/reports/reward-shaping-v4/live-checksums.json)

Implementation checks: prior full suite 496 passed; validation preflight 51 passed.
These checks verify software behavior, not live judge correctness.
