# Base versus final factual coverage

This is a post-hoc diagnostic on all 32 selection tasks, comparing saved base-model
(step 0) answers with saved final-model (step 8) answers from the main GRPO run.
No new policy answers or optimizer updates are generated. This is not confirmation-set
validation or a replacement for the original strict endpoint.

The same frozen Qwen397 v7 training-grader pipeline evaluates both arms with the same
atomic-claims-v4 rubrics and positive factual-coverage reward. Cases are deterministically
shuffled with seed 42 to mix arms. Candidate answers are unchanged. Invalid references
are omitted from grader input and never treated as trusted evidence; original submissions
and strict grades remain archived. Valid reference source and original extractor source
catalogs are preserved where original strict grading ran. Source hashes are verified.
The factory's training split flag is used solely to request its factual-reward judgment;
these development tasks are never added to optimizer inputs.

There are 64 answers total: 60 nonempty answers receive live grading; four missing answers
retain deterministic factual reward zero. Grader errors and semantic uncertainty remain
missing. The primary comparison uses only tasks with resolved scores in both arms. We
also report each arm's scoring coverage, strict passes, full-credit count and positive-credit
count. No case is omitted because its score worsens, and valid unfavorable judgments are
not retried. The original bounded mechanical repair remains enabled.

The uncertainty interval is a seeded 10,000-resample paired task bootstrap of the mean
final-minus-base reward difference. It does not incorporate repeated policy-generation
or judge variability, and this is a small, post-hoc selection-set diagnostic. A positive
point estimate alone is not proof of generalization.

Artifacts:

- [Frozen fixture and original strict evaluations](fixture.json)
- [Submission](submission.json)
- [Launch and reservation cap](launch.json)
- [Live/final grading output](results.json)
- [Summary](summary.json)
- [Paired task table](paired-results.csv)


## Completed result

All 32 pairs resolved. Base mean factual coverage was **0.557292** and final mean was
**0.437500**, a difference of **-0.119792** (-11.98 percentage points). Two tasks improved,
seven worsened, and 23 tied. The paired task-bootstrap 95% interval is **[-0.255208,
0.010417]**, which includes zero. This does not demonstrate improvement; the point estimate
is worse, but the small diagnostic does not conclusively establish regression either.

Full factual credit fell from 17/32 to 12/32. Positive factual credit fell from 20/32 to
17/32. Original strict passes remain 6/32 versus 7/32, with three original final strict
grades unresolved. Factual grades resolved for all cases independently of those strict
audits. Strict and factual results measure different properties, so the small increase
in strict passes does not establish broader factual improvement.

No more training was launched. This evidence favors examining the seven worsened pairs
and comparing saved intermediate checkpoints under a fixed diagnostic before spending on
an unbounded continuation. The held-out confirmation set remains unused by this test.

Model reservations were $14.701848, plus the bounded controller reservation; actual billing
is not known. [Raw judgment logs](../../artifacts/grader-paraphrase-validation-results/artifacts/experiments/grpo-base-final-factual-comparison-v1/).
