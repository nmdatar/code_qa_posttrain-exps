# Strict evaluation and positive training feedback

This page records the early positive-feedback reward design and subsequent
revisions. It is not a universal description of every retained study's reward.
See the [research progression](../experiments/README.md) for later comparisons.

The historical `positive-coverage-v4` / `all-claims-v6` diagnostic matched 4/7
expected calibration scores and blocked further rollout. Its config and report
are preserved in the [historical index](EXPERIMENT_HISTORY.md); shared reward
implementations and regression tests remain. The absence of GRPO updates in that
diagnostic is not a claim that no later GRPO experiments ran.

## Early-training reward contract

Training reward is the weighted mean of factual reference coverage: supported
complete = 1, supported partial = 0.5, no supported relevant content = 0.
There are **no deductions** for contradictions, unsupported assertions, citation
quality, or false execution claims in this early-training version. These defects
remain recorded and still affect strict evaluation. Correct content can receive
credit even in an otherwise mistaken answer or qualified abstention; the mode
label alone does not erase factual credit. Unrelated true filler earns nothing.

Grader v6 explicitly permits supported partial coverage within a bundled claim
without predefined `separable_subparts`. It must explain the correct portion and
missing/wrong portions, verify against pinned source, and assess incorrect
extracted assertions separately. Missing citations do not block factual coverage.
Extraction must inventory the candidate's assertions, not assertions from the
question. These instructions require live calibration; tests cannot prove judge
compliance.

Strict scoring rules and development grading instructions are unchanged. Both
strict and training scores plus diagnostic counts remain logged separately. The
new grading instructions apply only to train-split v6 audits. Historical v3 reward
arithmetic and grader versions remain available for audit replay; old reward
configs cannot launch as the new version.

Unresolved grading remains null and excludes its group. Empty submissions remain
zero. Invalid citation metadata still takes the existing deterministic zero path
before semantic grading; this is distinct from missing citations or failed
citation entailment. Equal-reward groups contribute no GRPO update.

This deliberately permissive reward can give high credit to an answer containing
correct required facts plus false extras. That is the requested early-training
tradeoff; strict evaluation remains the quality control. No automatic penalty
schedule or later transition has been introduced.

## Grader reliability, versions 4–5

- Overlapping pinned evidence ranges share one numbered source block. Every
  requested line and original evidence key is preserved. Conflicting overlapping
  source fails closed; no source is silently truncated.
- Reference context expands to 120 lines each side and the evidence-byte ceiling
  becomes 96,000. The provider's actual tokenizer/context check still enforces its
  configured context window. Evidence beyond either limit remains unresolved.
- New configs allow 8,192 judge output tokens and at most one mechanical repair
  per stage (four total calls worst case). Every attempt is logged and reserved.
- Repairs target truncation, schema errors, omitted IDs, nonexistent evidence keys,
  or missing support keys. A valid zero score, uncertainty, or reference dispute
  is never retried for a better result. Genuine ambiguity still requires review.
- The new instructions separate reference validity from candidate mistakes and
  explain shared source blocks. They do not automatically adjudicate references.

The strict scoring criterion is unchanged, but this is a new grader version;
its scores require fresh controls and must not be pooled with the old baseline.
Any subsequent formula/prompt change needs another version and frozen validation.

## Validation and launch gate

The [archived validation script](https://github.com/nmdatar/action-interview/blob/2b70198df1db7b3028448d8f05772e55c52ea6fc/scripts/validate_reward_shaping.py) produced
source-grounded synthetic ranking fixtures and an offline projection of selection
judgments. It has been retired from main; the reward/grader regression tests
remain. Those checks verify the arithmetic contract, not live judge calibration.
Confirmation answers are excluded from reward design.

The live diagnostic uses four family-stratified training questions selected before
seeing rewards, four fresh attempts each, temperature 1, eight rollout workers,
four judge workers, and no optimizer allocation. `reward-signal.json` reports
strict versus training reward variance, contributing groups, and whole-group
exclusions. A separate 32-question selection run uses temperature 0 and strict
scores. Neither diagnostic runs GRPO.

Require resolved coverage of at least 95%, answer completion of at least 90%,
validated accounting, and usable group variation before considering GRPO. Passing
these small diagnostics does not establish calibration, generalization, or a
throughput-selected concurrency. Inspect reward rankings on actual answers before
approving a full training campaign. The existing full baseline and throughput gates
still apply; no prepared Experiment 2 config is silently changed or launched.

## Rejected prototype

`v1` allowed negative rewards. The first live four-question diagnostic showed
that all-zero strict groups gained variance mainly because incomplete/uncited
answers stayed at zero while false answers became negative. That could reward
avoidance. The retained v1 logs are diagnostic history, not a training recommendation.
Version 2 clips at zero and preserves partial factual credit; v1 configs are no
longer accepted by the current launcher.

Version 5 also validates extraction ranges against catalog line counts and hashes
before reading source. An out-of-bounds request enters the same single mechanical
repair, with the actual line bound in the error. It never silently clamps a request
or discards requested source. The earlier v4 failure is retained.

Version 3 closes the training-only missing-citation bypass. Earlier nonnegative
v2 diagnostics often never reached semantic assessment because answers omitted
citations. This change measures factual support before applying citation penalties;
it does not award a completion/format bonus or loosen strict evaluation.
