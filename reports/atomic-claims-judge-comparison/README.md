# Atomic claims and judge comparison

The migration is complete: **987 tasks reviewed; 1,208 → 1,801 claims; 344 tasks
have more claims**. All 71 model-flagged tasks were resolved, with assistant
corrections/decisions on 164 tasks in total. The original release remains intact.
The paired Qwen/Nemotron comparison completed all **28 attempts** with no optimizer updates.
Qwen produced 14/14 resolved grades; Nemotron produced 5/14, with eight output-contract
errors and one unresolved assessment. Keep Qwen with the revised rubric for now; this
pilot does not establish broad grading accuracy or justify replacing it with Nemotron.

## Observed results

Values below are **training rewards**, not strict pass rates. `Error` and `Unresolved`
mean no usable reward was returned; they are not scored as zero.

| Saved/controlled answer | Qwen original | Qwen atomic | Nemotron original | Nemotron atomic |
|---|---:|---:|---:|---:|
| Correct, cited (motivating example) | 0.50 | **1.00** | 1.00 | 1.00 |
| Same correct text, uncited | 1.00 | 1.00 | Error | Error |
| One correct fact, uncited | 0.50 | 0.50 | Error | Error |
| Correct fact plus wrong fact, uncited | 0.50 | 0.50 | Error | Error |
| Wrong answer, uncited | 0.00 | 0.00 | Error | Error |
| Saved PyBryt partial answer | 0.50 | 0.333 | Unresolved | 0.00 |
| Saved abstention | 0.00 | 0.00 | 0.00 | 0.00 |

The original Qwen grade repeats the previously observed mistake: it interprets
`level_names` without `self.` as a materially different variable. Under the split
rubric, Qwen gives both required facts complete coverage and no longer makes that
error; strict score for this cited answer also moves from 0 to 1. The correct uncited
answer earns reward 1 while strict evaluation still applies its citation requirements.
Mixed correct/incorrect answers retain 0.5: the correct fact is rewarded and the wrong
fact earns nothing. All five IndexVariable controls have the expected reward ordering
under Qwen's revised rubric. This is one sample per condition, not a stability estimate.

PyBryt illustrates why a matching total can hide remaining grader problems. Qwen's
0.333 is `(0.5 + 0 + 0.5) / 3`: partial credit for collection, no credit for footprint
wrapping, partial credit for evaluation. The pre-run expectation of 1/3 instead assumed
one fully covered fact and two absent facts. Those totals coincide for different reasons.
The collection explanation may still under-credit a valid paraphrase. Nemotron treats
capitalized `Context` as a specific object/class and gives all three facts zero; its
original-rubric run instead asks for adjudication. The reference means the context
manager, so clearer names and explicit paraphrase examples remain useful follow-up
work. Do not call this a seven-of-seven validated gold test or infer readiness for a
large GRPO run from this sample alone.

Nemotron's eight hard failures persist after one repair each:

- Six extraction failures request lines 1–3072 as one evidence range, violating the
  explicit 600-line limit (partial, mixed and wrong uncited cases, in both rubrics).
- Two assessment failures cite evidence ID `e6` as an answer citation despite the
  answer having no citations (correct-uncited, in both rubrics).

The remaining unresolved attempt is the original-rubric PyBryt case. These are runtime
contract/reliability findings; they do not prove that Nemotron is generally less capable.
A model-specific contract calibration or different reasoning setting would need a
separate matched experiment. Bigger model size alone did not improve this deployed
judge setup. Historical graders and RL configurations were not silently switched.

[Results CSV](comparison-results.csv) · [Every required-claim decision](claim-decisions.csv)
· [Condition counts](comparison-summary.json) · [Complete raw grading logs](../../artifacts/atomic-claims-comparison-results/artifacts/experiments/atomic-claims-judge-comparison-reviewed-v1/comparison)

## Edited data and verification

- [Revised grading data](../../data/releases/repo-qa-atomic-claims-v1/private/grading.jsonl)
- [Manifest](../../data/releases/repo-qa-atomic-claims-v1/manifest.json)
- [Full parent-to-child audit](../../data/releases/repo-qa-atomic-claims-v1/private/atomic-claims-audit.json)
- [Readable claim mapping](claim-migration.csv), [assistant resolutions](manual-resolutions.json)
- [Migration counts](migration-summary.json), [loader validation](release-validation.json)
- [Runnable validation configuration](../../configs/experiments/atomic-claims/validation.json)

Task IDs, questions, split membership, reference-answer text, evidence and attribution
were preserved. The new manifest verifies all 278 artifacts. The standard loader
successfully loads 858 training and 117 development tasks. The other 12 native
structured tasks were also edited, but their runtime capabilities remain outside
this existing source-reading loader. A clean reproduction produces the same manifest.

Qwen proposed splits; Nemotron audited them; the assistant reviewed flagged cases
and revised changed decompositions that dropped conditions, invented facts or gave
duplicate credit. This is model-assisted review, **not human gold**. Single relations,
alternatives and causal conditions remain together when splitting would alter them.
Terse topic labels were not expanded with invented requirements. This migration
therefore does not establish complete coverage of every task's prose reference.

Two descriptions were clarified against pinned source: the plugin test checks emitted
logs rather than empty logs; the HTML cosmology test produces a read-back object that
differs from the original, rather than mutating the original object. Their decisions
are marked `source_clarification` in the audit.

Imported children retain equal fact weights under the existing adapter. Native children
share their parent's original weight. Thus splitting can change an imported task's
relative fact weights and can lower a partial answer's score; it is not a reward bonus.
No contradiction/citation penalty was added to `positive-coverage-v4`, and strict
aggregation remains `all-claims-v6`. Strict *results* can change with the new rubric.

Reproduce without new model calls, using a destination that does not already exist:

```sh
PYTHONPATH=. .venv-eval/bin/python scripts/freeze_atomic_claims.py --target /tmp/atomic-release
```

## Comparison procedure

Seven frozen saved/controlled answers, three source tasks, two rubrics, two judges:
28 grading attempts. Five cases probe the same IndexVariable task (correct cited,
correct uncited, partial, mixed correct/wrong, wrong); two additional saved responses
probe PyBryt and abstention. These are correlated diagnostic cases, not 28 independent
tasks or a held-out accuracy benchmark.

Current judge: `Qwen/Qwen3.5-397B-A17B`. Challenger:
`nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16`.
[Provider catalog](https://tinker-docs.thinkingmachines.ai/tinker/models.json),
[NVIDIA model card](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16).
Both use the same grading prompts, source evidence, answer text, no-thinking renderer,
temperature 0, context/output budgets and bounded repair policy. No policy generation
or optimizer updates. Both models participated in rubric editing, so this is not an
independent held-out model evaluation. Single calls do not establish repeatability.

[Expectations frozen before comparison](expected-rewards.json) are assistant judgments,
not human gold. In particular, PyBryt's terse reference and vague saved answer make
claim-level interpretation debatable; matching its numeric expectation is insufficient
to establish correct reasoning. Read the per-claim explanations as well as total reward.

Explicit user approval authorized transfer to Nemotron/Tinker after the initial
approval block. Decomposition finished successfully, then paused for review as designed;
the reviewed comparison was submitted separately. Both stages stay within the existing
$300 ledger and $100 model-reservation allowance. Reservations are conservative bounds,
not final provider bills.

## Logs and reproducibility

- [Decomposition report](live-progress.json), [launch](launch.json)
- [Comparison report](comparison-progress.json), [launch](comparison-launch.json)
- [Raw decomposition requests/responses](../../artifacts/atomic-claims-comparison-results/artifacts/experiments/atomic-claims-judge-comparison-v1/raw)
- [Full test run: 499 passed](full-tests.log); 27 focused tests passed again after adding the comparison-only route.

Decomposition bundle: `9cc2da326d7aeed6fb2cb3eb791d36f51f6442e62687c0bf9e0c403ffb3a673e`.
Reviewed comparison bundle: `245f3aae1309170bad24c2a72f89875fbea8992e30ac64b56524bb47ed3bfb37`.
Modal volume: `repository-qa-training-state-v2`; both campaign directories and raw run
artifacts are retained there. The revised release manifest hash is
`19c3da245bbd85bc0fb4c2296af70f92f67b7c86d8921909ef9b02ec94a20986`.

## Completion and cost

Both Modal controllers exited successfully. Decomposition took about 13.4 minutes of
model-stage time; the paired comparison took about 6.8 minutes. All 2,223 decomposition
files and 81 comparison files were downloaded, and JSON logs were parsed successfully.

Conservative reservations for this task total **$69.75**: $64.05 for decomposition/audit,
$3.79 for paired grading, and $1.90 for the two controller lifetimes. The shared ledger
now reserves $157.05 of its existing $300 cap. Actual provider billing is not available.
See [completion record](completion.json). No GRPO updates were run.
