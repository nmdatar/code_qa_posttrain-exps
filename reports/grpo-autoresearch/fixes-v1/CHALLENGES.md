# Learning stability: challenges, fixes, and remaining evidence

The training infrastructure performs real adapter updates and saves checkpoints. The challenge is obtaining a reliable, interpretable reward signal and showing generalization. The last completed run's factual reward was 0.599 → 0.615, with an uncertainty interval spanning no change; strict passes were 11 → 6 of 32. These results remain unchanged.

## Changes implemented in this continuation

| Problem | Evidence | Fix | Validation / limit |
|---|---|---|---|
| Underspecified training criteria | 78/858 training tasks contained only “Corrected central explanation and coverage of the original question.” The judge could invent the expected detail. | Expanded the existing source-reviewed reference prose into 316 explicit, separately scored claims in a new training-only release. Conditions, negative conclusions and hypothetical design recommendations remain explicit. | Pinned evidence verified; original reference prose retained; 909 other grading records unchanged. Assistant equivalence review, not new human gold or independent source certification. Some indivisible conditional relations remain together. |
| Valid tool intent rejected for an envelope key | Eight saved invalid actions used `action` instead of `tool`. | Optional `tool_action_policy=action-alias-v1` accepts only an exact two-key action/arguments object naming an already permitted tool. | Unknown/missing tool names, conflicting keys, invalid arguments and unsafe paths remain rejected. This does not repair all malformed JSON. |
| Strict judge inferred an incorrect function name | A saved PyBryt trace explicitly defines execute_notebook, but the judge inferred execute and rejected the correct attribution. | Optional `judge_evidence_policy=definition-context-v1` adds hash-verified Python definition headers/docstrings for mentioned symbols and definitions enclosing supplied evidence. The instruction requires source-based attribution. | Regression checks verify a header outside ordinary context padding reaches the strict judge, including when the candidate uses the wrong name. It never changes the candidate, citation ranges, score aggregation or stored historical grades. A model can still misjudge supplied evidence. |
| New release no longer matched old cohort identity | Input validation correctly rejected the changed data identity. | Created new cohort manifests with the same exact task membership and ordering, bound to the new release. | Selection, confirmation and eight-task benchmark membership checked unchanged. No confirmation answers generated or inspected. |

Full local suite: **523 tests passed**. See [full-tests.log](full-tests.log). Versioned configs enable both optional fixes; historical configs retain legacy behavior.

The initial repaired release was `repo-qa-training-claims-v5`; its successor adds the missing context-exit evidence described below. The current frozen path and manifest hash are in release-validation.json. [Release validation](release-validation.json), [explicit claims](claim-facts.txt), [configuration validation](config-validation.log).

## Why learning is still difficult

1. **Sparse comparisons within groups.** In the last run, 10/20 groups had identical rewards and supplied no relative learning signal. Only 40 of 80 trajectories contributed, across seven actual updates. More nuanced rubrics can help discriminate responses, but cannot manufacture correctness or useful variance.
2. **Tool reliability still limits answers.** Fifteen of 80 attempts exhausted their budgets, with 54 action errors. The alias fixes only one recognizable syntax pattern. Ten other invalid actions omitted the tool name; guessing would hide a policy failure. A verified tool-use warmup or constrained generation remains an experiment, not an implemented fix.
3. **Output clipping remains.** There were 87 clipped tool observations. Line pagination does not guarantee that serialized source fits the 3,500-byte observation limit. Byte-aware pagination/observation formatting is a separate next fix; simply raising limits risks exhausting the 8,192-token context. This continuation does not claim to have solved it.
4. **Strict evaluation and training reward measure different things.** Training reward preserves credit for required facts even when another assertion or citation is wrong. Strict evaluation also checks every assertion and citation. Higher factual coverage therefore need not increase strict pass rate. Both metrics must remain visible.
5. **Judge and generation variability can exceed a small learning gain.** Two untrained evaluations gave six versus eleven strict passes; answers differed on 23/32 tasks, and one identical answer received differing scores. The verified attribution error is one concrete cause, not an explanation for every failure. Repeated controls and a larger evaluation are still needed.
6. **Changed rubrics break direct reward-curve comparisons.** The 316 explicit claims change which facts are scored and their within-task weights. A fresh baseline under the same new data, evidence policy and environment is required. The old curve must not be spliced into the new one.
7. **The references remain automatically reviewed drafts.** Some tasks ask speculative performance, architecture or redesign questions. Source supports actual behavior and can refute exaggerated premises, but does not prove measured speedups, author intent or proposed concurrency guarantees. The repaired claims retain those qualifications. This task mix may still yield noisier RL feedback than narrow factual questions.
8. **Seven updates do not establish a learning trend.** Larger effective batches and a longer bounded run may reduce noise, but selecting attractive training rewards is not proof of generalization. If tool-use SFT is introduced, compare SFT against SFT+GRPO to isolate the RL contribution.

## Validation and next run

A frozen 12-case grader-control experiment uses four TRAINING tasks only: the complete existing reference, one explicit fact, and an unrelated answer for each. Expected ordering is complete > one fact > unrelated. Single-fact credit need not equal exactly 1/N because criteria can overlap; reasons must be inspected without relabeling outcomes. No policy rollouts or optimizer calls are made.

Run: `autoresearch-fixes-v1-grader-controls`; bundle `1fad00c5034bf845c6bdf74e565f7bcedb50e2b7d9990aabf4a4ddb714f8f45f`.
Model-reservation ceiling: $35, plus the bounded Modal controller. Before launch the authoritative GRPO ledger was $225.63509/$550. Project cap remains $1,000. These ledger amounts are reservations, not provider invoices.

After these controls, the prepared `fixes-v1-benchmark.json` should measure tool success and contributing-group counts. A new training run then needs its own initial evaluation. The benchmark and training configs have **not** been submitted. The earlier heartbeat automation remains paused; this continuation is direct user-authorized work.

Reproduce locally from the repository root:

```sh
PYTHONPATH=. .venv-eval/bin/python scripts/repair_training_placeholders.py --target <new-release-path>
PYTHONPATH=. .venv-eval/bin/python scripts/configure_autoresearch_fixes.py
.venv-eval/bin/python -m unittest discover -s tests -q
```

Release freezing refuses an existing destination. The checked-in report/config recipes bind the validated v5 release; a different target needs a matching release-validation.json before configuring. Never resubmit a bundle with a receipt or replay an uncertain optimizer call.

Offline replay also verified that **all eight** saved training actions using the `action` alias now pass parser and tool-argument validation. This is not a rollout-success or reward-improvement claim. See [alias-replay.json](alias-replay.json).

## Completed live validation

The 12-case run completed with **12/12 resolved factual scores and no request/format errors**. The strict pass resolved 11/12; one reference-evidence gap remained unresolved and is reported separately. Every complete reference scored **1.0**, every unrelated answer scored **0.0**, and the single-fact controls scored **1/7, 1/5, 1/5 and 1/4**, matching their respective rubric sizes. This demonstrates useful partial-credit separation on the four sampled training tasks. It does not validate every repaired rubric or establish policy improvement.

[Validation summary](validation-summary.json), [raw results and per-claim judgments](results.json), [frozen fixture](fixture.json), [Modal submission receipt](submission.json). Model reservations were **$3.077313**; controller reservations are recorded separately in [budget-after.json](budget-after.json). No new W&B training curve exists because this validation performed no training. The previous learning run remains linked in [the main report](../README.md).

The changes are ready for a fresh bounded benchmark/baseline. Scaling directly to a long GRPO run would still leave the output-clipping and low-contributing-group problems unmeasured under these new conditions.

## Evidence gap found and repaired by the controls

The strict judge could not verify `Response context exit calls close` because the reference excerpt contained `close` but omitted `__exit__`. The pinned file has `Response.__exit__` at lines 708–709, calling `self.close()`. A new immutable **v6** release adds those two lines and binds that claim directly to them; claim text and scoring weights are unchanged. The v5 release and its unresolved result remain preserved. See [source-backed repair](context-exit-evidence-repair.json) and [original uncertainty](strict-unresolved.json). Three affected training-only controls are being rerun under a distinct receipt; no optimization occurs.

The final v6 integrity comparison confirms **exactly 78 changed grading records, all training-only; 909 other records unchanged; public/evaluation files byte-identical**. See [integrity check](final-release-integrity.json). The new cohort hashes bind the new data identity while preserving selection and confirmation membership.

### Evidence follow-up result

The three affected controls completed under v6: factual rewards **1.0 / 0.2 / 0.0** for complete / one fact / unrelated. **All three strict assessments resolved**, with no reference disagreements, so the missing-source uncertainty is gone. Their strict scores remained zero: resolution is not acceptance, and the complete answer still failed citation-support checks. The synthetic controls test factual-credit separation, not robust strict-citation success. Original judgments remain unchanged. [Follow-up summary](control-v2/summary.json), [raw follow-up judgments](control-v2/results.json), [receipt](control-v2/submission.json).

No paid experiment remains active. No policy training was performed in this continuation. Final reconciled spending is in [the ledger summary](control-v2/budget-after.json).
