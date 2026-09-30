# Direct seed42 failure: read-only audit and safe recovery boundaries

The archived first launch `expanded-direct-seed42-v1` failed after one acknowledged, committed optimizer update. No optimizer or generation calls were made during this audit.

Evidence:

- Initial checkpoint: `ckpt-854c974bbda94a09bd2c0bc42380851a`.
- Last immutable acknowledged checkpoint: `ckpt-3b77a42a5f8d40a8a941fb0b2ffebe3e`.
- That checkpoint records optimizer_step=1, attempted_batches=1, cursor=8, and prior reward baseline count32/sum4.0.
- Batch1:32resolved trajectories,5contributors,meanreward0.125; update1acknowledged and checkpoint committed.
- Batch2:32resolved trajectories,32contributors,meanreward0.31875,priorbaseline0.125. The batch event was written before the backend update; no update2acknowledgment or checkpoint2 exists.
- The frozen CLI printed only `Training stopped: AmbiguousUpdate`. It suppressed the wrapped error's type; provider forward/backward outputs were not persisted.

## What is unknown

The backend groups forward/backward, local loss-reduction validation and optimizer execution in one protected block. Any exception poisons the trainer and produces AmbiguousUpdate. The archived evidence cannot establish which operation failed, whether optim_step was sent, or whether an unacknowledged optimizer update completed remotely. Higher contributing trajectory count increased work, but this is not evidence of a timeout. No numerical tolerance should change based on this incident.

## Recovery choices

1. Preserve the failed run as an infrastructure attempt. Launch a separately identified replacement from the unchanged fresh base under a frozen replacement policy, after confirming the local diagnostics patch and budget. This does not replay an optimizer request on the uncertain trainer and preserves the intended scientific starting condition. It still repeats work and costs; disclose the failed attempt and never selectively discard completed outcome data. Root controls launch authorization; this audit launches nothing.
2. A bespoke restore from immutable checkpoint1 could abandon the uncertain trainer, explicitly skip batch2's eight tasks, and continue on the remaining frozen schedule. This changes exposure and requires a revised matched-arm protocol: record exclusion IDs, restore exact baseline/optimizer state, specify whether skipped observations enter the baseline, and report fewer planned task updates or an identically applied replacement rule. Do not edit the original signed checkpoint. Existing ordinary fork resets task order/cursor/baseline/optimizer semantics and does not implement this recovery. A fork alone would silently revisit tasks.
3. Ordinary resume regenerates batch2 from checkpoint1 on restored state. It is not allowed under the current explicit no-replay instruction and must not run automatically. Never reuse the poisoned trainer or submit the previous optim_step again.

A known-good checkpoint proves the prior state, not the outcome of the later uncertain request. Reading provider metadata may establish corruption/availability but cannot retroactively infer acknowledgment without authoritative operation records.

## Future-run observability fix

Local code now attaches categorical diagnostics for `forward_backward`, `loss_reduction_validation`, or `optimizer_step`, cause class, whether optimizer invocation was attempted, acknowledgment=false, poisoned=true, and replay_safe=false. The CLI prints a structured `ambiguous_update` event. Raw SDK exception strings, request bodies, URLs and credentials are excluded. Loss validation failure and forward/backward failure explicitly show that the optimizer call was not attempted; neither permits reusing accumulated-gradient state. No retry or numerical objective changed.

Focused offline tests cover each failure phase, suppression of synthetic secret-bearing messages, absence of optimizer calls on pre-optimizer failure, and rejection of a second update on the poisoned trainer. Three tests passed plus27existing training-pipeline tests.

Source archive: `artifacts/expanded-direct-seed42-failure-results`. Archive report: `reports/expanded-studies/direct-seed42-failure-archive.json`.
