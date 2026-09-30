# Direct seed42 v2 loss-validation failure: offline diagnosis

The v2 structured log places the failure in **loss_reduction_validation**, with **optimizer_call_attempted=false**. Therefore the failed third batch did not call optim_step. Forward/backward did complete and may have accumulated gradients; the trainer must still be discarded. This is narrower than the first run's uncertainty, but it does not establish which validation assertion failed.

The frozen check can raise ValueError for mismatched output-row count, mismatched token-score count, nonfinite learner logprob, or disagreement between local loss and reported loss:sum. Original returned scores and measured loss were not saved on failure, so the archive cannot distinguish those cases. No new training, sampling, or provider-forward operation was invoked in this audit.

## Reconstructed cached batch

Last good checkpoint: `ckpt-ecffde9a9dfe476dafce13ea0e35c815`, optimizer_step2, cursor16, prior baseline count64/reward_sum9.45. Its training path is `tinker://db69c7c4-7a1e-5b6f-969c-ea9e56d13ca9:train:0/weights/ckpt-ecffde9a9dfe476dafce13ea0e35c815`.

The32cached batch3 trajectories reconstruct8groups,168training rows,329,867input tokens and6,977weighted generated tokens. All are tied to checkpoint2's sampler. With learner logprobs set equal to the cached behavior logprobs, the objective is−0.07239583333333334 and the sum of absolute terms0.2583821614583333: cancellation ratio3.569. The configured acceptance tolerance at that hypothetical objective is0.00007239583333333334.

Float32 input rounding under the same on-policy assumption changes the reconstructed objective by approximately5.39e−10; cached behavior logprobs are already exactly float32-representable. Total absolute weight-rounding error is6.09e−9. Serial Python summation and math.fsum differ by approximately1.46e−15. These are far below the unchanged tolerance. **The cached batch does not support blaming ordinary input float32 rounding, Python accumulation, or extreme cancellation for the failure.** Actual learner scores were not retained, so this is a diagnostic bound under the stated on-policy assumption, not a reconstructed provider result.

The two preceding acknowledged updates had local-vs-provider absolute loss differences of approximately2.18e−8 and2.24e−9. The SDK's local chunk-combiner explicitly sums :sum metrics, rather than averaging them. A normalization bug is not established by the available evidence.

Negative-advantage cancellation can make relative error against the net loss an ill-conditioned general test. At extreme cancellation, an appropriate tolerance must account for precision and the absolute contribution scale. This theoretical issue does not justify increasing tolerance on this incident: the observed reconstructed ratio is modest and actual failed outputs are absent. Do not replace the existing test with a blanket relaxed threshold, or use absolute-term scaling without a justified dtype/error model and false-acceptance tests.

## Safe next diagnostic, no optimizer replay

Prepare a bounded, separately logged **forward-only** diagnostic on a fresh trainer loaded from checkpoint2. Reconstruct the cached trajectory/token/advantage multiset from frozen artifacts and baseline64/9.45; do not regenerate or regrade trajectories. Validate model, tokenizer, renderer and checkpoint identities first. Reserve the329,867input tokens (at the frozen$0.737/M training-token reservation rate, approximately$0.243112 provider-forward reservation; add separately budgeted controller overhead).

Use trainer.forward(...,loss_fn='importance_sampling') only. Do not use forward_backward, optim_step, the discarded trainer, or Pipeline.run/resume. The existing inspect_loss helper calls forward only but requires loading a fresh training client. Capture numeric validation diagnostics before failing closed. If needed, save the private returned token-score tensors and numeric submitted loss inputs locally for exact offline reanalysis; never log source text/token IDs publicly. This diagnostic cannot change model weights or apply the uncertain batch update, but it is still a new paid forward request and needs the normal budget gate.

The exact original within-group submission order is not retained: trajectory events reflect completion order, and duplicate rewards prevent recovering every rollout slot. Reconstruct the same multiset in a documented canonical order. A new forward result can distinguish shape/missing/nonfinite/numeric problems if reproduced, but cannot recover the lost original response or prove that a transient error did not occur. Do not call the new request byte-identical replay.

## Local observability change

check_reduction now records a fixed categorical failure reason plus safe aggregate quantities. Numeric mismatch diagnostics include measured/expected losses, absolute error, unchanged accepted tolerance, absolute contribution sum, stable-sum estimate, float32-input estimate, positive/negative contribution counts, weighted token count and cancellation ratio. Shape and nonfinite failures have distinct categorical reasons. The update failure event includes these diagnostics; no raw SDK error strings, request text or token IDs are emitted. Existing1e−3relative/1e−5absolute tolerance and acceptance computation remain unchanged.

Six new offline reduction-diagnostics tests pass, alongside three update-diagnostic tests and27existing training tests. Tests explicitly verify that extreme cancellation remains rejected under the existing tolerance and wrong scaling is still rejected.

Machine-readable reconstruction: `reports/expanded-studies/direct-v2-offline-loss-audit.json`. Source archive: `artifacts/expanded-direct-seed42-v2-final-failure-results`.

## Confirmed forward-only reproduction and verified transport workaround

Subsequent authorized forward-only probes are saved under `reports/expanded-studies/forward-diagnostics/`. The unchanged168-row request reproduced a sum mismatch twice: local−0.07222141599705811 versus provider−0.1444428313698154, approximately exactly twice the expected sum. The retained numeric diagnostics establish a reduction/aggregation disagreement, rather than the previously speculative rounding explanation. No backward or optimizer call occurred in these probes.

The v3 probe submitted the same168rows/329,867tokens in six sequential requests, each under60,000inputtokens. **All six independently passed the unchanged loss check**, with errors around1e−9. The localSDK's default request limit is5,000,000encodedbytes; dense RL rows use approximately20bytes/inputtoken, so the original request crosses that limit and is internally split. The SDK submits internal chunks in parallel and sums each returned :sum metric. The observed factor2 is consistent with duplicated aggregate metrics during internal chunking; server internals are not available, so the precise server/SDK boundary is not asserted as proven.

The local workaround makes every forward/backward RPC explicitly small and sequential: no more than32rows or1MiB under a conservative24bytes/inputtoken estimate, respecting any smaller live SDK limit. Each response must pass the original check before another chunk is sent. Global per-token weights are preserved without renormalizing by chunk size. Gradients accumulate over all verified chunks, followed by **one optimizer call for the entire original batch**. A failed chunk poisons the trainer and prevents optimizer execution. Both provider and reconstructed loss totals are summed for reporting; no mismatch is divided away and no tolerance is relaxed.

Gradient accumulation across forward_backward calls and applying accumulated gradients in optim_step are documented by [Tinker](https://tinker-docs.thinkingmachines.ai/) and [Thinking Machines](https://thinkingmachines.ai/tinker/). This changes transport and validation granularity, not the intended batch objective, task schedule, or optimizer cadence. Floating-point summation order can still differ, as with ordinary microbatch accumulation.

Validation:4newchunkingtests,6reductiondiagnostictests,3updatefailuretests and27existingtrainingtests passed. Coverage includes preserved row/weight objects, sequential call order, exactly one optimizer, late-chunk failure halting all remaining calls, live SDK limits and oversized-datum rejection before paid work. Root controls the immutable-checkpoint continuation launch.
