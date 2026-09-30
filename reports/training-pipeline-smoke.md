# Training machinery verification

Implemented and validated on 2026-09-28. **The real Tinker toy smoke passed:** one
SFT update, two nonzero-contribution GRPO updates, saved checkpoints, optimizer-aware
resume, standalone evaluation, and a fresh-optimizer fork. No repository training
data was created, relabeled, or consumed.

## Delivered machinery

- `qa-train` / `python3 -m training_pipeline`: validate, run, evaluate, fork, resume,
  and bounded smoke commands.
- Optional/repeated synchronous SFT and GRPO stages, exact generated-token records,
  assistant-only loss masks, population-standardized advantages, whole-group
  exclusion/retry, and zero-contribution update skipping.
- Immutable checkpoint manifests with optimizer state, data order/cursor, RNG,
  lineage, and distinct evaluate/fork/resume behavior.
- Shared toy/repository episode runner; fresh Modal environments; bounded source
  tools; trusted strict-evaluator integration; private references kept on the host.
- Durable local events, optional W&B metrics/artifacts, source snapshots, process
  locking, persistent spend reservations, and explicit unknown cost reporting.

See [usage and input contracts](../docs/TRAINING_PIPELINE.md) and the
[example configuration](../examples/training-toy.json).

## Live evidence

Model: `Qwen/Qwen3.5-4B`, LoRA rank 8, Tinker SDK 0.30.4. The task was a deliberately
synthetic two-turn lookup game with a fixed output-dependent reward. Tools ran
locally; the model, training, and checkpoints were real Tinker operations.

| Operation | Evidence |
|---|---|
| SFT | Eight examples, 16 assistant-turn loss rows, one acknowledged update at learning rate `1e-4` |
| GRPO | Two acknowledged updates at learning rate `1e-5`; two tasks × four attempts per batch; real reward variation |
| Resume | Interrupted at committed optimizer step 2; a distinct training client restored optimizer state and completed step 3 |
| Fork | A distinct client loaded final weights with a fresh optimizer, sampled, and saved independently verified artifacts |
| Evaluation | Periodic, standalone, and public-CLI evaluation attributed to immutable sampler checkpoints; final CLI resolved 2/2 development toy tasks |
| Bounds | Two generations/episode, 64 tokens/generation, 512 context tokens; one SFT and at most two GRPO updates |

The actual built-in provider loss reductions were also checked with forward-only
calls against a local calculation from returned token log probabilities. No extra
optimizer updates were performed:

| Loss | Provider sum | Locally reconstructed sum |
|---|---:|---:|
| Cross-entropy | 0.13172162137925625 | 0.13172162301959697 |
| Importance sampling | -0.01635211706161499 | -0.016352126593556594 |

The adapter now performs this reduction check before each optimizer call.

Full run artifacts:
`artifacts/training-smoke/smoke-20260928T233237Z-4311c7/`.
The directory contains `smoke-report.json`, `provider-loss-check.json`, the initial
run's source snapshot, configuration, complete trajectories, private-free toy
results, checkpoint manifests, and standalone/CLI evaluation records.

Final committed checkpoint:
`run/checkpoints/ckpt-5270240e5e0441768deacec01764d88e.json` within that directory.
Its remote sampling weights are
`tinker://67e1d23e-457c-57bf-b0c1-fa483d927c47:train:0/sampler_weights/ckpt-5270240e5e0441768deacec01764d88e`.
Remote artifacts have a 24-hour TTL; local evidence remains available after expiry.

## Spending

The shared ledger reserved **$2.2986**, below the authorized $5 ceiling:

- $2.2857 conservative storage allowance for five checkpoint pairs through their TTL.
- $0.00754 training/forward token reservations.
- $0.00533 uncached sampling token reservations.

These are upper estimates, **not actual billing**. The billing API matched three
training sessions but returned no billing events or cost-data watermark yet.
Actual cost therefore remains unknown. Earlier pricing-fetch failures stopped
before any paid operation. All smoke invocations share the ledger at
`artifacts/training-smoke/spend-ledger.json`.

## Regression verification

**146 tests passed** using `.venv-eval/bin/python -m unittest discover -s tests -q`,
including 25 new training tests. The suite exercises real SDK tensor construction
and injected provider/Modal adapters without paid calls.

## Verification limits

The repository adapter is covered offline with injected Modal clients, source-tool
execution against archive-shaped fixtures, authenticated telemetry, and synthetic
strict judgments. No paid repository training or live calibrated repository judge
was claimed. W&B outage behavior and uploads are tested/integrated; the paid toy
smoke used disabled W&B tracking.

The initial live update loop's source is preserved in the artifact directory.
Subsequent hardening added loss-sum checks, setup timeouts, process/spend locks,
source snapshots, independent evaluation cadence, and archive-compatible repository
tools. Loss checks and the final CLI evaluator were additionally exercised against
the live checkpoint; the full final source is covered by the regression suite.
