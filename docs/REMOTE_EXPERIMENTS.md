# Remote experiment execution

All collection screens and experiments now run in a **detached Modal controller
sandbox**. The complete agent loop, grading orchestration, training orchestration,
and local artifact writes run there. Repository commands remain in separate,
credential-free Modal sandboxes. Tinker supplies policy sampling, frozen judge
sampling, forward/backward, optimizer updates, and checkpoint storage. The laptop
only prepares an immutable bundle, submits it, and retrieves results.

The shared scientific engine is retained; it is not a second local production
path. Collection execution from `qa-train run`, `baseline`, `benchmark`, `resume`,
or checkpoint evaluation refuses to allocate a backend outside the Modal worker.
Local offline validation, comparisons, and synthetic tests remain supported.
The generic `agent_harness.remote_cli` is a different task adapter and is not the
entrypoint for these collection experiments.

## Frozen grading contract

Initial configs use `Qwen/Qwen3.5-4B` for the policy and
`Qwen/Qwen3.5-397B-A17B` for the judge. `all-claims-v3` reuses the robust
`qa_eval.judging.judge`, schema/provenance validation, and `qa_eval.grading.decide`:

1. Independently extract every substantive answer assertion, including extra
   assertions, conditions, negations and execution claims.
2. Assess every required reference claim and every extracted assertion, with
   source evidence and citation entailment. Omitted/duplicate/unknown claims or
   invented evidence fail validation. Truncated/incomplete audits are unresolved.
3. Material falsehoods, false execution claims, unsupported extra assertions, or
   uncited assertions block reward. Otherwise the reward is required-claim
   coverage. An unresolved assessment stays null and is excluded from GRPO.

The extractor gets a catalog of observed/reference files and may request bounded,
hash-bound source ranges. Other source requests must identify a file in the pinned
inventory with its exact hash. Insufficient evidence is not proof of incorrectness;
review/uncertainty remains unresolved. Collection answers do not support diagrams.

This is the strict semantic verifier adapted to **automatically admitted research
references**, not a claim that those references acquired human review. We do not
bypass or relabel the strict production evaluator's human-gold gate. The same
frozen Qwen judge is used for training and selection, so selection is explicitly
**not independent evaluation**. The model had prior diagnostics; this new two-stage
adapter passed a two-case live smoke, but still needs broader calibration before quality claims. The prior
`source-claims-v2` results cannot be compared directly with these runs.

## Prepare and restart

Stop/reconcile the previous local runs before preparing a campaign. Do not cancel
an optimizer call and blindly repeat it. These code changes do not stop existing
processes. Use the newly versioned config run IDs; old outputs remain untouched.
Existing local ledgers are imported and cloud spending must extend that history;
a divergent ledger stops execution instead of resetting the budget.

From the project root:

```sh
.venv-eval/bin/python -m training_pipeline.remote prepare \
  --configs configs/experiments/01-throughput-c08-r1.json \
            configs/experiments/01-throughput-c16-r1.json \
            configs/experiments/02-direct-grpo-screen.json \
            configs/experiments/03-lr-5e-6-small-screen.json \
  --parallel-training \
  --output artifacts/remote-initial-screens

.venv-eval/bin/python -m training_pipeline.remote submit \
  --bundle artifacts/remote-initial-screens
```

Preparation is offline: it verifies inputs, computes the total against each shared
ledger, snapshots the exact source code, and packages the release and Git bundles.
The submission builds a CPU-only Modal image and launches the detached sandbox.
It requires Modal authentication locally and a Modal secret named
`repository-qa-training-credentials` containing `TINKER_API_KEY`, `MODAL_TOKEN_ID`,
`MODAL_TOKEN_SECRET`, and `WANDB_API_KEY`. Supply these through the Modal secret UI;
never put them in configs or bundles. Credentials only enter the trusted controller;
tool sandboxes receive none. The private secret and controller launch were verified
by the bounded live validation on 2026-09-29; see the live validation reports for
the separate grader and optimizer outcomes.

The singleton named controller `experiment-controller` prevents simultaneous
campaigns from writing the same volume from different hosts. Within it, up to four
training subprocesses may run concurrently and share process-locked ledgers.
Throughput arms always finish sequentially before training begins. Per-arm rollout
and judge limits still apply. All initial configs bound controller resources to
2 CPUs, 4 GiB RAM, and two hours. A timeout is an incomplete/possibly ambiguous
run, not permission to replay updates.

The receipt next to the bundle contains the sandbox ID. A pending receipt after a
lost submission response requires inspection; `submit` refuses blind retries.

```sh
.venv-eval/bin/python -m training_pipeline.remote status --sandbox-id sb-REPLACE
.venv-eval/bin/python -m training_pipeline.remote download \
  --bundle-id REPLACE_WITH_BUNDLE_HASH --output artifacts/remote-results
```

Logs/results live in the `repository-qa-training-state-v2` volume under
`campaigns/<bundle-id>/`; run traces, checkpoints and authoritative ledgers live
under `artifacts/`. The v2 volume uses background commits and an explicit final `sync` to persist
results. Every paid-call reservation is explicitly synced before dispatch. This
uses [Modal’s documented sandbox volume sync](https://modal.com/docs/guide/sandbox-files). Download after completion; inspect partial artifacts before any recovery.
Automated remote checkpoint resume/fork is not exposed by this launcher yet.

## Budgets and interpretation

Two judge calls, each reserving up to 4,096 output tokens, replace the old one-call
1,024-token judge estimate. Controller compute reserves its entire lifetime at
hard CPU/memory limits with a 2x allowance, charged conservatively to every arm.
Caps and ledger identities are unchanged. To fit those caps, small screens now
allow four attempted batches and up to four updates; larger direct GRPO allows
18 batches/updates. The older LR screens allow six batches/up to five updates.
No-signal and coverage stops still apply. Preparation sums reservations across
configs sharing a ledger; not every listed experiment can be funded together.

Reservations are not provider bills. Image builds and volume storage are additional
Modal infrastructure costs not metered by the token ledger; reconcile them against
the project's infrastructure/contingency allocation before submitting. These
changes have offline coverage and a bounded live smoke (see
`../reports/remote-live-validation.md`). That verifies remote placement, two grading
cases, and a disposable Tinker update/checkpoint, not broad grader calibration,
policy quality, or a throughput improvement.

## W&B logging volume

`tracking.upload_policy` defaults to `metrics-only`. This publishes aggregate
training/evaluation scalar metrics and update timing, preserving existing chart
names and step axes. Per-trajectory events, nested diagnostics, answer tables,
HTML viewers and artifacts remain in the experiment directory on the Modal
volume. Console capture, system-stat collection and automatic code uploads are
disabled. `tracking.flush_every` controls local answer/viewer persistence; it does
not enable remote media uploads.

Set `tracking.upload_policy: full` explicitly only when rich W&B uploads are
wanted. Full mode includes sampled trajectory tables/artifacts and final answer,
judge and trajectory viewers, which can consume substantial storage. The default
retains `events.jsonl`, `answers.json`, trajectories and generated viewer files
for inspection and recovery even if W&B rejects an upload.

Changes affect newly packaged launches. Existing detached controllers use their
immutable submitted source; editing local settings does not hot-patch them.
