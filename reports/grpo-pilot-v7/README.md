# GRPO pilot readiness

Ready for the pilot: 511 tests passed, the offline reward-curve smoke passed, and the frozen bundle was integrity-checked. The paid pilot has now been submitted to Modal; see submission.json for its sandbox identity. The user accepted the seven
unresolved strict baseline grades. They remain unresolved in the baseline report; the
95% confirmation coverage gate was not retroactively marked passed.

## Frozen pilot

- Policy: Qwen3.5-4B, rank 8, seed 42, learning rate 1e-5.
- Judge: Qwen3.5-397B-A17B, all-claims-v7, context 65536, output 8192,
  one mechanical repair, independent factual-coverage-v3 / positive-coverage-v4.
- Data: repo-qa-atomic-claims-v4; 858 training tasks, 117 development tasks.
- Up to four batches / four updates; two tasks × four attempts = eight attempts per batch.
- Fixed 32-task selection evaluation before training, at step 2 and at completion
  (normally step 4). Confirmation tasks are not used to train or select checkpoints.
- Stop after three initial batches without a contributing update. Unresolved training
  groups are excluded, with no whole-group retry. Equal-reward groups cannot update.
- Original policy rollout limits are preserved. Checkpoint each successful update.

[Config](../../configs/experiments/grpo-pilot-v7/pilot.json) ·
[Preflight](preflight.json) · [Frozen bundle identity](bundle-preparation.json)

## Reward curve

W&B project: `repository-qa-training`. Each run saves its actual W&B URL in
`tracking-url.json` after successful initialization. [Live W&B dashboard](https://wandb.ai/nmdatar-harvard-university/repository-qa-training/runs/02-grpo-pilot-qwen4b-v7-seed42).

Plot `training/mean_reward` and `training/reward_ema` against `attempted_batches`.
Mean reward includes every resolved attempt, including resolved members of an excluded
group; `training/eligible_mean_reward` separately describes complete usable groups.
The EMA uses alpha 0.3 and is checkpointed for resume. An entirely unresolved batch
has a missing reward value, not zero, and does not update the EMA.

Also plot `training/scoring_coverage`, `training/excluded_group_fraction`,
`training/zero_variance_groups`, and `training/contributing_trajectories`. Every attempted
batch is logged, even if no optimizer update follows. The optimizer_step recorded on a
batch is the policy that generated its answers, before its subsequent update.

For strict evaluation plot `evaluation/mean_reward`, `evaluation/demonstrated_quality`,
and `evaluation/scoring_coverage` against optimizer_step. The first uses resolved grades;
demonstrated_quality divides confirmed strict passes by all attempted evaluation tasks.
It does not convert unknown episode grades into zeros. The selected task set stays fixed.

The run directory also contains `reward-curve.csv`, `events.jsonl`, `answers.json`, and
individual trajectory records. These local records survive W&B failure. CSV reward and
EMA values are raw and explicitly smoothed series, respectively; no favorable batches
are selected. A four-update pilot checks training operation and early signal, not reliable
learning improvement. Training task mix changes between batches, and upward reward is
not guaranteed; compare the fixed held-out evaluation as well.

## Budget and launch

The conservative new-run bound is $320.676165 including controller reservations.
With $33.337902 previously reserved, the GRPO ledger upper bound is $354.014067,
within its existing $550 allocation and the recorded $1000 project allocation.
This is a reservation bound, not a billed-cost prediction. The baseline ledger's
remaining allocation is not being reused. Submission rechecks remote budget state.

The frozen bundle can be submitted with:

```sh
.venv-eval/bin/python -m training_pipeline.remote submit --bundle artifacts/grpo-pilot-v7-final-bundle
```

Preparation performs no provider training updates. Live provider availability, W&B
connectivity, and enough nonzero within-group reward variance remain runtime conditions.

[Offline logging smoke](offline-smoke.json) · [Full test log](tests-final.log) ·
[Tracking tests](tracking-tests.log) · [Readiness tests](readiness-tests.log)
