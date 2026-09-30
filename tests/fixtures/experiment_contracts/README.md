# Frozen experiment contract fixtures

These byte-for-byte config copies come from commit
`2b70198df1db7b3028448d8f05772e55c52ea6fc`. They preserve schedule/cost and
reward/grader regression inputs while their superseded campaign bundles are
removed from main. They are test inputs, not recommended launch configurations.

- `grpo-schedule.json`: original `configs/experiments/02-direct-grpo.json`.
- `reward-diagnostic.json`: original `configs/experiments/reward-v4/diagnostic.json`.

The remote-execution contract tests share `examples/remote-validation-base.json`
with the bounded remote validation utility. Its historical settings are also
unchanged. Frozen cohorts, task manifests and budget identities have not been
rewritten. See `docs/EXPERIMENT_RETIREMENT.json` for original hashes.
