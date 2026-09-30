# Shell-only frozen-policy experiment (procedure 6)

The sole intervention is the tool interface: list_files/search_code/read_file
versus one `shell` tool accepting `{ "command": "..." }`. The shell arm implements
`source-shell-v1`, a deliberately restricted command-string dialect for `rg`,
`ls`, `cat`, `head`, and `sed -n`. It is not unrestricted Bash: there are no
pipelines, substitutions, redirects, writes, arbitrary programs, or code probes.
All commands use the existing sandbox reader and pinned source inventory, with
identical JSON observations and source hashes. Thus this tests command-string
syntax and its instructions, not native Unix output or arbitrary shell power.

The current uncommitted project was snapshotted on this worktree before edits;
`experiment-baseline-snapshot.json` records hashes and the original commit.
Main is untouched. Configs retain current-v8's frozen source manifest, Qwen3.5-4B,
Qwen3.5-397B judge, dataset v6, grader v7, budgets and 32 selection task IDs.
Two repetitions per arm, temperature zero, sequential order control-r1,
shell-r1, shell-r2, control-r2. No training or confirmation tasks are launched.
This is a selection screen; a winner still needs a separately locked confirmation.

Each arm has its own $40 reservation ceiling ($160 aggregate). Preparation records
conservative policy, judge, sandbox and controller estimates; these are not actual
provider invoices. No historical budget ledger is reset. Raw trajectories and
strict grades remain in the remote run outputs. Compare paired tasks, repository
clustered uncertainty, coverage/completion, invalid commands, tokens, tool calls,
latency and reserved costs. Missing grades remain explicitly unresolved.

Run from this worktree using the existing environment's Python:

```sh
PYTHONPATH=. /Users/ndatar/Documents/ChatGPT/action-interview/.venv-eval/bin/python -m unittest tests.test_shell_tools
PYTHONPATH=. /Users/ndatar/Documents/ChatGPT/action-interview/.venv-eval/bin/python -m training_pipeline.remote prepare --configs configs/experiments/shell-only-v1/control-r1.json configs/experiments/shell-only-v1/shell-r1.json configs/experiments/shell-only-v1/shell-r2.json configs/experiments/shell-only-v1/control-r2.json --output artifacts/shell-only-v1-bundle
PYTHONPATH=. /Users/ndatar/Documents/ChatGPT/action-interview/.venv-eval/bin/python -m training_pipeline.remote submit --bundle artifacts/shell-only-v1-bundle
```
