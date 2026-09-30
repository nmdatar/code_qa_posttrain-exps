# Shell-only eval: completed comparison

The approved four-run campaign completed successfully on Modal. All 128 episodes
were attempted, with no optimizer updates. The structured baseline passed 15/64
repeated task episodes (23.44%); shell-only passed 0/64 (0%). Keep the baseline.

| Arm | Repeat 1 strict passes | Repeat 2 strict passes | Resolved grades | Completed answers |
|---|---:|---:|---:|---:|
| Structured baseline | 7/32 (21.88%) | 8/32 (25.00%) | 57/64 | 63/64 |
| Restricted shell | 0/32 (0%) | 0/32 (0%) | 64/64 | 64/64 |

These are two inference repetitions on the same 32 selection tasks, not 64
independent tasks. Seven baseline grades were unresolved; percentages use all
assigned episodes, while unresolved grades remain null in raw data. Baseline
coverage (87.5% and 90.6%) is below the 95% promotion gate. No confirmation tasks
were used. The Qwen grader is not independently human calibrated.

## Failure analysis

62 of 64 shell episodes attempted no tool call. The other two each repeated
`rg -n -F 'PriorityThreadPoolExecutor' --files` five times, outside the documented
dialect which separates file listing from content search. No shell command
succeeded. All 64 final answers lacked citations and received deterministic zero
before semantic grading. There were also 49 rejected answer attempts citing
unobserved files. Baseline attempted 267 tool calls; 266 succeeded.

This particular shell interface and prompt failed to elicit tool use from the
frozen model. It does not establish that shell investigation is inherently worse.
The intervention includes tool name, syntax and instructions, and uses a restricted
command-string dialect over existing source readers, not native Bash. A useful
next step is a separately versioned small smoke test with a concise concrete
search/read example before another full comparison. No follow-up was launched.

## Uncertainty, efficiency and cost

Shell-minus-baseline differences were -21.88 and -25.00 percentage points.
Repository-cluster bootstrap 95% intervals were [-30.51, -4.76] and [-38.36, 0.00]
points: 7 families and 10,000 draws. Unresolved-grade sensitivity bounds are in
summary.json. These are selection-screen results, not a confirmed promotion.

Shell used 127,827 input tokens versus 726,392 for baseline; median episode latency
was about 4.3–4.6 versus 18.3–18.5 seconds. It skipped investigation and semantic
grading, so these reductions are not a quality-preserving efficiency gain.
Recorded reservations total **$28.29**, including controller allocations, within
the approved $160 ceiling. Actual provider billing is unavailable; reservations
are not an invoice. No historical ledger was reset or unrelated run interrupted.

## Artifacts

- summary.json: quality, coverage, tokens, timing and paired intervals.
- diagnostics.json: tool use, errors, failure reasons and reservations.
- paired-tasks.csv: matched scores with unresolved values preserved.
- launch-status.json: approval, receipt, bundle ID and completion.
- ../../artifacts/shell-only-v1-results/: 1,018 downloaded campaign/run artifacts,
  including trajectories, private grades, telemetry and ledgers.
- Branch: codex/shell-only-harness; implementation commit: 5593462.
- Bundle: e4ee3860fc7dec3ed376f93ac034214b7031245c803f6dd4fad893730c380266.

Using the existing venv Python with PYTHONPATH=. from this worktree:

```sh
python scripts/collect_shell_eval.py --download
python scripts/summarize_shell_eval.py --root artifacts/shell-only-v1-results
python scripts/diagnose_shell_eval.py
```

The packaged code matched the verified implementation at submission; 58 targeted
regression tests passed before launch. Published model and Modal rates matched
the configs at the 2026-09-29 preflight.
