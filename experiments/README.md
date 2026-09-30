# Baseline experiment records

## Current runnable configuration

Use `baseline-qwen35-4b.generated-dev.v2.json` for the corrected baseline:

```sh
.venv-eval/bin/python -m eval_pipeline.baseline --config experiments/baseline-qwen35-4b.generated-dev.v2.json
```

V2 keeps the same solver, judge, tasks and total output-token budget. It returns
invalid tool arguments as observations instead of terminating the task, allows
at most 12 investigation calls, and reserves 2,500 output tokens for an explicit
final-answer phase. Missing evidence must be disclosed. This is a distinct harness
configuration, not a claim that the model independently learned to stop sooner.

The first full v1 run
[13yhygmt](https://wandb.ai/nmdatar-harvard-university/repository-qa-eval/runs/13yhygmt)
attempted all 12 tasks but produced zero final answers: context/tool limits,
malformed actions, and terminal path-error handling prevented completion. It is
retained as a failed integration baseline. The corrected run is
[qfedsspa](https://wandb.ai/nmdatar-harvard-university/repository-qa-eval/runs/qfedsspa).
It completed all 12 answers and grades, with 144 tool calls and no runner failures.
Mean weighted claim coverage was 30.83%; this confirms pipeline operation, not
high answer quality. See [the verification report](../reports/baseline-pipeline-v2.md).
Execution-status text in a frozen experiment config reflects when it was authored;
the run summary and verification report are authoritative for completed outcomes.

## Original configuration

`baseline-qwen35-4b.generated-dev.v1.json` records the first intended live
baseline: untouched Qwen3.5-4B, a larger Qwen3.5-397B-A17B judge, twelve
generated development tasks, Modal tools, and online W&B tracking.

Run this config with the native Tinker/Modal baseline runner:

```sh
.venv-eval/bin/python -m eval_pipeline.baseline --config experiments/baseline-qwen35-4b.generated-dev.v1.json --validate-only
.venv-eval/bin/python -m eval_pipeline.baseline --config experiments/baseline-qwen35-4b.generated-dev.v1.json --limit 1
.venv-eval/bin/python -m eval_pipeline.baseline --config experiments/baseline-qwen35-4b.generated-dev.v1.json
```

`--limit 1` is a separately tagged smoke subset. Omit it for all 12 tasks.
The existing `eval_pipeline.run` command remains a separate retrieval runner.
This tool loop uses one JSON action per model response, with thinking disabled
through the model's Hugging Face chat template. It records the template hash.
The solver receives only public prompts and repository observations. The judge
receives references separately. Weighted claim coverage is not a strict RL reward.
Invalid responses, exhausted budgets, and provider failures remain visible.
Context overflow fails explicitly rather than silently dropping evidence.
The JSON adapter accepts equivalent flat tool arguments. When a response ends
naturally with complete string values but lacks only terminal object braces,
it can append those braces; every such repair is counted and raw output retained.
It never repairs truncated string content or responses stopped at the token limit.

The first successful live smoke run is
[3acsb3l5](https://wandb.ai/nmdatar-harvard-university/repository-qa-eval/runs/3acsb3l5):
one task answered and graded, real Modal source tools, and uploaded artifacts.
The reference-claim score was 0.8333. This confirms the pipeline, not model quality
or independent verification of every execution claim. Earlier debugging runs remain
in W&B and are tagged `smoke-subset`; exclude those when comparing full baselines.

## Authenticate once

From the project directory, run in your own terminal:

```sh
.venv-eval/bin/tinker auth login
.venv-eval/bin/tinker auth status
```

The login command prints a Tinker Console link. Create a key there and paste it
into the terminal's hidden prompt. The SDK stores it in
`~/.tinker/credentials.json`, outside this repository. Alternatively, the SDK
accepts `TINKER_API_KEY`. Do not put a key in the experiment config or chat.
The same account serves both solver and judge. A funded Tinker account is needed.

Tinker SDK 0.30.4 is installed in `.venv-eval`. Dependency versions are recorded
in `requirements-baseline.txt`. W&B credentials already exist locally; live
access must still be checked at launch.

## Track comparable experiments

Copy the config under a new experiment ID when changing models, prompts,
budgets, grading, or task selection. Each run must save its exact config and
SHA-256 in a unique output directory, plus resolved model/renderer identities,
SDK versions, image IDs, prompts, trajectories, judgments, and failure counts.
Use the experiment ID as the W&B group/name prefix and a unique run suffix.

The config pins both existing releases with hashes and an explicit list of all
12 task IDs. Their paths currently point to the local dataset-generation
worktree. A moved release may use a new path only if the hashes still match.
Keep references outside the solver context. These tasks have draft references
and are development/regression checks, not a held-out quality benchmark.

The proposed grader is reference-claim assessment with observed tool evidence;
it is not a calibrated strict RL verifier. Record its prompt/version at launch.
Keep provider failures unresolved and show coverage against all 12 tasks.

Sources: [Tinker authentication](https://tinker-docs.thinkingmachines.ai/tinker/cli/auth/)
and [model catalog](https://tinker-docs.thinkingmachines.ai/tinker/models/).
