# Run and understand evaluations

The pipeline now supports a source-retrieval baseline, importing answers from an
external agent, an independent reference-answer judge, local HTML reports, and
optional W&B experiment tracking. It is separate from the stricter claim/evidence
contracts under `qa_eval/`: these initial scores do **not** certify citations,
validate runtime claims, or implement the official SWE-QA-Pro scoring protocol.

## See the report now

```bash
python3 -m eval_pipeline.run --mode demo
```

The command prints the path to `report.html`. Open it in your browser. Demo runs
use synthetic answers and scores, including an intentional failure. They test
reporting, not a model. A verified example is at
`artifacts/runs/demo-verified/report.html`.

Each run gets a unique folder, preserving `config.json`, `results.jsonl`,
`summary.json`, and `report.html`. An explicit `--output` must be a new directory.
Results are written after each question, so completed rows survive interruption;
resuming interrupted inference is not implemented.

## W&B

```bash
python3 -m venv .venv-eval
.venv-eval/bin/python -m pip install -r requirements-eval.txt

# Works without a W&B account; stores a real W&B run locally.
.venv-eval/bin/python -m eval_pipeline.run --mode demo --wandb offline

# Authenticate locally, then upload a demo run to your project.
.venv-eval/bin/wandb login
.venv-eval/bin/python -m eval_pipeline.run --mode demo \
  --wandb online --project repository-qa-eval --entity YOUR_TEAM
```

Omit `--entity` to use the account default. Online mode uploads question text,
reference answers, predictions, judge explanations, report files, and metrics to
that W&B project. API keys are read from environment variables and are not put
in run config. Offline mode is useful for testing or deferring upload; W&B prints
an exact `wandb sync` command. Disabled mode has no SDK dependency. A logging
failure preserves local reports and exits with code 2.

In W&B, compare runs using `eval/mean_correctness`, `eval/mean_completeness`,
`eval/completion_rate`, `eval/scoring_rate`, `eval/mean_latency_seconds`, and
`eval/total_tokens`. Open the **examples** Table to inspect answers and the
**report** HTML panel for the local-style explorer. Each run also stores an
`evaluation` artifact with the result files. Filter out runs tagged `demo`.
These use W&B's [Tables](https://docs.wandb.ai/models/track/log/log-tables)
and [logging APIs](https://docs.wandb.ai/guides/track/log/).

## Evaluate a model

The baseline selects source excerpts with lexical retrieval from the verified
repository snapshot and sends only the question and those excerpts to a model.
It has one retrieval step; it is not an iterative tool-using agent. The model
cannot execute commands or browse this project's gold-answer files.

Both endpoints must support `/v1/chat/completions`. Point them at a hosted API
or a local inference server. Use the model IDs supported by your endpoints.
Configure `MODEL_API_KEY` and `JUDGE_API_KEY` in your shell if authentication is
required; do not put keys in source control. Endpoints without authentication
can leave those variables unset. The judge receives the reference and candidate
answer separately from generation.

```bash
python3 scripts/prepare_dataset.py --checkout
.venv-eval/bin/python -m eval_pipeline.run \
  --mode retrieval \
  --model YOUR_ANSWER_MODEL --base-url https://YOUR_PROVIDER/v1 \
  --judge-model YOUR_JUDGE_MODEL --judge-base-url https://YOUR_JUDGE_PROVIDER/v1 \
  --wandb online --project repository-qa-eval
```

Use `--api-key-env` / `--judge-key-env` to select other environment-variable names.
No endpoint or API key is bundled. The default URL is `http://localhost:8000/v1`.
Every request has a 120-second timeout and no automatic retry. Live runs cost
whatever the selected providers charge; cost estimation is not implemented.

## Evaluate an existing agent

Write one JSONL row per task using IDs from `tasks.jsonl`:

```json
{"id":"swe-qa-pro-a03405efff358801","answer":"The implementation is in …","status":"success"}
```

```bash
.venv-eval/bin/python -m eval_pipeline.run \
  --mode import --predictions path/to/predictions.jsonl --model YOUR_AGENT_LABEL \
  --judge-model YOUR_JUDGE_MODEL --judge-base-url https://YOUR_JUDGE_PROVIDER/v1 \
  --wandb offline
```

Missing predictions remain failures. Duplicate or unknown IDs are rejected.
Non-success status is recorded as agent failure. Imported agent tokens and runtime
are not inferred: usage remains unknown and measured latency covers judging only.
The agent that generates those predictions must keep gold files outside its tools'
accessible filesystem. This import command does not sandbox external agents.

## How to interpret results

1. **Check coverage first.** `answered / expected` identifies generation failures;
   `scored / expected` also captures judge failures. A high score with low coverage
   is not a good run. Non-demo runs return exit code 1 if any task fails.
2. **Read correctness and completeness separately.** An answer can be accurate
   but omit half the question. The five 1–10 dimensions total 5–50. This is a
   subjective reference-based rubric, not accuracy, pass@1, or proof of correctness.
3. **Inspect the lowest-scoring answers.** Open a question to compare the candidate,
   reference, dimension scores, and judge rationale. Review whether the judge's
   criticism is supported. Clarification and abstention can be appropriate.
4. **Check efficiency after quality.** Tokens describe the answer call when the
   provider reports usage; missing usage is shown as unknown, never zero. Judge
   token usage is recorded separately in JSONL/W&B. Latency is end-to-end per
   question, including retrieval and judging (judging only for imported answers).
   Retrieval baseline tool-call count is one harness retrieval operation, not
   shell commands or agent tool calls.
5. **Compare like with like.** Keep task IDs, source commits, context budget,
   judge model, and rubric fixed. Config records dataset and content hashes,
   prompt version, model names, and runner hash. Three questions are a plumbing
   check; they cannot establish a meaningful model ranking. Scale the held-out
   sample and review paired per-question changes before drawing conclusions.

The judge sees the reference, not the full repository. It may favor reference
wording or miss reference errors; citation correctness is unmeasured here. Do not
use these scores as final RL rewards without the stronger evidence checks and
human calibration described in the project's research.

## Validation

```bash
python3 -m unittest discover -s tests -v
```

Preparation, failure denominators, invalid judge output, duplicate IDs, and HTML
escaping have automated coverage. Actual W&B offline logging was exercised with
SDK 0.30.0. No hosted answer/judge model or online W&B upload has been run yet.
