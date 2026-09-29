# Run configurations

Each JSON file describes one reusable run setup. Evaluation configs are consumed by `eval_pipeline.run --config`; training configs use the separate `training_eval run --config` schema documented in [TRAINING.md](../TRAINING.md). Do not interchange the two formats.

## First baseline

[baseline-qwen3.5-9b.json](baseline-qwen3.5-9b.json) selects `Qwen/Qwen3.5-9B`, the first small-model example in [the project description](../PROJECT_DESCRIPTION.md). The description also suggests 4B; duplicate this config and change the model/name/tags to compare it later. The [official model card](https://huggingface.co/Qwen/Qwen3.5-9B) identifies this as the released post-trained model. Here “baseline” means no additional SFT or RL by this project.

The runner performs lexical source retrieval, one answer generation, and independent reference-answer judging. It does not run training, RL, or the future multi-turn environment. The three-question smoke dataset checks plumbing; it is not a representative benchmark score.

```bash
cp configs/eval.env.example configs/eval.env
# Edit configs/eval.env with your dataset directory, endpoints, and judge model.
source configs/eval.env
python -m eval_pipeline.run --config configs/baseline-qwen3.5-9b.json
```

Use a Python environment with `requirements-eval.txt` installed. Authenticate with `wandb login --verify` once. Model keys remain in `MODEL_API_KEY` and `JUDGE_API_KEY`; config JSON stores only their environment-variable names. This runner does not automatically read dotenv files. `configs/*.env` is ignored by Git.

A string of the form `${NAME}` resolves from the environment at launch. Missing values stop execution before inference or logging. Explicit CLI flags override config fields, including unresolved environment references. Config-relative `data`, `predictions`, and `output` paths resolve against the config file's directory. With no `output`, a unique run directory is created under the current working directory's `artifacts/runs/`.

```bash
# Same experiment, another display name or provider-specific model alias:
python -m eval_pipeline.run --config configs/baseline-qwen3.5-9b.json \
  --run-name baseline-qwen3.5-9b-repeat2 --model PROVIDER_MODEL_ALIAS
```

The provider must actually serve the selected Qwen model; setting its name does not deploy it. No judge or endpoint is assumed. Answer generation uses the endpoint's generation defaults, so record/fix those serving settings when comparing runs.

## Organization

`experiment_id` becomes the W&B group; `run_name` is a human-readable label. Every invocation receives a unique run ID even when reusing the same config. `tags` contains `baseline` and `evaluation-only`, and W&B job type is always `evaluation`. `notes` describes the condition. `source_run_id` optionally links a later evaluation to its originating training run. These fields and the config-file checksum are preserved in the local run manifest. W&B uploads the run's answers, reference answers, scores, explanations and report.

To create a new experiment, copy the JSON, choose its model/settings, and change the experiment ID, display name and tags. Keep dataset, judge and rubric fixed for comparable scores. Optional `entity` selects a W&B team; omission uses the authenticated default.
