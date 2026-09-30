# Bash-only correctness GRPO pilot

Live run: [W&B](https://wandb.ai/nmdatar-harvard-university/repository-qa-training/runs/bash-correctness-grpo-v1).

This experiment post-trains Qwen/Qwen3.5-4B using rank-8 LoRA adapters, seed 42, GRPO, learning rate 1e-5, eight questions per batch and eight sampled attempts per question. Three batches allow at most three optimizer updates and 192 training attempts. Each attempt has a 30-call bash budget, 31 generations, 6,000 output tokens total, and 512 tokens per generation. Bash has no command allowlist; the existing sandbox isolation remains.

The selected training subset contains 32 questions from the previously validated 274-question pool. Three batches consume 24 training questions. The fixed 32-question validation cohort is evaluated before and after, at temperature zero, and is never included in gradient updates. The separate 85-question confirmation cohort is not used.

## Completed results

All three optimizer updates succeeded: 192 training attempts across 24 questions, with 8 attempts per question. All training grades resolved. The final checkpoint was saved and successfully evaluated; the controller exited with code 0. W&B recorded zero tracking failures. The local archive contains 2,107 files with SHA-256 checksums.

| Metric | Base model | After 3 updates |
|---|---:|---:|
| Required-fact correctness mean, resolved cases | 0.2028 | 0.6694 |
| Resolved evaluation cases | 30/32 | 30/32 |
| Full required-fact coverage | 4/32 | 17/32 |
| Citation diagnostic mean, assessed cases | 0.0893 | 0.4162 |
| Evaluation attempts using bash | 5/32 | 32/32 |
| Mean bash calls per evaluation attempt | 2.7813 | 16.5313 |

The unresolved cases are not identical across passes. On the **29 matched resolved cases**, correctness increased **0.1925 → 0.6580**, a mean difference of **+0.4655**. Eighteen cases improved, one worsened, and ten tied. This is a substantial gain on this validation cohort, with the limitations of a small, single-seed experiment and an uncalibrated model judge. It is not a measured result on the untouched 85-case confirmation set.

Training batch correctness means were **0.0391, 0.1322, 0.1582**; citation means were **0.0150, 0.0279, 0.0549**. These batches used different questions. Seven groups had equal rewards and contributed no gradient. No training group was excluded for unresolved grading.

Recorded budget reservations were **$93.45**, below the approved $850 ceiling; these are not provider invoices. The final model references and checkpoint manifest are in [posttrained-model.json](posttrained-model.json). Its Tinker checkpoints have a 48-hour TTL from saving; the local JSON files are metadata, not model-weight exports.

Inspect [trajectories and assertions](trajectories.html), [paired case results](paired-results.csv), [machine-readable summary](summary.json), and [per-batch logs](training-batches.json).

## What the reward means

The Qwen3.5-397B-A17B judge independently checks required facts against pinned repository evidence. Correct supported coverage earns credit, partial coverage earns partial credit, and missing/unsupported coverage earns zero. The mean across required facts is the reward. Missing or invalid citations do not block factual grading or lower this reward. Additional incorrect claims do not receive a separate penalty.

Citation score is the fraction of extracted answer claims supported by a citation. Missing/invalid citations score zero. An incomplete citation assessment stays unknown, not a fabricated zero. The older strict audit is retained only for diagnostics.

Watch these W&B charts:

- `training/mean_correctness_score` and `evaluation/mean_correctness_score`.
- `training/mean_citation_score` and `evaluation/mean_citation_score`.
- `training/scoring_coverage`, `training/excluded_groups`, and `training/zero_variance_groups`.
- `training/contributing_trajectories` and `optimizer_step`.

Full trajectory and grading viewers are uploaded at the end. Raw commands, stdout/stderr observations, model generations, final answers, and per-claim grading records are retained in the experiment artifacts.

## How post-training works here

1. Start from the base model and save an untrained checkpoint; evaluate the 32 validation cases.
2. Sample eight answers for each of eight training questions at temperature 1.
3. Compute a correctness reward for every answer and normalize rewards within each question's group.
4. Use the resulting advantages to update the LoRA adapter through Tinker. Tool-output tokens are observations; gradients are applied to model-generated tokens. Equal-reward groups contribute no gradient; unresolved groups are excluded.
5. Save a checkpoint after every acknowledged optimizer update and evaluate the final checkpoint on the same 32 validation cases.

The checkpoint manifests identify both training-state and sampling checkpoints. Sampling checkpoints can be used for model inference; training-state checkpoints support restoring optimizer state for an interrupted run. Extending a completed experiment requires a new run/configuration with `execution.operation` set to `fork` and a checkpoint available on its state volume. A longer experiment is not launched by this pilot. Tinker checkpoints currently have a two-day TTL; durable model-weight export or longer retention should be arranged before expiration if retaining the model beyond the pilot.

## Reproducibility and results

Configuration: `configs/experiments/bash-correctness-grpo-v1/run.json`; exact task IDs: the adjacent `subset.json`. The immutable submitted bundle is `artifacts/bash-correctness-grpo-v1-bundle` and its submission receipt records the isolated Modal sandbox and volume. The run has an $850 reservation ceiling; reservations are not provider invoices or a provider-enforced billing cap.

`summary.json`, `training-batches.json`, `paired-results.csv`, and `trajectories.html` hold the measured results. A three-batch, single-seed pilot can establish that the training loop runs and provide an initial signal; it cannot establish a reliable general performance gain.
