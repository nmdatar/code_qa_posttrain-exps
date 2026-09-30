# Project

## Goal

## Dataset

Start with [SWE-QA-Pro Bench](https://huggingface.co/datasets/TIGER-Lab/SWE-QA-Pro-Bench):
260 repository-level coding Q&A examples across 26 repositories, with reference
answers and exact source commits. Its dataset card declares MIT. This matches the
project's codebase-understanding task and includes an
[official evaluation harness](https://github.com/TIGER-AI-Lab/SWE-QA-Pro/tree/main/eval).

Run a small dataset smoke test (Python 3.11+ and Git; no Python dependencies or API key):

```bash
python3 scripts/prepare_dataset.py --checkout
python3 -m unittest discover -s tests -v
```

This downloads a checksum-verified release, prepares three questions from
`getsentry/responses`, and checks out its exact benchmark commit. It writes
`data/swe-qa-pro/tasks.jsonl`, separate `references.jsonl`, and a provenance
`manifest.json`. Repositories are cached under `artifacts/repos/`.
The smoke test validates data preparation and source availability; it does not
generate answers or report a model-quality score.

See [the dataset quickstart](DATASET_QUICKSTART.md) for the evaluation contract,
full-dataset command, and next steps for a live model run.

## Model Training

Modal environment infrastructure is available in `agent_harness`: prepare a pinned
repository image once, then create a clean, bounded sandbox for each attempt.

```bash
python3 -m pip install -e '.[modal]'
python3 -m agent_harness --help
```

See [Modal environments](docs/MODAL_ENVIRONMENTS.md) for recipes, lifecycle APIs,
offline tests, and the opt-in live smoke test.

The reusable Tinker pipeline now supports SFT, GRPO, checkpoint evaluation,
fresh-optimizer forks, and optimizer-aware resume:

```bash
python3 -m pip install -e '.[training,modal,tracking]'
python3 -m training_pipeline validate --config examples/training-toy.json
python3 -m training_pipeline smoke --output artifacts/training-smoke
```

See [training pipeline usage](docs/TRAINING_PIPELINE.md) and
[live machinery verification](reports/training-pipeline-smoke.md). The smoke uses
synthetic local tools and a persistent $5 ceiling. Repository training still
requires an eligible release and calibrated verifier.

## Evaluation

Generate an interactive local results report:

```bash
python3 -m eval_pipeline.run --mode demo
```

The demo is explicitly synthetic. For real evaluations, the runner supports a
source-retrieval baseline or importing agent answers, then reference-based LLM
judging. Optional W&B logging stores summary metrics, per-question answer tables,
and report artifacts in offline or online mode.

See [EVALUATION.md](EVALUATION.md) for model configuration, W&B setup, and how to
interpret correctness, completeness, failures, and efficiency.

## Product

**action-trace** is the local repository-research UI: choose a pinned repository and
Tinker checkpoint, ask a question, and inspect live tool activity, source evidence,
and isolated execution output. A clearly labeled scripted preview works without
provider credentials.

See [setup and usage](docs/PRODUCT.md), the [product plan](requirements/PRODUCT_PLAN.md),
and its [editable Excalidraw flow](requirements/diagrams/product-flow.excalidraw).

## Correctness-first evaluator

The stricter `qa_eval` package implements evidence checks, claim-level judging,
structured diagrams, authenticated episode metrics, gated RL rewards, calibration,
and independent checkpoint comparisons. Speed earns credit only for fully accepted
answers. Reference-baseline scores above are not substitutes for these gates.

```bash
python3 -m qa_eval demo --out reports/synthetic-smoke-test.json
python3 -m unittest discover -s tests -v
```

See [the evaluation specification](docs/EVALUATION_SPEC.md), [JSON schemas](schemas/),
and [calibration status](reports/calibration.json). The demo uses synthetic judgments;
human calibration and repository-training experiments remain pending. A real
untrained baseline and a synthetic training-machinery smoke have separate reports.
