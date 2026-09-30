# action-trace: repository QA experiments

This project studies how post-training, tool interfaces, and efficiency rewards
change a model's ability to answer questions about a pinned source repository.
It includes dataset preparation, evidence-based evaluation, bounded SFT/GRPO/
REINFORCE training, and a local app for comparing investigations.

## Start here

- **Understand the research:** [experiment progression](experiments/README.md) — questions, findings, limitations, and reproducible inputs.
- **Try the app:** [product setup](docs/PRODUCT.md) — choose a repository, checkpoint, and tool harness; inspect answers and evidence side by side.
- **Run or extend a study:** [configuration index](configs/experiments/README.md), [training guide](docs/TRAINING_PIPELINE.md), and [remote execution](docs/REMOTE_EXPERIMENTS.md).
- **Understand the measurements:** [evaluation guide](EVALUATION.md), [reward shaping](docs/REWARD_SHAPING.md), and [decision evidence](reports/experiment-decision-evidence/README.md).

Results are exploratory. Dataset, judge, reward, and harness versions differ
between studies; scores across versions are not automatically comparable.
Unresolved grades are reported separately. A prepared config is not evidence
that an experiment ran, and saved checkpoint metadata does not guarantee that
remote weights are still available.

## Local setup and offline checks

Python 3.11+ and Git are required. From the repository root:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m eval_pipeline.run --mode demo
```

The evaluation demo is synthetic. See the [dataset quickstart](DATASET_QUICKSTART.md)
for pinned source preparation and [product setup](docs/PRODUCT.md) for the app's
additional Python/Node dependencies and scripted preview. Live model runs require
provider credentials and explicit experiment budgets; the commands above do not
launch training.

## Repository map

| Area | Purpose |
|---|---|
| `dataset_builder`, `data` | Task generation, release manifests, source and split provenance |
| `agent_harness` | Repository tools, execution limits, Modal environments |
| `qa_eval`, `eval_pipeline`, `training_eval` | Grading, evaluation contracts, reports and checkpoint evaluation |
| `training_pipeline`, `posttrain` | Training strategies, data collection, budgets, checkpoints and remote orchestration |
| `product_api`, `product_web` | Local research and comparison app |
| `configs/experiments` | Frozen study inputs and versioned suites |
| `reports` | Study summaries and supporting evidence; enter through the research index |
| `scripts`, `tests` | Study utilities and regression coverage |

## Historical experiments

Early readiness screens, reward diagnostics, and superseded pilots have been
retired from the main checkout. Their conclusions and exact Git-history links
are in the [historical index](docs/EXPERIMENT_HISTORY.md). Informative negative
studies, shared implementations, and frozen inputs needed by surviving studies
remain. Existing experiment branches preserve the incremental development history.
