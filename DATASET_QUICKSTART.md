# Repository Q&A dataset quickstart

## Selection

Use [TIGER-Lab/SWE-QA-Pro-Bench](https://huggingface.co/datasets/TIGER-Lab/SWE-QA-Pro-Bench)
for the first pipeline. The downloaded release has 260 examples, ten per repository,
with `repo`, `commit_id`, `question`, `answer`, `cluster`, and `qa_type` fields.
The [dataset card](https://huggingface.co/datasets/TIGER-Lab/SWE-QA-Pro-Bench/blob/596892dac60b6f500f01a7dc2becb9f66593b7b7/README.md)
declares MIT; retain its attribution and the source repositories' own notices.

This is a direct fit for answering questions about an unfamiliar codebase.
HumanEval/MBPP primarily measure code generation, while SWE-bench measures patches.
[RepoProbe](https://github.com/Tencent-Hunyuan/RepoProbe) is a useful later second
suite with checklist-based judging; introducing two datasets now would add setup
work before the first run is working.

## Run

From the project root:

```bash
# Three questions, one small repository, pinned source checkout.
python3 scripts/prepare_dataset.py --checkout

# Validate preparation contracts.
python3 -m unittest discover -s tests -v

# Prepare all questions in a separate output directory (without cloning 26 repos).
python3 scripts/prepare_dataset.py --all --output data/swe-qa-pro-full
```

Python 3.11+ and Git are sufficient. Initial preparation needs network access to
Hugging Face, and `--checkout` needs GitHub. Cached preparation works offline.
The script does not install or execute benchmark repository code.
Use `--repo owner/name --limit N` to choose another small slice. Add `--checkout`
to the full command only when you need all repository snapshots.

The pinned dataset revision is `596892dac60b6f500f01a7dc2becb9f66593b7b7`.
The script verifies the source SHA-256 before parsing and rejects unexpected
counts, malformed commits, missing fields, and duplicate questions. Existing
checkouts with local changes are rejected rather than overwritten.

## Outputs and agent contract

| Artifact | Purpose |
| --- | --- |
| `tasks.jsonl` | Agent inputs: stable ID, repository, commit, question |
| `references.jsonl` | Evaluator-only reference answers and category annotations |
| `upstream.jsonl` | Selected rows with the original upstream schema |
| `source.jsonl` | Verified full upstream download; evaluator-only |
| `DATASET_CARD.md` | Upstream attribution and methodology |
| `manifest.json` | Dataset revision, checksum, selected IDs, verified checkout paths |

Pass only task inputs and the corresponding repository snapshot to the agent.
The source and reference files contain gold answers: file separation alone is
not access isolation. A tool-using agent should receive only the repository in its
accessible filesystem, with the evaluation data kept outside that environment.
Do not give it unrestricted access to this project directory.

A future answer adapter should emit one JSONL result per task with `id`, `answer`,
`model`, `status`, elapsed time, tool-call count, and total token usage. Missing
answers and tool/provider failures must remain visible in the run report. The
current implementation ends at dataset preparation; it includes no model client
or answer judge.

## Live model evaluation

The [official harness instructions](https://github.com/TIGER-AI-Lab/SWE-QA-Pro/blob/main/eval/README.md)
provide direct-answer and tool-agent runners plus an LLM judge. Its score totals
five subjective dimensions on a 5–50 scale; it is not executable code pass@1.
Choose an answer model/endpoint and a fixed judge before comparing models.
Keep the same task IDs, repository versions, tools, budgets, and judge for each run.

Source inspection found integration details to handle before using it here:

- The upstream CLI loads its own floating dataset and has no local input-file or
  dataset-revision option. `upstream.jsonl` preserves its schema, but using our
  selected rows requires adapting its loader; it is not currently a drop-in CLI input.
- The agent runner does not validate repository commits. Use our manifest and
  verified checkouts when wiring up an adapter.
- The tools do not enforce repository-only filesystem access. Run in isolation
  with gold answers outside the agent environment.
- Use fresh output files; the agent runner appends even with `--no-resume`.
- Require the expected number of successful answers and scored rows. Do not let
  skipped judge rows or agent errors silently improve the reported average.
- The requirements include vLLM, which complicates a native macOS hosted-API run.
  Prefer a suitable Linux environment for the complete upstream stack.

These findings are based on the upstream source reviewed on September 28, 2026;
pin the harness revision when implementing the model adapter.

Use this three-question slice for plumbing only. Reserve the remainder for broader
evaluation, keep this benchmark out of training, and manually inspect judge
decisions before interpreting small score differences.

## Verified locally

On September 28, 2026, preparation validated all 260 records across 26 repositories,
exported three selected questions, and fetched `getsentry/responses` at
`3e029949c8f3c1b3b2b48f9e437fedd7cf27746d` (25 tracked files).
All three preparation contract tests passed. No live answer-model or judge call
has been run; these results establish dataset readiness, not model performance.
