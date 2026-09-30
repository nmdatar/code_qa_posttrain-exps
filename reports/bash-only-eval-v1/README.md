# Bash-only versus structured-tool validation — completed

The current structured harness outperformed unrestricted bash-only for the untrained Qwen3.5-4B model on the same 32 validation tasks. Bash-only had 0 full strict passes versus 9 for structured tools. This measures the implemented prompts, action protocol, and five-tool-call budget together; it does not establish that bash is inherently worse or predict the result after GRPO/tool-use training.

## Results

| Metric | Current structured harness | Bash only, no command/path allowlist |
|---|---:|---:|
| Tasks attempted | 32 | 32 |
| Full strict passes | 9/32 (28.1%) | 0/32 (0%) |
| Unresolved strict grades | 1 | 0 |
| Mean strict score, resolved tasks only | 0.3118 (31 tasks) | 0.0000 (32 tasks) |
| Observed strict credit / all scheduled tasks | 0.3021 | 0.0000 |
| Completed answer submissions | 31 | 32 |
| Tasks with no tool call | 0 | 13 |
| Mean tool calls | 4.19 | 2.94 |
| Policy input tokens | 367,584 | 193,596 |
| Policy output tokens | 7,768 | 7,815 |
| Rollout + grading wall time | 64.7 s | 50.8 s |
| Reserved cost including controller | $6.54 | $4.89 |

The all-task credit metric records observed credit; it does **not** relabel the unresolved grade as a known failure. Across 31 paired resolved tasks, bash worsened on 10, tied on 21, and improved on none. Its mean strict-score difference was −0.3118 (exploratory paired task-bootstrap 95% interval −0.4731 to −0.1613). The full-pass rate difference over all scheduled tasks was −28.1 percentage points.

Total recorded reservation: **$11.4344**, within the approved $100 combined cap and below the $65.30 conservative estimate. Reservations are not invoices; unmetered image-build/storage overhead is excluded. Both arms ran concurrently; timings are one-run observations rather than controlled speed benchmarks. Bash's lower cost/time came alongside frequent early answers without tool use.

## What the traces show

- All 94 executed tool calls in the bash arm were `bash`. The structured arm executed 70 searches, 56 file reads, and 8 listings.
- Bash submitted uncited answers without any tool call on 13 tasks. Structured tools were used on every task.
- Of 19 bash episodes that used tools, 17 started with a broad `find ... | head -20` listing and two with `ls`. All 32 structured episodes started with `search_code`. Eighteen bash episodes used the full five-call budget, frequently spending calls locating files before reading source.
- Bash recorded 7 invalid-action observations (five multiple-JSON parsing errors and two citation-path errors) and 6 nonzero shell exits. Nonzero exits included grep misses, so these are not all infrastructure failures. No episode in either arm terminated with infrastructure error.
- Bash's 32 zero scores comprise 13 missing-answer/citation failures, 2 citation validity/size failures, 8 material-error judgments, and 9 other strict content/citation failures. Some answers contained correct facts but still failed strict evaluation.
- The existing strict audit's secondary supported-claim coverage averaged 60.8% on 27 eligible structured episodes and 52.5% on 17 eligible bash episodes. These unequal eligible subsets are **not** a matched factual-accuracy comparison. The independent training factual-coverage grader was not run on eval tasks.

## Verification and artifacts

Verified 597 downloaded files against SHA-256 hashes, all 64 traces against the saved answer tables, identical task IDs and public task inputs (apart from permitted tools), identical model/policy, limits, seed, judge, and concurrency. The 48 relevant harness/collection/pipeline tests passed before launch.

- [Interactive trajectories](trajectories.html): all prompts, actions, outputs, submissions, and grades, labeled by arm.
- [Summary](summary.json), [paired scores](paired-results.csv), [per-task details](task-details.json).
- [Verification](verification.json), [archive checksums](archive-checksums.json).
- Configs: `configs/experiments/bash-only-eval-v1/`.
- Downloaded complete run artifacts: `artifacts/bash-only-eval-v1-results/`.
- Approved immutable bundle: `artifacts/bash-only-eval-v1-funded-bundle`.

The initial automatic upload block was resolved by explicit user approval. Both remote jobs exited successfully. W&B remained offline; the full local archive is available.

## Design and limitations

Two fresh frozen-policy Qwen3.5-4B evaluations on the same 32 selection tasks from the larger-model study's dataset. The larger-model score band selected 274 **training** tasks; these evaluation tasks are separate, family-disjoint and were not filtered by strong-model scores. The 85-task confirmation set is not used.

Both arms use temperature 0, seed 42, 16 rollout workers, 8 judge workers, the same pinned source/assertions, all-claims-v7 strict grader, and limits of 6 generations, 5 tool calls, 512 tokens per call, 3,072 output tokens, 3,500 output bytes per observation, and 300 seconds per episode. No training or checkpoint updates. Full local trajectories are retained; W&B is offline.

Structured arm retains list_files, search_code and read_file. Bash arm exposes only bash(command) and passes the command verbatim to bash -lc, without command/path allowlists. Existing isolated, network-blocked, credential-free Modal repository sandboxes and resource limits are retained. The reused images are source-reading environments: their source is read-only and grading uses the pristine snapshot. Bash has no command/path allowlist, but OS permissions and network blocking remain. No repository dependencies or additional command packages were installed for this experiment.

Citation adapter difference: structured compact citations require a file previously read through read_file. Bash output is unstructured, so compact citations bind to a path in the original source inventory without a tracked-read requirement. The exact same source/hash/range and semantic citation checks run afterward. Bash does not pretend that the entire catalog was observed or feed it to the judge. This is a comparison of usable harnesses, not a perfectly isolated change of tool names. Shell prompt explains commands and numbered source reads instead of structured-tool syntax.

The primary metric is existing strict eval reward/pass rate, with unresolved grades reported separately. Factual claim support in the strict semantic audit will be reported separately where available; the independent factual-coverage training reward is not run on development tasks. Single greedy rollout per task does not measure repeated-run variability. The judge remains uncalibrated and in the same model family.

Conservative reservation estimate: $65.30 total including controller reservations, with $50 maximum ledger reservation per arm. These are not invoices and exclude image-build/storage overhead. Runs use a fresh isolated controller and volume.


## Environment audit correction

The seven selected environment recipes use `python:3.12-slim` and all record empty `install_commands`. This was a shell-based source-reading comparison, not a fully provisioned development/test environment. Six manifests explicitly record read-only source; the seventh is a legacy source manifest without runtime-policy metadata. The standard image builder removes source write permissions and runs as UID 65534. All 94 shell observations had empty stderr; 88 exited zero and 6 were grep misses with exit 1. No command-not-found or permission-denied error was observed. The commands exercised were bash, find, head, xargs, grep, sed, ls, and cat; availability of unexercised utilities and repository test dependencies was not verified. The earlier description suggesting a writable source workspace was incorrect.
