# Strong-model assertion validation

Matched pilot: Qwen3.5-4B versus Qwen3.5-397B-A17B, 16 frozen training tasks across 16 repository families, one attempt per model. Same seed, task/rubric release, tools, episode limits, factual-coverage reward, and all-claims-v7 judge. Assertions are unchanged. No training, checkpoint creation, or confirmation-set use.

## Concurrency and trace logging

Each arm uses 16 rollout workers and 8 judge workers (previously 4 and 4). The two model arms run sequentially under the existing benchmark scheduler. All attempts within an arm can overlap. Provider queueing and grading can still limit throughput; configured concurrency is not measured speedup.

Every finished trajectory is stored as full JSON with prompts, generated actions, observations, final submission, token usage, timings, and verification. W&B online tracking publishes the full interactive trajectory table and HTML viewer on run completion. Some individual rollout artifacts are sampled, but the full JSON records and completed interactive table contain every recorded episode.

## Budget and provenance

User authorized the comparison upload and execution after the initial automatic approval block. Fresh isolated Modal controller and volume avoid existing training jobs and their ledgers. Conservative combined reservation estimate: $51.15282432; total allocation cap: $60. Reservations are not invoices; image-build and storage overhead are outside token-ledger metering.

Prepared bundle: `artifacts/strong-model-validation-v1-concurrent-bundle`. Receipt: the adjacent `.submission.json`. The earlier blocked bundle was never launched. Local configs are in `configs/experiments/strong-model-validation-v1/`.

## Verification and interpretation

13 existing concurrency and trajectory-viewer checks passed. The reporting smoke check confirms identical archived inputs yield zero paired difference and unresolved grades remain explicit. Results will distinguish factual assertion coverage from strict source/citation passes, report every task including missing grades, and include a paired task-bootstrap interval. This is a small exploratory pilot using an uncalibrated same-family judge, not independent proof of general capability.

## Completed results

All 32 trajectories were downloaded and checked against the answer tables. No W&B tracking failures occurred. Open `trajectories.html` for all steps from both models, or use either run's W&B `trajectories` table. Full trace JSON and checksums are retained alongside this report.

| Metric | Qwen3.5-4B | Qwen3.5-397B |
|---|---:|---:|
| Mean factual assertion coverage (16 tasks) | 15.625% | 48.661% |
| Full factual coverage | 2/16 | 7/16 |
| Recorded strict passes | 2/16 | 1/16 |
| Unresolved strict grades | 0 | 2 |
| Episode-limit exhaustion | 3 | 0 |
| Rollout + grading elapsed time | 43.0 s | 60.5 s |

Paired factual coverage improved on 7 tasks, tied on 9, and worsened on none. Mean paired difference: +33.0 percentage points; exploratory task-bootstrap 95% interval: +14.3 to +54.9 points. This interval excludes repeated-generation and judge uncertainty. Deterministic empty-answer failures count as zero; two unresolved strict grades remain unknown. No 4-worker timing control was run, so these durations do not establish a measured concurrency speedup.

Recorded reservations including controller allocations total $11.5424, below the $60 cap. These are conservative accounting reservations, not invoiced costs.

## Grading caveats found in traces

- `import-6eb252a9807d12bd2e517e08` (pytest): the 4B answer reverses the meaning of `_early_rewrite_bailout`, yet receives a recorded strict pass. The independent coverage grader assigns zero and identifies the inversion. Pinned `src/_pytest/assertion/rewrite.py:100` returns early when bailout is true; lines 221–225 also confirm the boolean meaning. The stronger answer corrects the inversion and receives full factual coverage. Preserve the original scores: this is evidence the strict grading path needs calibration.
- `import-39feeb31b2d0a61411fad808` (docker-py): the strict judge flags the reference's SSH pool count for review based on the adapter default of 25. However, inspecting the caller shows `docker/api/client.py:140` selects `DEFAULT_NUM_POOLS_SSH` (9) for SSH and passes it into `SSHHTTPAdapter` at lines 171–172. The review flag does not establish a bad reference; the judge missed caller context. The original unresolved strict result is preserved.
- `import-61119dd3437675675599db61` (Django): strict grading remains unresolved over the reference's CSRF-bypass premise and whether the provided evidence establishes it. No source-backed resolution was attempted here.

The pilot supports improved factual coverage from the larger model under identical limits. It does not demonstrate improved strict pass rate or validate every assertion in the dataset.
