# Baseline pipeline verification

[Finished W&B run](https://wandb.ai/nmdatar-harvard-university/repository-qa-eval/runs/qfedsspa)

Experiment: `baseline-qwen35-4b-generated-dev-v2`. Solver: Qwen3.5-4B. Judge: Qwen3.5-397B-A17B. Both use Tinker native sampling, with no training updates.

**12/12 tasks answered and graded; zero runner failures; 144 tool calls.** Mean weighted reference-claim coverage: **30.83%**. One task had all reference claims supported. These are automatic development-reference assessments, not verified accuracy or calibrated RL rewards.

V2 caps investigation at 12 calls and explicitly requests a final answer with reserved tokens. All 12 tasks used that final-answer phase. Model-directed probes are not guaranteed: successful source-tool execution alone satisfies the pipeline tool check. See each trajectory and judge execution-support field before interpreting runtime claims.

| Task | Weighted claim coverage | Tool calls | Seconds |
|---|---:|---:|---:|
| pydantic-frozen-copy-8960 | 66.7% | 12 | 56.7 |
| tanstack-hydration-fetch-freshness-8936 | 100.0% | 12 | 56.7 |
| sqlalchemy-filtered-collection-7654 | 50.0% | 12 | 55.8 |
| pydantic-typed-extras-runtime-policy | 30.0% | 12 | 58.1 |
| pydantic-factory-data-signature-order | 33.3% | 12 | 54.5 |
| pydantic-alias-choice-unused-inputs | 8.3% | 12 | 62.1 |
| tanstack-signal-consumption-unsubscribe | 10.0% | 12 | 52.5 |
| tanstack-manual-cache-edit-cancel-revert | 10.0% | 12 | 51.8 |
| tanstack-inflight-options-retry-capture | 0.0% | 12 | 49.8 |
| sqlalchemy-savepoint-preflush-state | 20.0% | 12 | 52.5 |
| sqlalchemy-json-joinedload-uniquing | 0.0% | 12 | 58.2 |
| sqlalchemy-none-default-json-matrix | 41.7% | 12 | 56.7 |

Usage: 1,233,900 solver input / 21,606 solver output tokens; 267,225 judge input / 7,069 judge output tokens. An uncached token-price estimate for **this v2 run only** is $1.28; actual Tinker billing may be lower from caching. Excludes Modal and earlier debugging runs. Rates: [Tinker pricing](https://tinker-docs.thinkingmachines.ai/tinker/models/).

Local result directory: `/Users/ndatar/Documents/ChatGPT/action-interview/artifacts/runs/baseline-qwen35-4b-generated-dev-v2-20260928T223738Z-41e06b`. It contains the exact experiment config, runner source and hashes, model/template identities, every trajectory, answers, grader responses, usage and summary. W&B has the result table and evaluation artifact.

Validation: 121 local tests passed; the saved runner/config hashes match their recorded identities, and all 12 result IDs are unique.

The original [v1 full run](https://wandb.ai/nmdatar-harvard-university/repository-qa-eval/runs/13yhygmt) produced no final answers. It is retained as a failed integration baseline. V2 fixes terminal handling of recoverable tool errors and reserves answer synthesis; compare it as a different harness configuration.
