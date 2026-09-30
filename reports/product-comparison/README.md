# Trace model comparison verification

Implemented Single/Compare navigation, two independent workers and SSE streams,
persisted comparison URLs and history, per-side and combined cancellation,
isolated follow-up context, tool/citation inspection, live metrics and token cost
estimates. Hosted model discovery includes Qwen3.5-397B-A17B; raw pretraining and
unsupported renderer families are disabled. Public pricing comes from Tinker's
model metadata, with optional server-side overrides and per-run rate snapshots.

## Demo

Server: http://127.0.0.1:8000/?mode=compare

Saved real comparison:
http://127.0.0.1:8000/?comparison=6850fa58b4974b9e865617955776752b

Repository: octocat/Hello-World at pinned commit 7fd1a60.
Question: What message is in this repository’s README?

| Model | Outcome | Run time | Tools | Input tokens | Output tokens | Estimated token cost |
|---|---|---:|---:|---:|---:|---:|
| Qwen3.5-4B | Completed, cited answer | 13.2 s | 2 | 6,179 | 189 | $0.00223 |
| Qwen3.5-397B-A17B | Completed, cited answer | 11.9 s | 3 | 4,889 | 131 | $0.01565 |

Both runs launched concurrently. This small smoke test verifies integration;
it does not establish a general speed or answer-quality ranking. Times are
recorded harness run times; catalog discovery and browser loading add overhead.

A no-cost scripted demo is available from the comparison landing page.
Restart with `./scripts/demo-product.sh` from the repository root. This is a local,
loopback-only demo, not a public hosted deployment.

## Validation

- Frontend TypeScript and production build passed.
- Existing product tests: 19 passed.
- Comparison tests: 7 passed, including simultaneous workers, isolated follow-ups,
  one-sided stop and launch failure, event replay, persisted history, interrupted
  restart recovery, 397B discovery, accounting and invalid rates.
- Harness regression discovery: 3 passed.
- Browser: real model selection/submission, independent timelines and final
  metrics, pinned citation inspection, saved live comparison restoration.
- Browser console errors: none observed on the live comparison.

![Live comparison](live-comparison.png)


## Assertion grading demo

Live comparison: http://127.0.0.1:8000/?comparison=72fcc92d8c8d4fd5b0e53a82d64539f7

Both models answered `pydantic-frozen-copy-8960` against the same pinned repository
and six-assertion draft reference. The blind, temperature-zero 397B grader returned
validated assertion judgments; application code computed weighted coverage.

| Model | Coverage | Duration | Tools | Estimated answering cost |
| --- | ---: | ---: | ---: | ---: |
| Qwen3.5-4B | 50/100 | 139.5s | 37 | $0.16923 |
| Qwen3.5-397B-A17B | 100/100 | 139.1s | 12 | $0.31401 |

Separate estimated grading token cost: $0.05643. References are draft and not
human-reviewed. These scores measure this rubric's assertion coverage, not general
model quality or an exhaustive factual audit. The 4B answer incorrectly said
same-class instance validation rejects the invalid copy by default; the table
shows the contradiction and the exact answer passages.

Verified in the browser: automatic grading state transitions, both numeric scores,
per-assertion verdicts and credit, source inspector links, live reload persistence,
reference-status disclosure, and no console errors. Frontend production build
passed. Product/claim checks passed (37 tests), followed by grading/limits/claim
checks (19 tests, including new supervisor and restart cases).

![Assertion scores](assertion-scores.png)
