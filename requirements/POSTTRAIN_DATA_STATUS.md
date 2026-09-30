# Post-training data readiness

The published collection now contains **100 newly authored training tasks across 10 new repository families**, plus a frozen **20-task development cohort drawn from the existing 992 tasks**. The cohort adds strict private rubrics; it does not add 20 new task identities. There are 1,092 unique tasks across training and development.

| Release | Tasks | Families | Environments | Verified artifact hashes | Human approved |
|---|---:|---:|---:|---:|---:|
| `repo-qa-1000-v1` original development collection | 992 | 41 | 44 | 305 | 0 |
| `repo-qa-training-posttrain-v1` | 100 | 10 | 10 | 74 | 0 |
| `repo-qa-development-posttrain-v1` cohort | 20 | 10 | 10 | 63 | 0 |

`reports/posttrain/final-data-release-audit.json` verifies every packaged hash, exact public/private task bindings, independent machine-review rubric digests, environment bindings, and split separation. All original 992 tasks remain unchanged and development-only. Every cohort public record matches its original record. Training task IDs and families are disjoint from all 41 original development families.

## Admission and quality limits

Both new manifests explicitly say `diagnostic_only: true` and `human_admitted_tasks: 0`. All 120 private task records retain `gold_status: draft` and `human_reviewed: false`. The current inventory's automated review policy recognizes the 100/20 independent machine-reviewed records; this is not human approval or measured reward calibration. These releases support bounded integration experiments. Do not interpret automated admission as satisfying the agreed human calibration gate for longer experiments.

All 100 fresh questions have authored claim rubrics, exact pinned source spans, private reference answers, and a separate agent's source review. Ten fresh Modal source-reading environments passed readiness, isolation, source attestation, and tool replay; the development cohort reuses ten previously checked environments. The release audit checks recorded evidence, not a new live run. Source-only environments provide stateless listing, searching and reading; they do not claim executable dependency/fixture readiness or persistent cross-call state.

The ten training families are go-task/task, open-telemetry/opentelemetry-python, huggingface/datasets, nushell/nushell, pnpm/pnpm, jqlang/jq, biomejs/biome, astral-sh/ruff, vitest-dev/vitest, and zed-industries/zed, with ten tasks each. Activity and collection timestamps, pinned commits, GitHub fork/parent metadata, and known family separation review are preserved in release attribution. This does not establish absence of every shared third-party utility. Pinned licenses are included; Zed settings and jq documentation/source have distinct licensing details and are not represented as uniformly permissive.

Questions are source-grounded, but concentrated in selected modules and behaviors. Difficulty, coverage representativeness, solver success rates and grader accuracy remain empirical gaps. Independent machine review is not independent human judgment. This work did not manufacture blind solver trajectories from reference answers; the question/rubric release alone is not an SFT trajectory dataset.

The original development package contains 12 strict verifier-compatible draft rubrics, 975 additional reviewed reference records lacking the full strict rubric contract, and five quarantined references. The new 20-task cohort adds explicit strict rubrics for selected imported tasks without mutating that package. The five unresolved references remain outside scored releases.

## Calibration

Forty clearly labeled constructed cases cover 20 development tasks: a reference-based supported candidate and an explicitly contradicted required claim per task. They are not model-generated solver outputs. The original live Qwen evaluation-judge pass is preserved in `reports/posttrain/calibration/scored-v1`, with one accepted and 39 unresolved judgments and a private human-review packet. No human decisions have been supplied.

A captured response showed the original prompt conflated assertion enumeration with factual verification: it enumerated both claims but returned incomplete extraction and abstention. Prompt version 1.1 separates these stages, namespaces extracted claims separately from rubric IDs, and retains every strict evidence/assessment check. A bounded follow-up produced a valid partial judgment; it does not establish calibration. Raw diagnostic requests/responses are private in `reports/posttrain/calibration/extraction-diagnostic-*`. Full packet retries are not implied.

The unchanged statistical gate requires two independent agreeing human reviews, at least 100 unique tasks and ten families, no unresolved/disputed items or correlated variants within the measured strata, and 95% Wilson-bound thresholds: material-error detection at least 95%, false acceptance at most 2%, false rejection at most 5%. Forty variants of 20 tasks cannot satisfy the task-count requirement. More diverse cases and genuine reviewer decisions are required before claiming calibration.

Live judge and environment costs use the shared spending ledger at `artifacts/posttrain/spending.json`; conservative reservations are not provider-invoice measurements. No zero-cost or completed-human-review claim is made.

## Interfaces

`posttrain.data.inventory_release`, `adapt_task`, `load_tasks`, and `validate_artifacts` provide inventory, exact binding, strict task loading, and artifact verification. Missing rubric fields are not fabricated. Specify and report the review policy when interpreting readiness.

`posttrain.calibration.export_packet` emits a private HTML review and decision template with predictions hidden. `import_reviews` merges genuine incremental reviewer decisions, normalizes identities, rejects conflicting replacements, and preserves the original statistical gate. Recorded reviewer attestations are not identity authentication.

`python -m posttrain.calibration_scoring` grades constructed cases through the private verifier using bounded shared-ledger requests. Its output has no fabricated rollout telemetry or training reward; unresolved responses remain unresolved. The generated packet is review input, not proof of quality.
