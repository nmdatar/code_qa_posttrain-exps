# Recovered teacher collection: complete coverage, SFT gate not met

The merged collection contains exactly **1,200 original attempt slots on 600 distinct training tasks**: 955 preserved original outcomes and 245 recovery outcomes. All 103 replaced infrastructure failures are excluded from the merged outcome set but retained in provenance. Archive input hashes and every recovery episode's frozen slot identity were verified; no duplicate successes or replacement of semantic failures was allowed.

| Outcome | Count |
|---|---:|
| Completed answers | 1,041 |
| Budget-exhausted attempts | 157 |
| Preserved infrastructure-labelled final answers awaiting grading | 2 |
| Resolved grades | 1,178 / 1,200 (98.17%) |
| Unresolved grades | 22 |
| Strict-passing trajectories | 188 |
| Distinct tasks with a strict-passing trajectory | 148 |
| Perfect factual training rewards | 486 |

Unresolved outcomes remain unresolved. The 22 include 20 completed answers and two preserved infrastructure-labelled final answers. Their existing generations were not resampled. These are teacher collection outcomes, not a student evaluation or evidence that training improved the student.

Under the **unchanged** source/native-token/complete-investigation admission rules, 44 trajectories were admitted, collapsing to **34 unique training lineages across 15 repository families**. There are 8,599 supervised target tokens. Of the 188 strict passes, 74 failed the complete clean tool-sequence requirement and 70 had no useful verified prefix. Perfect factual reward alone did not qualify any demonstration.

The preregistered SFT readiness gate requires **200 distinct lineages across 15 families**. Family coverage now meets the minimum, but the lineage count does not: **the gate fails**. The frozen 34-example release is an audit artifact, not a substitute for the promised adequately sized study. Even admitting every strict-passing task without the existing tool checks would yield only 148 distinct tasks, below the gate. No SFT training or other paid action was launched during this audit.

Machine-readable merged outcomes, all 1,200 slot-to-file hashes, excluded predecessor episode IDs and admission details are in `teacher-merged-audit.json`. Frozen complete-investigation manifest and selected sources are in `data/sft/complete-investigations-teacher-merged-v1/`; manifest SHA-256 is `1a028d93f7e905879a66269222cbfd2f75b44e4ca1ac1d89d95d3f9570a226d3`.

Validation: 12 focused recovery and SFT-audit tests passed, including exact union, duplicate/missing recovery rejection and changed-source-hash rejection. Confirmation data remained untouched.
