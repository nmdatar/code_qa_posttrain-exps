# Current experiment version migration

Updated all six execution procedures and the broader roadmap to point to the current v7 citation-fixed contract. Added 16 version-aligned execution configs and planning metadata for procedures awaiting implementation. Historical configs/results and active submitted config remain unchanged.

Validation: all 16 configs passed schema, dataset and cohort validation; all eight throughput configs passed regenerated task-manifest validation. Confirmed experiment 2 is an exact copy of its submitted source, LR arms match outside their intended rate/identity/budget differences, and run IDs are unique within the new set. Relevant existing config, readiness and remote-execution tests: **31 passed**.

Budget: matched training-arm operation estimate $477.46; full experiment-1 schedule $1,072.74. These use recorded price snapshots and exclude controller costs and existing ledger commitments. No caps were changed and no new paid runs were submitted. The $80 LR allocation cannot fund the matched arm; the $300 baseline allocation cannot fund the full rerun schedule.

See [current configs](../../configs/experiments/current-v7/README.md), [validation and estimates](validation.json), and [config/source checksums](checksums.json). Offline checks do not claim live provider readiness or experimentally demonstrated quality improvement.
