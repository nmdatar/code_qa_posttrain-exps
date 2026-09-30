# Automated rubric admission

The user replaced the earlier mandatory human-annotation requirement with automated rubric review. This policy supersedes the human calibration prerequisite in the earlier post-training plan and status documents.

New post-training configurations default to `review_policy: "automated"`. Human annotation and human reward calibration are optional. To retain the old stricter policy for a particular experiment, explicitly select `review_policy: "human"`.

## Required automated checks

- Load tasks, private rubrics, and independent review records from hash-verified release artifacts.
- Bind each supported review to the exact task and complete rubric hash.
- Require distinct nonempty author and reviewer identities; reject unsupported, disputed, stale, or missing reviews.
- Keep private grading outside policy context and verify source evidence, citations, answer completeness, and critical errors during scoring.
- Preserve task-family split separation, environment requirements, unresolved-judgment exclusion, and existing budget controls.

Human fields remain false where no human reviewed the records. Automated admission is not evidence of human approval or measured grader accuracy. Optional calibration tools retain their original meaning; their statistical human-review gate is not required by automated-mode runs.

## Existing data and runs

The new runtime admission policy accepts the 100-task training release and 20-task development release on their existing manifest-bound independent automated reviews. Frozen release metadata and historical audit reports are not rewritten; their `diagnostic_only` and human-admission counts describe the policy at publication. See `reports/posttrain/automated-review-admission.json` for the current policy's read-only re-audit.

Diagnostic limits are separate from review policy. For a longer experiment, set `diagnostic: false` and choose explicit update counts, evaluation cadence, and a budget. The $20 cap is unchanged. This policy change starts no model runs and spends nothing. Existing checkpoints preserve their historical configuration; changing review policy is an experiment change and should use a new run/fork rather than silently reinterpreting a running experiment.

## Verification

Seventy targeted tests passed, including real-release admission, longer-run readiness without human calibration, normal automated scoring without changing human labels, and rejection of tampered or non-independent reviews. End-to-end live RL validation remains a separate requirement and is not implied by admission.
