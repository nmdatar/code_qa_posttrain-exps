# Bounded remote validation — 2026-09-29 UTC

Final attempt passed in Modal sandbox `sb-wbpzfarJjnTklBDfeWdVFU` (exit 0).
Machine-readable result: [remote-live-validation-v4.json](remote-live-validation-v4.json).

- Agent loop, repository execution, claim extraction orchestration, source evidence
  checks, assessment orchestration, and reward calculation ran on Modal.
- Model calls used Tinker: Qwen/Qwen3.5-4B policy and Qwen/Qwen3.5-397B-A17B judge.
- Supported header-merging answer: resolved score 1.
- Same answer with a false file-deletion assertion: resolved score 0.
- One real repository episode completed with three generations. Its answer lacked
  the required answer/citations, so it received zero reward.
- One disposable importance-sampling update completed on Tinker using an explicitly
  synthetic advantage. Checkpoint save, verification, and sampler loading passed.
  This demonstrates update plumbing, not successful GRPO learning or policy quality.
- Cumulative reservations across all four attempts: $1.566792 of the unchanged
  shared $2 cap. Actual provider billing is not yet reconciled.
- 57 targeted offline tests passed after the final prompt changes.

Earlier failures remain in `remote-live-validation.json`, `remote-live-validation-v2.json`,
and `remote-live-validation-v3.json`: an empty optional symbol; an ambiguous admitted
reference about manifests versus package data; and extraction conflating unsupported
claims with incomplete enumeration. The adapter now normalizes only empty optional
symbols, explicitly distinguishes extraction from truth assessment, and instructs
the assessor to flag unverifiable reference claims for review. Strict completeness,
provenance, and all-claim checks remain enforced. Prompt changes enter reward identity.

The remote execution path is smoke-tested. Broader grader calibration and useful
training signal remain to be established. Full experiments were not relaunched by
this validation, and existing main-thread processes were not stopped or modified.
