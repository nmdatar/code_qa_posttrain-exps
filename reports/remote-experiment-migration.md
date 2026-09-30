# Remote experiment migration verification

Implemented the detached Modal controller sandbox launch path, cloud-only collection execution guard, full two-stage semantic claim verifier, and Qwen/Qwen3.5-397B-A17B judge pins.

Validation: the full offline suite passed 414 tests before the final controller-local lock adjustment; targeted remote/concurrency/training checks passed after that adjustment (see test modules). Staging tests use a real temporary Git repository and verify exact source objects survive packaging. Dispatch/orchestration tests mock provider calls; no live cloud success is claimed.

The controller uses a single named sandbox and a Modal v2 persistent volume. Process locks live on the controller's local filesystem, avoiding reliance on distributed file locks. Every paid-call reservation is synced before dispatch. The laptop only stages/submits/retrieves. No cloud deployment, paid grading, or experiment launch was performed; existing processes were not stopped.

Initial four-arm conservative reservation: $166.01, before existing ledger balances and additional image-build/volume-storage costs. Existing caps are unchanged. Small screens now permit four batches/updates; full direct GRPO permits eighteen; older LR screens permit six batches/five updates. The new grader's results are not directly comparable with prior source-claims-v2 results.

Before restart, follow docs/REMOTE_EXPERIMENTS.md: reconcile/stop old jobs, provision the named credential secret, prepare a fresh immutable bundle, and submit. A live remote smoke and diagnostics for the new two-stage adapter remain unverified. Prior Qwen judge diagnostics are not validation of this new adapter. Reference admission remains automated, and training/selection use the same judge (not independent evaluation).
