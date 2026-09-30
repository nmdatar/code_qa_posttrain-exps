# Longer GRPO completed: no demonstrated held-out improvement

Run `grpo-long-v6-seed42-v1` exited successfully. W&B recorded runtime **24 minutes 29 seconds**, within the estimated 20–40 minutes. All 16 batches / 128 training attempts completed, with 128 resolved rewards and **11 acknowledged optimizer updates**. Confirmation was untouched.

Latest frozen v6 repaired training rubrics, all-claims-v7, positive-coverage-v4, definition-context-v1, action-alias-v1 and paginate-v1 were used with fresh Qwen3.5-4B, rank 8, seed 42, LR 1e-5.

| Selection metric | Initial | Final |
|---|---:|---:|
| Strict full passes | 8/32 | 6/32 |
| Demonstrated strict credit | 25.0% | 18.75% |
| Resolved grades | 28/32 | 31/32 |
| Completed answers | 32/32 | 31/32 |

Demonstrated-credit delta: −6.25 percentage points; paired task bootstrap 95% interval [−21.875, +9.375] points. Unresolved grades contribute no demonstrated credit; they are not relabeled incorrect. This small single-seed selection comparison does not establish improvement or reliably quantify degradation, and the interval does not incorporate repeated generation/judging uncertainty.

Mean training reward was 0.212240; final EMA 0.114029. Seventeen of 32 groups had zero within-question reward variance; 60 trajectories contributed to updates. No groups were excluded for unresolved grading. Training rewards across changing tasks do not establish generalization.

Final checkpoint: `ckpt-c87047fe8c894be2aa152e458d38867d`. Final evaluation meets the 95% coverage eligibility rule; initial evaluation does not. Its automatic best-eligible designation does **not** mean it outperformed the initial policy. Remote retention is 14 days; the sampler archive was downloaded and its SHA-256 verified. The archive does not establish support for reimporting weights or a permanent optimizer-state export.

Incremental ledger reservations: **$68.4910**, versus the conservative $347.0109 bound. Cumulative GRPO ledger: **$362.5695 / $650**. These are reservations, not provider invoices. Project ceiling remains $1000.

Archived and JSON-verified **1467 files**, including campaign outputs, traces, checkpoints, sampler archive and authoritative ledgers. Checksums are in checksums.json; raw archive: ../../artifacts/grpo-long-v6-results/.

[W&B run](https://wandb.ai/nmdatar-harvard-university/repository-qa-training/runs/grpo-long-v6-seed42-v1) · [Machine-readable results](results.json) · [Paired comparison](paired-comparison.json) · [Archive](archive.json)
