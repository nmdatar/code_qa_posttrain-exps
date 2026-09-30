# LR=1e-5, G=4: clean rerun complete

Final eval score: **18.75% (6/32 strict passes)**, up from 15.625% (5/32) initially.
This ties the original LR=1e-5, G=8 final score. It does not establish G=4 as a
quality winner, and it remains below the 31.25% observed at LR=5e-6, G=4.

| Setting | Final eval score | Strict passes | Graded |
|---|---:|---:|---:|
| 1e-5, G=4 — clean rerun | 18.75% | 6/32 | 29/32 |
| 1e-5, G=8 — original sweep | 18.75% | 6/32 | 28/32 |
| 5e-6, G=4 — original sweep | 31.25% | 10/32 | 29/32 |
| 5e-6, G=8 — original sweep | 12.50% | 4/32 | 27/32 |

The new run completed all 16 batches and 128 training attempts with 13 acknowledged
optimizer updates, 128 resolved training rewards, no excluded groups and **zero
infrastructure failures**. Training episode terminations were 101 completed and
27 budget-exhausted; this is the episode budget, not an account billing failure.
Initial/final eval used the same 32 selection tasks. Final answers completed on
31/32 tasks. Three final grades remain unresolved, so 90.625% scoring coverage
misses the 95% requirement. This is a single-seed selection comparison, not a
confirmed optimum or independent human evaluation. No confirmation tasks ran.

The original interrupted 1e-5/G4 result (15.625%, 5/32, two updates, 104 infrastructure
failures) remains preserved; the new run has its own ID, controller, volume and
ledger. All scientific config fields and frozen runtime code match the original
bundle. Only run/output/ledger identity, tracking metadata and campaign allocation
changed. There was no shell-harness intervention in this rerun.

Total conservative reservations: **$62.48**, below the approved $320 ceiling.
Actual provider billing is unavailable. The controller exited successfully and
1,452 remote artifact files were downloaded. archive-checksums.json covers local
copies plus the collector status file. Checkpoint manifests retain provider
references; this does not claim permanent local weight/optimizer archiving.

## Links

- [W&B rerun](https://wandb.ai/nmdatar-harvard-university/repository-qa-training/runs/lr-1e-5-group-4-v6-rerun-v1-seed42)
- [Presentation with updated learning-rate/group-size charts](https://app.excalidraw.com/s/8Ufs2ZMhWhu/3Yoeeh0Fn8H)
- summary.json: exact scores, task IDs, updates and ledger reservations.
- preparation.json: approval, exact frozen bundle identity and submission receipt.
- ../../artifacts/lr-1e-5-g4-rerun-v1-results/: raw trajectories, grades, logs,
  checkpoint manifests and ledger.

The presentation compares learning rates at fixed G=8, then group sizes at fixed
LR=1e-5, explicitly labeling G=4 as the clean rerun. It preserves the full-sweep
highest observation and unresolved-grade counts without altering measured scores.
