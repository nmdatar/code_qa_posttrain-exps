# Isolated parallel campaigns

Use `python -m training_pipeline.remote submit --bundle PATH --isolated` for a fresh independent campaign. It gets a deterministic controller and private Modal volume keyed by the entire immutable bundle ID. Default submissions retain the shared singleton controller and existing storage. No modifications to active remote code, controllers, volumes or ledgers.

An explicit campaign-only budget allocation and empty ledger seeds are required. Historical ledgers, external checkpoint dependencies and specialized workflows are rejected in isolated mode. Aggregate budget approval remains the caller responsibility; isolated volumes do not share a global spending ledger. Each campaign still enforces its own frozen cap. Provider quotas remain shared and may affect throughput. Keep timed throughput measurements separate.

Submit receipts include the volume, controller and isolation flag. Poll helpers read receipt volume, falling back to legacy storage. Download with `python -m training_pipeline.remote download --bundle-id ID --output PATH --isolated`. Preserve this flag for isolated campaigns. Duplicate submission receipts and active same-campaign controller names prevent routine duplicate launches.

Worktree runtime was seeded from the already frozen sweep bundle because main workspace contains uncommitted prerequisites absent from HEAD. Only remote.py is changed inside the new experiment bundle; original bundle and main workspace code remain unchanged. Four arms execute sequentially in their private controller, concurrently with the original teacher/REINFORCE campaign. Tests cover distinct controllers/volumes, active shared controller, duplicate dispatch, fresh-budget enforcement, and existing remote behavior (41 passed).

Live verification: isolated sweep submitted as sb-p5lZbBnhfrgOCTLMGeIgw7. Original controller sb-OuAPccGFF0sGuxUGm4ty96 remained running. Private campaign status was starting. No modifications to original controller.

Sweep complete: see [RESULTS.md](RESULTS.md) and [comparison.json](comparison.json). Provisional best observed setting is LR5e-6/group4. Billing failures invalidate the other group4 learning-rate comparison; all final grades fall below the coverage requirement.
