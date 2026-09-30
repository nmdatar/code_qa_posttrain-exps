# Isolated REINFORCE replacement stopped

Run expanded-direct-seed42-v2 attempted three of32scheduled batches and acknowledged two optimizer updates. The third batch stopped at local loss_reduction_validation with ValueError. Diagnostics state optimizer_call_attempted=false: no third optimizer request was dispatched. Forward/backward had returned, so the trainer remains poisoned and must not be reused. The actual numerical discrepancy or shape failure was not retained by this frozen version, so the evidence does not yet justify changing a tolerance.

The isolated controller exited. No experiments in this study remain active. Automated launches are paused, and no further fresh-base replacements or optimizer replay will occur while the validation check is investigated offline. This is an incomplete run, not a model-improvement result. Confirmation remains untouched.

This run reserved $53.075600344856966; the total expanded study reserved $570.7733412454412. These are conservative reservations, not invoices. The provider credit ceiling is not the blocker.

Archive: artifacts/expanded-direct-seed42-v2-final-failure-results. Integrity inventory: direct-seed42-v2-failure-archive.json. See teacher-merged-audit.md for the separate SFT data gate: only34verifiedlineages versus200required.

[Run on W&B](https://wandb.ai/nmdatar-harvard-university/repository-qa-training/runs/expanded-direct-seed42-v2)
