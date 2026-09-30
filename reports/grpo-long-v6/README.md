# Longer direct GRPO run

User requested launch on 2026-09-29. Fresh Qwen3.5-4B, rank 8, seed 42, LR 1e-5. Latest frozen repo-qa-training-claims-v6 release, all-claims-v7, positive-coverage-v4, definition-context-v1, action-alias-v1 and paginate-v1. No SFT initialization.

16 attempted batches, 2 tasks × 4 attempts = 128 training attempts, up to 16 updates. Original rollout limits retained. No initial no-signal stop; constant-reward groups remain excluded from updates. Checkpoints every successful update. Strict initial/final evaluation on the unchanged 32-task selection set; confirmation untouched.

Expected duration 20–40 minutes from dispatch, inferred from latest v6 smoke episode timings (48 training episodes totaled 1368 worker-seconds; 32-task evaluations took 98–105 seconds), sequential batch barriers, optimizer/checkpoint overhead, and startup allowance. This is an estimate; hard controller timeout 2 hours.

Project ceiling remains $1000 using previously authorized rebalancing: baseline $260 (reserved $257.27584), GRPO $650 (reserved $294.07845), LR $20 (reserved $10.77109), other reserves $70. All affected ledger histories included in bundle so worker applies both reductions and increase. New conservative bound $347.01091; cumulative GRPO upper bound $641.08936. These are reservations, not invoices.

Offline configuration/release validation passed (858 training, 117 dev). 67 focused grader, reward, remote and budget tests passed. Frozen bundle verifies all code/input file hashes. No grader/rubric source code modified for this launch.

## Launch

Successfully submitted after explicit user approval of external upload and remote spending. Modal sandbox `sb-4tThg2ZsP7NWu7fh0pZgPl`; bundle `7d791f7d7943bb2f5c2f08b770fdf41b42cd1986164b2b2ae2ce8fd15b488b12`. Initial status: controller active, campaign starting, run metadata written. Heartbeat `follow-longer-grpo-run` enabled every five minutes for progress and final results.
