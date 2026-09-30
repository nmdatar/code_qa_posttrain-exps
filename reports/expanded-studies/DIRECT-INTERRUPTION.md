# Direct REINFORCE interruption

The first actual direct seed42 launch stopped with AmbiguousUpdate during the second batch. One optimizer update is acknowledged and persisted in checkpoint ckpt-3b77a42a5f8d40a8a941fb0b2ffebe3e. Batch2 has32 resolved trajectories and no update acknowledgement. The log reports only the exception class, so it cannot establish whether forward/backward, the local loss check or optimizer dispatch failed. Do not replay or resume the uncertain call.

The complete stopped campaign/run and authoritative ledger snapshot were archived to artifacts/expanded-direct-seed42-failure-results (774files, zero JSON parse errors). The student ledger snapshot is $62.58380890057104, including the earlier aborted phase1 controller allowance; teacher ledger $354.0393030000065. These are reservations, not invoices. No final evaluation or confirmation result exists for this incomplete run. Initial strict demonstrated credit9/32 with only26/32resolved is diagnostic, not evidence of improvement.

Teacher recovery is independent and now submitted for245missing/infrastructurefailed attemptslots. Future RL work awaits a documented safe recovery strategy; local failure observability is being improved without changing the frozen failed run.
