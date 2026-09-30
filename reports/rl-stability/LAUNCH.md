# LR 1e-5 stability autoresearch launch

User requested LR 1e-5 and launch on 2026-09-29. This supersedes the earlier design-only LR 5e-6 schedule for this bounded campaign.

All five candidates use fresh Qwen3.5-4B rank8, prior-running-mean REINFORCE, LR1e-5, seed42, two training tasks per batch and four trajectories per task. Parallel read width4, total episode calls5, generations6, original data/reward/judge/selection splits unchanged. Eight attempted batches and at most eight acknowledged optimizer updates per arm (64 trajectories). Initial/final evaluation uses the same32 selection tasks; confirmation85 is untouched.

Control runs first. The durable controller uses observed training failures to choose the next unused intervention: overlong loss masking, malformed-format termination at zero, inverse tool-width advantage scaling, then the combination. It does not rewrite the reward, judge, datasets or code while running. Each arm starts fresh; checkpoint continuation preserves its exact condition. Five arms and four hours are hard bounds, not a claim that reward will improve.

Every acknowledged optimizer step and skipped RL batch saves framework/cursor/baseline state plus remote weights/optimizer artifacts. Controller state and decisions are atomically persisted. Recovery only accepts a proven finished/committed boundary; pending saves, uncertain optimizer outcomes and infrastructure failures require reconciliation, never blind replay. General remote resume dispatch is still not exposed: controller recovery is internal, and this launch does not authorize resubmitting a failed sandbox automatically.

Selection rule: final coverage>=95%; same cohort/data/reward identity; improve demonstrated strict quality by at least0.01 against the control, beat the incumbent and avoid completion loss>0.02. This is a provisional screening rule, not a statistical superiority claim or deployment decision. Training reward and contributing trajectories are diagnostics; no reward bonuses for formatting. A failed coverage gate is inconclusive, not proof of policy failure. The earlier design's bootstrap promotion rule remains work for replication, not implemented by this short screen.

Budget: conservative upper estimate $931.365506 across five arms, including controller reservations; campaign reservation cap $1000. The documented existing $6000 allocation is preserved: $1000 historical allocation, $517.697741 retired shared study reservations, $1800 existing isolated-direct allocation, $1000 new stability cap and $1682.302259 remaining reserve. No historical ledger is reset and no credits are purchased. Reservations are not provider invoices. Policy/judge prices were refreshed from official Tinker data; sandbox prices checked at https://modal.com/pricing on2026-09-29.

Routine remote checkpoint retention48hours; eligible-best retention14days. Sampler archives are not permanent optimizer exports. The controller's final state, checkpoint receipts and each run's W&B tracking file are authoritative for what actually completed.

Preflight: 157 final focused offline tests passed. New loss/format flags also passed dedicated unit tests. A real provider/sandbox execution can still fail; the controller stops rather than relabeling infrastructure errors or retrying optimizer calls. Extreme importance-ratio masking remains deferred pending pre-backward learner-logprob support.

Submission and live status are recorded separately in submission.json and STATUS.md. Never resubmit a pending receipt.
