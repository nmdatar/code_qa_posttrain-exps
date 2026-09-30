# Experiment 1/2 readiness under the $1,000 ceiling

The current launch plan is in `configs/experiments/ready-v4/README.md`; original historical configs use the previous protocol. Full experiments have not been submitted. Provisional runtime setting: 32 rollouts / 4 judges; the formal repeated Experiment 1 comparison still selects the final setting.

## Changes

- Visible agent budgets now match the actual enforced limits; remaining-response reminders reserve a final answer. Read output includes line numbers, EOF requests clamp safely, invalid actions explain the validation error, and prompts request literal repository searches and concise valid JSON.
- All answer rows include reward, grading status/reason, phase and optimizer step in the W&B `answers` table. Evaluations have separate namespaced metrics with successful optimizer step as their x-axis. Long artifact names no longer exceed W&B limits; controlled stops finish successfully.
- Strict grading uses stable short evidence aliases, source line counts, larger judge context, bounded evidence reads, preserved oversized ranges split into chunks, and source context around reference snippets. Unresolved grades remain missing and their detailed diagnostics stay private. No tasks or false assertions are silently dropped to raise coverage.
- Cloud ledger cap migration preserves prior reservations. Explicit allocation manifests enforce the approved $1,000 project ceiling. Full-study estimates include larger grader contexts and retained best checkpoints.
- Best-checkpoint selection updates its pointer only after archive checksums and remote optimizer/sampler reload succeed. The live Tinker API rejected exporting optimizer-state archives. Sampler files can be archived; optimizer recovery remains a bounded remote capability. No off-provider durable optimizer restoration claim is made.

## Matched diagnostic results

These are readiness evaluations of unchanged base weights, not evidence of training improvement. Both r3 arms use the same protocol, reward identity and 32-task selection cohort, temperature 0, four judges.

| Rollout concurrency | Wall time | Resolved | Completed | Demonstrated quality |
|---|---:|---:|---:|---:|
| 16 | 76.06 s | 31/32 (96.9%) | 28/32 (87.5%) | 12.5% |
| 32 | 75.26 s | 32/32 (100%) | 29/32 (90.6%) | 18.75% |

The subsequent eight-judge check took 79.74 seconds, resolved 30/32 (93.75%), completed 28/32 (87.5%), and scored 9.375%. It also included a bookkeeping fix to remaining-tool reminders. It did not justify increasing judge concurrency; keep four judges.

32 rollouts pass the small-check absolute gates, but bring only 1.1% throughput improvement with four judges. These are single small observations; formal concurrency selection still requires Experiment 1's repeated concurrency-1 controls. Different scores at temperature 0 demonstrate residual inference/grader variation, not a causal concurrency quality benefit.

W&B r3 c32: https://wandb.ai/nmdatar-harvard-university/repository-qa-training/runs/readiness-selection32-v4-r3-c32-eval-054ba59c
W&B r3 c16: https://wandb.ai/nmdatar-harvard-university/repository-qa-training/runs/readiness-selection32-v4-r3-c16-eval-52e73e67

## Budget and scope

Cumulative ledger allocations: baseline/throughput $300, direct GRPO $550, lower-LR $80, reserves $70. Total $1,000. Historical spending stays in each ledger. Additional multi-seed training and additional LR arms have no allocation.

Experiment 1 scheduled-operation estimate is $236.60 before prior spending and readiness checks. Seed-42 direct GRPO is at most 14 attempted batches / 14 successful updates, estimated $447.44 before prior spending, including conservative archival storage reservations. Plans stop at lower limits when no signal, regression or budget controls fire. These are reservations, not actual bills.

The bounded experiment reduces the original procedure's 30-update, 60-attempt, three-seed scope and uses zero whole-group retries. Remote best-checkpoint retention is 14 days; routine checkpoints retain 48 hours. The original indefinite optimizer archival requirement is not satisfied by this provider integration.

## Verification

451 local tests passed after the final runtime changes. Focused regression tests verify ledger history preservation, refusal of unapproved cap changes, archive failure preventing best-pointer updates, optimizer reload acknowledgment, and lossless splitting of oversized source ranges. New W&B answer-table media and local `answers.json` contain all 32 answers with reward/status columns; no tracking-failure events were reported by either r3 run.

## Live recovery evidence

`readiness-archive-restore-v4-r3` passed entirely on Modal/Tinker with zero optimizer updates. Sampler archive: 72,980,480 bytes, SHA-256 `cee8a50d6600fe8e53acea707b8b77e734fef5ec4dedd19910c0f953e2077ed5`. A fresh Tinker training client acknowledged optimizer restoration and the sampler was reloaded. Remote retention is 1,209,600 seconds (14 days). Raw optimizer export and archive re-import remain unsupported. Two earlier failed diagnostics are retained and their reservations included, not hidden.

Candidate confirmation is separately prepared for 85 tasks and cannot affect best-checkpoint selection. Training was shortened to 14 batches/updates to include its conservative cost within the direct-GRPO allocation.

## Final cost check and launch status

With all readiness reservations preserved, baseline/throughput projects $274.43 against its $300 ledger. Direct training plus the 85-task confirmation projects $510.58 against $550. These bounds include existing reservations; the lower-LR allocation and $70 reserves remain separate. See `budget.json` for exact operation-level estimates.

Experiment 1 is ready to launch. Experiment 2 is prepared as the bounded single-seed study, conditional on the full Experiment 1 gates and the documented 14-day remote optimizer recovery window. The original indefinite off-provider optimizer-archive requirement and multi-seed study remain outside this implemented, budgeted scope. No full-study bundle has been submitted.
