# REINFORCE versus GRPO: promising point estimate, inconclusive comparison

Both runs completed 16 batches /128 training attempts on the same 32 training tasks, with all128 training rewards resolved. REINFORCE completed in **24 minutes 9 seconds** (W&B runtime); GRPO took24minutes29seconds.

| Measure | GRPO | Running-baseline REINFORCE |
|---|---:|---:|
| Acknowledged optimizer updates | 11 | 16 |
| Contributing trajectories | 60 | 122 |
| Zero-variance question groups | 17/32 | 14/32 |
| Mean training reward | 0.21224 | 0.22161 |
| Initial strict passes | 8/32 | 7/32 |
| Final strict passes | 6/32 | 9/32 |
| Initial demonstrated credit | 25.0% | 21.875% |
| Final demonstrated credit | 18.75% | 28.125% |
| Within-run change | −6.25 points | +6.25 points |
| Initial resolved grades | 28/32 | 29/32 |
| Final resolved grades | 31/32 | 29/32 |

REINFORCE's change exceeded GRPO's by12.5percentage points; paired task-bootstrap95%interval **[−12.5,+37.5] points**. Its own change interval is[−9.375,+21.875]points. Four tasks gained demonstrated credit and two lost credit within REINFORCE. Unresolved grades remain unresolved and contribute no demonstrated credit, not proof of incorrectness. Final completed answers fell32→30.

The result demonstrates that prior-running-mean REINFORCE can update on groups with identical rewards, including all-zero batches. It does **not** establish better generalization or statistical superiority. This is a sequential single-seed comparison on a small selection set; the intervals do not capture repeated generation/judging uncertainty. Both use nominal LR1e-5, but different advantage scales imply different effective update scales. No independent factual regrade or confirmation evaluation was launched.

Matched checks passed: exact training task multiplicities, selection membership, data identity, environment identity and reward version. Both started from fresh Qwen3.5-4B rank8 seed42 with identical rollout/evaluation limits and v6 rubrics/v7 grader. REINFORCE uses reward minus cumulative mean of eligible rewards from earlier batches, initially zero, without group standardization. Complete-group admission and assistant-token masks remain matched. The final baseline checkpoint contains128 rewards summing28.3666667.

Final REINFORCE checkpoint: `ckpt-e443ccb1419e456bb3037f46aa68b62b`. Remote sampler/optimizer retention is48hours from save. Neither REINFORCE evaluation met95%coverage, so no best-eligible extended-retention checkpoint or sampler-weight archive was produced. Downloaded checkpoint manifests are not permanent weight exports.

Incremental REINFORCE ledger reservations: **$65.7540**, versus $288.814942 conservative bound. Cumulative shared GRPO ledger: **$428.3234/$655**. GRPO comparison run added$68.4910. Reservations are not provider invoices; project ceiling remains$1000.

Archived and JSON-verified1462files, including campaign outputs, events, trajectories, evaluations, checkpoint manifests, W&B data and authoritative ledgers. SHA-256 hashes recorded in checksums.json. Raw archive: ../../artifacts/reinforce-v6-results/. Monitoring is being paused after reporting this completed comparison.

[REINFORCE W&B](https://wandb.ai/nmdatar-harvard-university/repository-qa-training/runs/reinforce-v6-seed42-v1) · [GRPO W&B](https://wandb.ai/nmdatar-harvard-university/repository-qa-training/runs/grpo-long-v6-seed42-v1) · [Comparison data](comparison.json) · [Archive](archive.json)
