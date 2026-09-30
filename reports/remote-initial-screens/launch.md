# Initial screens: remote launch

Submitted four user-authorized screens in detached Modal sandbox `sb-shEcW844hwZMT2Lr29PoEL`.
Bundle: `5fc1c9dc978d7e05b5d09007c47a867653b4f492b74fedb38628b434e1e0d6b2`.
Receipt: `artifacts/remote-initial-screens.submission.json`.
Volume: `repository-qa-training-state-v2`.

Three subagents handled campaign/throughput and the two training screens. The existing controller executes c08 and c16 sequentially, then both training screens in parallel. All experiment orchestration runs in Modal; policy/judge sampling, training, and checkpoint storage use Tinker.

Conservative campaign upper reservation estimate: $166.005343232. This is not an actual bill; existing caps and ledger history are preserved.

W&B online sync confirmed for c08:
https://wandb.ai/nmdatar-harvard-university/repository-qa-training/runs/throughput-c08-r1-claims-qwen397-v1-all-claims-v3-modal

Project: https://wandb.ai/nmdatar-harvard-university/repository-qa-training

Current-chat heartbeat `report-initial-modal-screen-results` checks every five minutes and will report terminal results/failures here, download artifacts, and pause itself. No blind retries or optimizer replay are authorized by that monitor.

Training screens allow four attempted batches/up to four updates, with 16-task initial/final selection evaluations. Three initial batches without trainable rows trigger a controlled no-signal stop. Report demonstrated quality alongside scoring coverage; same frozen judge is used in training and selection. Checkpoints have 48-hour retention.
