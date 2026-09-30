# Initial Modal screens — complete

All four subprocesses exited 0. Modal campaign 5fc1c9dc978d7e05b5d09007c47a867653b4f492b74fedb38628b434e1e0d6b2 reports complete. Orchestration ran on Modal and model operations on Tinker.

| Benchmark | Episodes | Seconds | Episodes/s | Resolved | Positive rewards | Mean reward across all attempts |
|---|---:|---:|---:|---:|---:|---:|
| c08 | 64 | 95.66 | 0.6691 | 59/64 | 0 | 0% |
| c16 | 64 | 48.62 | 1.3164 | 60/64 | 3 | 4.6875% |

c16 delivered 1.97x throughput. These benchmark episodes use training tasks and are not the development evaluations below. Unresolved answers contribute no demonstrated reward but are not classified as incorrect.

| Training screen | Initial demonstrated quality | Final demonstrated quality | Initial/final scoring coverage | Updates | Batches | Outcome |
|---|---:|---:|---|---:|---:|---|
| Direct GRPO 1e-5 | 12.5% | 25% | 87.5% / 87.5% | 1 | 4 | complete |
| Lower LR 5e-6 | 12.5% | 12.5% | 87.5% / 100% | 0 | 3 | stopped_initial_zero_batches |

Each evaluation uses the same 16 selection tasks. Resolved-only mean reward: direct 14.29% to 28.57%; lower-LR 14.29% to 12.5% (denominator changed, demonstrated quality unchanged). Direct final coverage is below the runner's 95% eligibility threshold. No reliable LR conclusion: lower-LR received no update, and this is a tiny single-seed screen using the same judge for training and selection.

Direct saved three checkpoint manifests (initial, after update, terminal); lower-LR saved two unchanged step-zero manifests. Tinker sampler and training references are preserved under artifacts/remote-initial-screen-results; TTL 172800 seconds (48 hours).

Ledger reservations: baseline cumulative $15.69778395 (includes prior history); direct $12.99828599; lower-LR $10.77108735. Benchmark incremental episode reservations: c08 $4.38321924; c16 $4.44010722. These are reservations, not provider bills.

W&B: https://wandb.ai/nmdatar-harvard-university/repository-qa-training
The lower-LR run has two tracking_failure ValueError events uploading evaluation artifacts; evaluation metric events and authoritative local/Modal reports exist. Other runs have no tracking_failure events in the retrieved logs.

Agent contract: inspect pinned source with list_files, search_code, read_file; return JSON only, with an answer of at most 120 words and citations containing path/start_line/end_line from source actually read. Do not claim code execution. Config limits: 6 generations, 5 tool calls, 512 output tokens per call, 3072 total output tokens, 8192 context tokens, 300 seconds. Public task metadata advertises larger budgets; effective runner caps come from experiment config.

Reward is required-claim coverage subject to the full-claim audit: unsupported, false, or uncited assertions can reduce reward to zero. Invalid/incomplete judge audits remain unresolved. Missing answer or citations scores zero. Representative failures include an uncited Checkov answer and budget exhaustion without submission. A c16 Matplotlib sentinel answer received reward 1; this is the judge's assessment, not independent validation of every claim.

Evidence: latest-snapshot.json, answer-samples.json, and downloaded evaluation/checkpoint/ledger files under artifacts/remote-initial-screen-results.
