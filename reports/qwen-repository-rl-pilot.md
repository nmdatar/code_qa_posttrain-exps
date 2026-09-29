# Qwen3.5-4B repository RL pilot

**Live integration succeeded: one acknowledged repository GRPO optimizer update.**

- Base model: Qwen/Qwen3.5-4B; rank-8 LoRA; learning rate 1e-5; no SFT.
- Four attempts on one real Matplotlib repository task; rewards: 0.75, 0, 0.75, 0.75.
- Tinker importance-sampling loss sum: 0.0016648360; numerical reduction verified.
- Training and sampling checkpoint saved, verified, and sampling weights loaded for evaluation.
- One development task: 0.5 before and 0.5 after. No demonstrated quality improvement.
- Frozen base-model reference judge is experimental and uncalibrated.
- 350 offline tests passed.
- All pilot attempts and recovery: **$4.0556 reserved of $5**. Reservations are conservative estimates; actual provider billing remains unavailable.
- New diagnostic checkpoints expire after **one hour**. Earlier pilot checkpoints retain their original one-day TTL.

## What was fixed

1. Prebuilt Modal source images need `/workspace`, and the original exec transport timed out. The replacement uses one isolated sandbox per episode with a bounded sequential stdin/stdout command server; live readiness and tools passed.
2. The original answer example contained literal `TASK_ID`, and strict parsing rejected Markdown fences. Prompts now use compact answer text/citations; the harness binds metadata to the task and observed source files. Raw token-level generations remain unchanged.
3. Three judge responses contained an extra quote after numeric score `0.75`. The parser now repairs only that unambiguous syntax error, without inventing a score. The complete unused group was recovered against its original immutable policy and optimizer state; no optimizer update was replayed. An exclusive recovery marker prevents reissuing it.

## Artifacts

- Result and checkpoint: `artifacts/repository-grpo-qwen-recovered/result.json`
- Update events: `artifacts/repository-grpo-qwen-recovered/run/events.jsonl`
- Recovery provenance: `artifacts/repository-grpo-qwen-recovered/run/recovery.json`
- Original raw judge responses: `artifacts/repository-grpo-qwen-v3/run/private/`
- Shared cost ledger: `artifacts/repository-grpo-pilot-v1/spend.json`
- Current configuration: `examples/training-repository-qwen-v3.json`

Clone the current configuration with a new run ID, output path, and an explicitly funded ledger for another experiment. The model stays Qwen3.5-4B. The small development sample and uncalibrated judge do not support conclusions about learning efficacy; an unchanged-model reevaluation also varied during debugging.
