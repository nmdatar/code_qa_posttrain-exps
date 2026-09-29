# Qwen policy / Nemotron judge preflight

The new configuration is `examples/training-repository-qwen-nemotron.json`.
The unchanged solver is Qwen/Qwen3.5-4B. The frozen training and development
judge is NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 with the cookbook
`nemotron3_disable_thinking` renderer, temperature 0, 16,384 context tokens,
and up to 1,024 output tokens.

`qwen-nemotron.json` records successful remote model-capability and renderer
preflight, using Tinker 0.30.4 and cookbook 0.5.7. No inference, optimizer update,
Modal episode, or checkpoint save was requested. Validation resolved 858 training
and 117 development tasks. Preflight used the existing `.venv-posttrain`
dependency environment; `.venv-eval` still needs the optional cookbook package
before running this judge. See the optional installation command in
`docs/TRAINING_PIPELINE.md`.

Offline validation includes injected provider tests for sampling-only capability,
separate model identity, model-specific pricing, shared-budget refusal before
sampling, context overflow, failed-preflight cleanup, truncated-grade raw
persistence, configuration identity changes, and unchanged legacy behavior.

Published judge pricing was checked on 2026-09-29 UTC at
https://tinker-docs.thinkingmachines.ai/tinker/models.json:
$0.195 per million input tokens and $0.495 per million output tokens.
These are estimates at published rates, not billing evidence. Refresh before
execution. The example retains the old pilot's ledger and creates no additional
spending authorization.

No accuracy benefit has been measured. Freeze this judge and re-score the base
and candidate models under the same evaluation protocol before making an
improvement claim. The previous one-update pilot demonstrated integration only.
