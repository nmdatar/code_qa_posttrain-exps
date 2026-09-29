# Single live Nemotron grading check

One Qwen/Qwen3.5-4B repository rollout was generated on the previously used
Matplotlib source-reading task `import-55b0a97c4c4fa3ceab342e86` and graded once
by frozen NVIDIA-Nemotron-3-Nano-30B-A3B-BF16. No resampling, training, optimizer
update, checkpoint save, or experiment was performed.

The check passed the integration criteria: a completed cited answer, trusted
sandbox/tool telemetry, valid judge JSON, a resolved score, and sandbox cleanup.
The answer used three tool calls and 266 generated policy tokens. The complete
check took 22.56 seconds. Nemotron assigned 1.0. Conservative reservations were
$0.05965; actual provider billing remains unavailable. Raw generations, private
judge input, telemetry, and ledger are under `artifacts/nemotron-single-rollout-check/`.

Source review supports the answer's central claims: `None`/`BboxBase` handling,
`len()` with TypeError converted to ValueError, padding two coordinates with zero
width/height, and `Bbox.from_bounds(*bbox)`. However, the answer omits transform
storage/application and overgeneralizes invalid-sequence error handling. Full
credit is therefore arguably generous. This demonstrates that the judge adapter
works; it does not establish calibrated grading accuracy. No further paid checks
were run to obtain a more favorable result.

New experiment checkpoints request 172800 seconds (48 hours) for both training
state and sampling weights. This was checked in code/configuration and offline
tests; this check deliberately did not purchase a checkpoint save.
