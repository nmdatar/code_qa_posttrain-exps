# Bounded rollout concurrency

`training_pipeline` now supports parallel episode collection for collection GRPO,
development/base evaluation, and a training-task throughput benchmark. No paid
concurrent run has been launched; provider throughput and rate-limit behavior
still need the controlled benchmark.

```json
"concurrency": {"rollouts": 16, "judges": 4}
```

Both limits are positive integers at most 32. Omission keeps one rollout and one
judge. Limits apply to a single pipeline process: there can be at most `rollouts`
active episodes/sandboxes, and at most `judges` in-flight collection judge calls.
A judge limit greater than the episode count is valid; effective concurrency is
still limited by the number of episodes. Independent processes do not share a
sandbox semaphore; run benchmark arms sequentially. Strict calibrated repository
adapters remain serial until their external judge adapter is verified.

The worker pool submits at most its limit, returns results in input order, and
stops submitting replacements on a fatal error. It drains active episodes before
propagating failure or closing clients, so each episode can run its cleanup.
Provider/episode time limits still apply. Existing in-flight episodes may finish;
this is not a provider cancellation API. Reservations remain charged for unknown
outcomes. The process-locked spending ledger is shared by all workers.

Policy and judge SDK request submission and tokenizer/rendering access are
serialized; waiting on independent response futures overlaps. The collection
judge is initialized once under a lock. JSONL writes and optional W&B access are
serialized; each trajectory, judge request, and sandbox has a unique local path.
Private references remain in the host-side grading path.

GRPO collects attempts across all task groups using one immutable policy. Whole
groups with retryable infrastructure outcomes are retried within the same worker
limit; unresolved groups remain excluded and equal-reward groups contribute zero.
The optimizer and sampler refresh run only after the complete batch has drained.
No asynchronous learning, replay, clipping, or extra optimizer epochs were added.

## Experiment 1 benchmark

Eight frozen-policy configs, two repetitions per concurrency value, are in
`configs/experiments/01-throughput-c{01,08,16,32}-r{1,2}.json`.
They pin `experiments/qwen4b-throughput-v1.json`: 16 deterministically stratified
training tasks, four attempts each, temperature 1. The judge cap stays four in
all arms. A retry adds a complete group; reports count all attempted episodes.

```sh
# Offline validation only.
qa-train validate --config configs/experiments/01-throughput-c08-r1.json

# Paid benchmark — prepared, but do not execute until the user starts Experiment 1.
qa-train benchmark --config configs/experiments/01-throughput-c08-r1.json
```

The benchmark creates no trainer, gradient, optimizer update, or checkpoint.
It writes `benchmark.json` with wall time, episodes/second, completion, scoring
coverage, exclusions, token/tool totals, provisioning/generation/action/grading/
cleanup timings, judge queue/sampling time, and conservative cost reservations.
Phase times summed across workers can exceed wall time. Actual billing is
reported as unavailable rather than inferred from reservations. Complete
trajectories and failed attempts remain local.

Run the baseline selection, confirmation, and selection repeat plus the eight
throughput arms against the same $120 baseline ledger. This reallocates $75 from
the previous confirmation/contingency reserve; the project total remains $1,000.
Provider estimates and remaining ledger balance must be checked at launch.
Training configs remain serial until the throughput comparison selects a worker
count; set `concurrency` in a new run/fork, not by changing a resumed run.
