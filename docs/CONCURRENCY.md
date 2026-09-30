> Production collection runs now execute in a detached Modal sandbox. Use
> [Remote experiments](REMOTE_EXPERIMENTS.md); the local paid command examples
> below are historical and now fail closed. Worker limits still apply remotely.

# Bounded rollout concurrency

`training_pipeline` now supports parallel episode collection for collection GRPO,
development/base evaluation, and a training-task throughput benchmark. Measured throughput and rate-limit behavior are study-specific; consult the linked reports rather than treating configured concurrency as a benchmark result.

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

## Throughput benchmark designs

The retained [campaign-v8 suite](../configs/experiments/current-v8/README.md)
contains eight frozen-policy throughput designs, two repeats at each concurrency.
Their exact task manifest, judge cap, retry behavior and budgets are specified in
the configs. Run timed throughput arms sequentially so they do not compete.

```sh
# Offline configuration validation; no provider allocation.
python -m training_pipeline validate --config configs/experiments/current-v8/01-throughput-c08-r1.json
```

Use the [remote execution guide](REMOTE_EXPERIMENTS.md) for packaging and
submission. The benchmark creates no trainer or optimizer update. It reports
wall time, completion and scoring coverage, token/tool totals, stage timings,
and conservative cost reservations. Summed concurrent stage times can exceed
wall time; reservations are not provider invoices.

The early baseline failed its scoring-coverage gate, and its planned throughput
sweep was incomplete. Its old allocation and launch instructions are preserved
in the [historical index](EXPERIMENT_HISTORY.md), not recommendations for new runs.
