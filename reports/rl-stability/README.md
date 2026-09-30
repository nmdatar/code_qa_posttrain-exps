# Parallel tools and checkpointed stability experiments

Parallel read-tool execution and a bounded adaptive controller are implemented. **The LR1e-5 campaign stopped after a parallel-transport failure and is archived.** See [RESULTS.md](RESULTS.md). See [STATUS.md](STATUS.md) and [LAUNCH.md](LAUNCH.md) for the active protocol. The original design below is retained as background and is superseded by LAUNCH.md wherever they differ.

## What is implemented

Set `environment.tool_parallelism` to 2–8 for collection training/evaluation; omitted or 1 preserves the serial protocol. The policy can emit:

```json
{"tool_calls":[{"tool":"search_code","arguments":{"query":"parse"}},{"tool":"read_file","arguments":{"path":"src/parser.py","start_line":1,"end_line":60}}]}
```

Only independently valid, permitted `list_files`, `search_code`, and `read_file` actions can share a batch. A dependent read must wait for search results. Final answers remain separate. The entire batch is validated before dispatch; each member consumes one call. Whole-batch reservations enforce remaining task, runner and sandbox budgets. Results have stable zero-based `call_index` values. Existing single-action parsing and action aliases remain supported; batches require canonical tool keys.

Modal batch execution uses independent exec processes. Its serial persistent stream cannot execute simultaneous commands and is deliberately bypassed for batches. Commands overlap while owner-thread locking protects lifecycle, budgets and journals. Workers drain before cleanup; an uncertain command outcome fails the episode without replay. Nonzero command exits remain observations. The combined response retains the existing 3,500-byte budget, trimming individual outputs while preserving all call identities at that cap. Full results remain in trajectories. Timing records wall duration for the batch; trajectories expose `tool_turns`, `max_parallel_calls`, and `mean_calls_per_tool_turn`.

This reduces serial model/tool round trips when a policy chooses useful independent calls. Actual end-to-end speed and quality require measurement: remote exec overhead, shared CPU, extra context and duplicate searches can erase the gain.

## Evidence-based starting point

Use fresh Qwen3.5-4B, rank 8, LR **5e-6**, group **4**, two questions per batch. Keep the v6 repaired atomic rubrics, `all-claims-v7`, `positive-coverage-v4`, pagination, action aliases and definition-context evidence. Do not initialize from tool-only SFT.

The prior sweep's 5e-6/group4 arm improved from 6 to 10 strict passes out of 32. All final grading coverage was below 95%; the 1e-5/group4 comparison suffered 104 infrastructure failures. This is a provisional anchor, not an established optimum. Running-baseline REINFORCE used 122/128 trajectories versus GRPO's 60/128, but its quality advantage was inconclusive and was measured at 1e-5. Combining it with 5e-6 is a new experiment, so retain matched GRPO controls. Tool-only SFT reduced end-to-end passes from 5 to 3/32.

Evidence: main checkout `reports/reinforce-v6/RESULTS.md`, `reports/sft-tool-warmup-v1/RESULTS.md`, and parallel-experiment-campaigns worktree `reports/parallel-experiment-campaigns/RESULTS.md`. `inherited-source.json` records source hashes copied from the active checkout; these inherited files were not all present in its HEAD.

## Sequential experiment schedule

1. **Frozen speed screen:** width 1, 2 and 4 on the same 32 selection tasks, twice per arm. Alternate arm order between repetitions. Keep total calls=5, generations=6, context=8192 and output budgets unchanged. Record p50/p95 episode and model/tool latency, strict/factual quality, grading coverage, call duplication, serial turns, tokens and cost. Do not credit a speed gain caused by missing answers. This measures batching at fixed total work budget, rather than granting a larger search budget.
2. **Matched RL screen:** fresh-base serial GRPO, parallel-4 GRPO and parallel-4 running-baseline REINFORCE, all at 5e-6/group4. Use the same shuffled 32 training questions, 16 attempted batches and 128 rollout attempts. Evaluate at initial, batch 8 and final batch 16 checkpoints; report attempted batches and acknowledged updates separately. A 1e-5 REINFORCE bridge arm is optional only within a separately estimated budget.
3. **Stability ablations:** control; overlong masking alone; invalid-format termination alone; inverse parallel-width scaling alone; then the combined treatment. Fresh matched starts isolate each change. Do not adapt the reward/judge at the same time. Selection of a combination is exploratory and must be replicated.
4. **Replication:** incumbent versus at most one challenger at seeds 43 and 44. Freeze the chosen recipe, then evaluate the untouched 85-task confirmation cohort once. Confirmation never feeds the search loop.

Use the final scheduled checkpoint as the primary outcome. Selection of a transient best checkpoint is secondary. Require at least 95% resolved selection grades, report paired task and repository-cluster bootstrap intervals plus missing-grade bounds. An exploratory promotion requires nonnegative point quality change and a paired interval excluding a loss over 1/32; on quality ties prefer lower p50 and p95 latency, then cost. A 32-task screen often cannot pass this precision gate: retaining the incumbent is a valid result. Do not assert statistical superiority after searching multiple candidates.

## Mapping Cognition's instability ideas

The [SWE-grep instability section](https://cognition.com/blog/swe-grep) motivates masking overlong/extreme-ratio trajectories, eliminating format bonuses, terminating malformed actions with zero reward, and scaling advantages by tool width. These are experiment hypotheses for this task, not measurements on our agent.

- **Overlong mask:** add explicit termination causes for output-token truncation and total generated-token/generation caps; do not equate all `budget_exhausted` events with overlong generation. Keep resolved rewards in group baseline statistics, mask their loss rows, and retain the pre-mask contributing-count denominator so filtering does not automatically amplify survivors. Log masked fraction. Tool-result clipping alone is not an overlong policy trajectory.
- **Malformed format → zero:** distinguish malformed policy envelopes/arguments from permission errors, valid command nonzero exits, judge failures and infrastructure failures. Only the preregistered syntax/schema violations terminate at zero. Keep the invalid assistant tokens eligible for negative advantage; zero reward is not zero loss. For a malformed batch, reject all members before executing any. No format bonus is added in any arm.
- **Tool-width scaling:** for trajectory j, use `A_j / max(1, calls_j / tool_turns_j)`, where tool turns exclude final answers and invalid unexecuted envelopes. Apply once after the algorithm's baseline/standardization; do not recompute group centering. Preserve the existing token normalization. This is our explicit inverse-width hypothesis: the article does not specify an exact formula. Measure the effect on negative as well as positive advantages and duplicate calls. It is distinct from dividing by total episode calls.
- **Extreme ratios:** defer until a forward-only learner-logprob prepass and row-to-trajectory mapping exist. Compute log sequence ratios only over assistant tokens across all turns; never environment tokens. Prespecify thresholds using training-only diagnostics, for example log-ratio outside ±log(10) as an exploratory starting threshold, and record retention/ESS. Mask before backward. Current post-backward diagnostics are too late. Sequence weighting would also require replacing/compensating the current token-IS loss to avoid applying ratios twice; it is a separate loss ablation, not part of parallel dispatch.

## Durable autoresearch controller contract

Persist immutable candidate config/source/data identities, run ID, reserved budget, submission receipt, last committed sampler/optimizer checkpoint, completed evaluations and decision. Use the states in `experiment-plan.json`. Write state atomically before/after transitions. On restart reconcile the remote job and receipt before submitting anything; a missing acknowledgement is not permission to relaunch. Allow one training job at a time and at most five adaptive candidates. Predeclare thresholds and candidate order; do not rewrite prompts/judges from selection failures.

Existing local `Pipeline.run(..., purpose="resume")` restores weights, optimizer, Python RNG, task cursor and the REINFORCE running baseline. It saves every acknowledged RL update. Exact service-side sampling reproducibility is not guaranteed by restoring Python RNG alone. Changed scientific settings require a new fork/run and fresh optimizer. Add durable checkpoints for skipped batches and decision state before enabling unattended continuation. An ambiguous optimizer outcome must discard the client and reconcile/restore a committed state, never blindly replay.

**Remaining implementation before unattended remote use:** expose and test exact `resume` in the remote dispatcher; the current remote API supports run/fork/evaluate but not resume. Implement the durable candidate state machine and stability flags above. Preserve normal 48-hour remote state retention and 14-day eligible-best retention; verify expiration before resume. Downloaded sampler archives do not imply durable optimizer-state export. Keep recovery checkpoints distinct from promoted best checkpoints, and archive checksums/configs/rollouts/evaluations even for rejected arms.

Budget gate: reconcile active runs and authoritative project ledgers, update the price snapshot, and estimate the entire next arm including intermediate evaluations, judge retries and controller time. Do not inherit the old sweep's allocation as fresh spending authority. No caps were increased by this work.

## Verification and review

Final combined regression run: **221 offline tests passed**; see `tests.log`. It covers both parallel paths, budget admission, ordered outputs, truncation, failure cleanup, existing checkpoint/storage behavior, training strategies, adapters and remote interfaces. No live provider or paid training jobs ran. `offline-timing.json` records a synthetic four-command median of 177.8 ms serial versus 46.9 ms parallel (3.79×); this is a dispatch sanity check, not a measured agent speedup.

`implementation.patch` isolates this task’s code/test changes from the copied working-tree baseline. `implementation-files.json` records the changed paths and any baseline source drift. General agent usage and platform limitations are in [generic-agent.md](generic-agent.md). Work remains on branch `codex/rl-stability`; the main checkout files were not edited.
