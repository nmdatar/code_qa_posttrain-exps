# Agent harness — build, research, optimize

Design proposal · 2026-09-28. The Modal environment factory and sandbox backend
are now implemented in `agent_harness`; see [usage and boundaries](../docs/MODAL_ENVIRONMENTS.md).
The shared episode loop and Tinker training integration are now implemented; see [training pipeline usage](../docs/TRAINING_PIPELINE.md). The integrated [research runtime](../docs/AGENT_HARNESS.md) and [remote coordinator](../docs/REMOTE_HARNESS.md) provide typed tools, full-loop workers, separate graders, and detached execution. These retain separate interfaces from the training runner; see [integration status](../docs/WORKTREE_INTEGRATION.md).

[Editable Excalidraw diagram](https://app.excalidraw.com/s/8Ufs2ZMhWhu/7us72y2PT2q) · [Local editable copy](diagrams/agent-harness.excalidraw)

The diagram has three sections: implementation milestones, the research episode loop, and policy/harness optimization. The purple inner feedback arrow is the next on-policy training iteration. The outer cycle represents periodic independent evaluation and separately versioned harness experiments; changing tools is not a requirement at each optimizer step.

## Design and interfaces

Build a Python `agent_harness` package shared by interactive research, evaluation, teacher collection, and RL rollouts. Keep the loop independent of model providers, tool domains, and sandbox backends.

| Contract | Responsibility |
|---|---|
| ResearchRequest | Question, resource references, requested tool scope, run limits; no private gold or rubric |
| ToolSpec | Name/version, extensible type string, capabilities, input/output JSON schemas, execution policy and limits |
| Tool.execute(arguments, context) | Execute with host-authorized resource handles; return a structured observation |
| ToolRegistry | Explicitly configured built-ins and Python entry-point plugins; validate uniqueness and schemas |
| ModelAdapter | Tool action or final response, provider usage, and exact sampled tokens/log probabilities for RL |
| SandboxBackend | Prepare, execute, and destroy an isolated episode environment |
| EpisodeResult | Submission, termination reason, trajectory reference, trusted metrics reference |

Use `type="code_understanding"` initially. Capabilities include list, search, read, symbols, and execute. Types remain extensible strings rather than a closed domain enum.

Available tools are the intersection of configured plugins, requested domain/capabilities, task-permitted tool names, and host-authorized resources. Recheck authorization at dispatch. Type metadata does not create a security boundary. Adding a plugin must not require editing the loop. MCP loading is a later adapter.

The repository output adapter uses the existing AnswerSubmission schema. Reuse EpisodeRecorder and the qa_eval grading/reward contracts. Keep private TaskSpec contents, signing keys, grading data, and model credentials outside all tool environments.

## Research loop and baseline tools

Resolve the pinned snapshot and tool set; build context; request one action; validate scope, schema, and budget; execute; record the observation and host telemetry; update context; repeat or finalize.

Use sequential tool calls inside an episode and parallel episodes across workers. Keep append-only trajectory events and content-addressed output artifacts. Bounded context management retains instructions, the question, recent turns, and evidence references; replace old payloads with retrievable artifact references and log each transformation.

| Tool | Behavior |
|---|---|
| list_files | Paginated tracked-file catalog with glob filtering |
| search_code | Bounded literal/regex search with paths and line numbers |
| read_file | Bounded source spans with commit and full-file hash |
| find_symbols | Python AST definitions; explicitly report unsupported languages |
| read_artifact | Bounded retrieval of this episode's previous tool output |
| run_tests | Structured selection using a manifest-defined runner and validated arguments |
| python_probe | Bounded Python snippet in the isolated execution environment |

Preserve the pristine citation snapshot separately from execution state. Record stdout/stderr, exit status, hashes, and elapsed time. Agent probes are not private evaluator fixtures. Enforce paths, symlink restrictions, output limits, timeouts, and child-process cleanup.

Every terminal path persists telemetry, including completed, budget_exhausted, agent_error, and infrastructure_error. Missing trustworthy usage is unresolved for grading rather than fabricated as zero.

## Remote scaling

Use Modal for training environments and Docker for local development and contract parity tests. A remote coordinator runs bounded worker pools for sampling, sandbox execution, and grading; training throughput does not depend on a laptop.

Cache immutable repository images and isolate each episode's writable state. Build dependencies before rollouts; deny runtime network access and avoid credentials or host mounts. Use durable shared artifact storage, episode/attempt IDs, policy versions, and idempotent completion records. Restart interrupted episodes cleanly while preserving attempt telemetry.

Require explicit experiment settings: supported model/version, concurrency, group size, budgets, retry limits, and spending cap. Measure a smoke run before freezing comparison budgets. Distinguish provisioning/queue time from dispatch-to-answer latency; include all attempts in total experiment cost.

This extends the existing training design's local-artifact default for remote rollout use. The remote harness provides separate Modal Volumes for public inputs, private grading inputs, rollout artifacts, grades, and coordinator journals.

## Optimization through RL

1. Measure retrieval and untrained tool-policy baselines on the same development cohort.
2. Calibrate the existing correctness/evidence evaluator before trusting numerical rewards.
3. Optionally SFT on reviewed training trajectories when tool use or output structure needs a warm start.
4. Sample complete same-task, same-policy groups using real tools. Preserve exact token IDs, sampling log probabilities, model/tool/environment versions, and all usage.
5. Pass verified reports through qa_eval.rl.prepare_group. Exclude an entire unresolved group; equal rewards yield zero advantages.
6. Apply the Tinker-backed training strategy described in the training architecture. Optimize assistant-generated tokens only, including tool calls; mask prompts and environment observations.
7. Independently evaluate checkpoints with frozen tasks, tools, budgets, and grading contracts. Require existing promotion gates and human audit.
8. Run separate controlled experiments on tool descriptions, symbol tools, execution availability, context handling, and budgets. Version and remeasure these changes instead of mixing them into an RL comparison.

Keep the existing reward tiers:

- Failed: 0.
- Partial: 0.2 × supported coverage.
- Accepted: 0.9 + 0.1 × efficiency.
- Unresolved: no numerical reward; retry or quarantine the group.

RL learns investigation choices and stopping behavior. It does not rewrite the dispatcher, tools, or authorization rules. Log acceptance, material errors, citation support, repeated searches, invalid calls, timeouts, reward variance, excluded groups, policy drift, and all-attempt cost per accepted answer.

Keep family-level train/development/final-test separation. Final-test data does not influence training or routine harness tuning. Training reward alone cannot select a release.

See [training architecture](TRAINING_ARCHITECTURE.md), [training interfaces](TRAINING_INTERFACES.md), and [training decisions](TRAINING_DECISIONS.md) for the reusable optimizer framework; this document does not silently change its proposed algorithm or loss reduction.

## Implementation milestones and acceptance

1. Contracts, registry, fake adapters, and deterministic loop tests.
2. Repository tools, isolated execution, artifacts, and evaluator integration.
3. A real end-to-end investigation and baseline budget measurements.
4. Remote rollout coordination and Tinker trajectory export.
5. Calibrated RL comparisons and independent checkpoint review.

Acceptance covers plugin loading without loop edits; rejected duplicate names and invalid schemas; scope/path enforcement; citation validity against pinned code; sandbox isolation; budget/error handling; complete telemetry; Docker/Modal parity; replay fidelity; token/log-probability/mask alignment; held-out and unresolved-group exclusion; and recovery without duplicate training updates.

## Diagram verification

The scene was visually inspected after correcting feedback-arrow routing. The local export contains 82 editable elements and 22 arrows, with valid source and target bindings. It is a private scene under existing account permissions; no public share was created. Exported scene version: `8d73ca58`.

Diagram checks validate the design artifact, not the implementation acceptance criteria above.
