# 06 — Frozen-policy tools, retrieval, and history

## Current runnable scope

Current-v8 provides runnable frozen-base, twice-repeated control/symbols, lexical/hybrid-subword, and hybrid-subword/history comparisons, plus finalist-only confirmation configs. The actual collection adapter now exposes Python AST definition/reference-occurrence tools, 512-token bounded source retrieval, and an evidence ledger with episode-scoped archived-observation readback. The first hybrid variant uses explicitly frozen char-trigram-hash-256-v1 embeddings; pretrained neural embeddings remain a distinct follow-up rather than being claimed implemented. Sources/index parameters are manifest-bound and exclude private grading data. Python execution remains disabled for the source-reading pool.

See the [current suite](../configs/experiments/current-v8/README.md) for authoritative settings. The design below includes historical larger-scope extensions; it does not override current configs.

**Question:** Which harness changes improve Qwen3.5-4B quality or cost before paying for retraining?

**Hypothesis:** Better evidence access or representation can outperform additional training, but extra tools and compression may also increase errors, latency, or citation loss.

**Dependencies:** [Baseline and throughput](01-baseline-and-throughput.md), [tuned policy](03-grpo-learning-rate-and-group-size.md), optionally [efficiency comparison](05-quality-gated-efficiency-reward.md); [roadmap items 6, 7 and 9](../experiments.md). If no trained model is confirmed better, use the unchanged base and say so.

## Frozen arms and controls

Use one durably retained policy checkpoint with no optimizer calls during harness screening. Freeze Qwen identity, grader, task IDs, episode budgets, and source snapshots. Run selection experiments at evaluation temperature 0; repeat each comparison twice to expose provider/runtime variability.

1. **Tools:** list/search/read control versus control plus structured symbol-definition/reference lookup. The added tool must expose pinned source, paths, line numbers, and hashes. No private rubrics or answer-location hints enter the index.
2. **Retrieval:** with the selected tool set fixed, compare on-demand lexical snippets versus hybrid lexical/embedding retrieval, both through a bounded `get_context` interface. Cap each retrieval response at 512 tokenizer-counted tokens and keep the same total episode context/output budgets. Freeze the embedding model/version, index parameters, and source-only index manifest before executing this arm; if absent, mark hybrid retrieval blocked rather than choosing a model during the run.
3. **History:** with retrieval fixed, compare raw history versus a versioned, deterministic evidence ledger plus retrievable archived observations. Start with no auxiliary summarizer model. Compact only completed old tool exchanges, preserving citation/source handles and artifact references. Do not rewrite generated policy actions or their stored token/logprob records.

Do not combine multiple changes in one comparison. Keep the last confirmed harness control if a stage is inconclusive. Repo-map representations, automatic context injection, generic-shell comparisons, and learned annotations remain follow-up ablations rather than expanding this first procedure.

## Prerequisites and execution

1. Verify that each tool is exposed through the actual training/evaluation adapter, with isolated sandbox access, path bounds, output limits, schema validation, telemetry, and cleanup tests. Availability in a separate research runner is not sufficient. Persist a concrete implementation/configuration manifest for each arm before running it.
2. Price the repeated selection comparisons, index creation, embedding calls, confirmation evaluation, and retained artifacts. Include cold indexing and amortized per-question costs separately. Existing authorization does not cover these calls.
3. Evaluate tool arms on the 32 selection tasks. Lock the winning variant using shared quality selection and cost tie-breaks. Repeat sequentially for retrieval and then history. Do not inspect confirmation results during this choice.
4. Evaluate the selected combined harness and original harness on the 85 confirmation tasks using the same frozen checkpoint, with two repeat evaluations of each. Report paired, family-clustered differences and per-task regressions. These are inference repetitions, not training-seed replications.
5. Execution probes are a separately registered extension only on tasks whose environments, dependencies and verifier support execution. The current source-reading pool does not meet that requirement: mark this arm blocked, never silently enable arbitrary execution or reinterpret source-only scores.
6. Only after a frozen-policy harness wins confirmation, register retraining: train original-harness and improved-harness arms from the same selected initialization, using the same winning GRPO recipe, seed 42 screening and seeds 43/44 confirmation. This avoids assuming a policy trained under old tools automatically benefits from the new interface. Retain the frozen-policy measurements as separate results.

## Metrics, stopping, and promotion

Measure quality, scoring/completion coverage, citation support, relevant evidence retrieved, repeated reads, tool errors, context overflow, lost citations after compaction, input/output tokens, p50/p95 latency, indexing cost, and all-attempt cost. Stratify descriptively by lookup versus cross-file tasks; do not relabel source-only tasks as runtime tests. Embedding/index work must not see questions, answers, or private grading context except public inference queries at retrieval time.

Stop on private-reference leakage, altered source identity, cross-episode state leakage, loss of token attribution, or archive/readback failure. Promote a harness for positive confirmed quality change under the shared rule, or for quality noninferiority within 0.02 and at least 10% confirmed cost reduction. Require at least 95% scoring coverage and no completion regression greater than 0.02. Inconclusive changes stay experimental. Archive the winning weights **and** exact harness/index artifacts together; preserve the prior policy+harness champion until replacement reload and regression checks pass.

## Checkpoints, safety, and required artifacts

Apply the [shared measurement, checkpoint, and budget contract](README.md). Keep Qwen/Qwen3.5-4B and the pinned 858/117 source-reading split; use the frozen 32-task selection and 85-task confirmation cohorts. Repurposed benchmark provenance and uncalibrated reference grading must remain visible. The one-update pilot demonstrated integration, not quality improvement.

Promotion requires at least **95% scoring coverage** (31/32 selection tasks and 81/85 confirmation tasks). Repeat training finalists with seeds 42, 43, and 44; select their checkpoints before inspecting confirmation results. Frozen-policy-only comparisons use the repeated-inference protocol stated above instead of claiming training-seed replication.

For any training arm, commit resumable state and sampler weights after every acknowledged GRPO update; use the frozen arm’s evaluation cadence (every six successful updates and final evaluation for current GRPO). Skips do not count as updates. Retain routine checkpoints for 48 hours and retain selection-best remote state for 14 days and preserve sampler archives/checksums; indefinite durable optimizer restoration is not established. For frozen-weight studies, retain the source checkpoint plus the complete harness variant bundle instead of inventing an optimizer checkpoint.

Exclude unresolved groups; use zero whole-group retries in the current v7 campaign; quarantine unresolved groups. Equal-reward groups contribute zero, and all-zero batches skip optimization. Stop on ambiguous optimizer outcomes and restore the last committed boundary without blindly retrying. Stop at the separately authorized total budget; the earlier $5 smoke ceiling does not authorize this experiment. Count unsuccessful attempts, graders, evaluation, storage, and retries in cost.

Persist a frozen experiment specification, resolved configurations, dataset/cohort hashes, model and grader identities, source revisions, all raw trajectories and grades, local events, cost ledger, per-task evaluation tables, checkpoint/archive manifests, and a decision report. W&B is optional. Mark unsupported capabilities as prerequisites; use only the [documented CLI](README.md#budget-stopping-and-commands), never invented flags. Report implementation readiness, live execution, and quality evidence separately.
