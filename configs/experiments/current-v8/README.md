# Current runnable experiment suite — campaign v8

This versioned design suite superseded current-v7. **Campaign v8 uses dataset v6 and grader v7**; these numbers identify different components. The original preparation validated 43 execution configs offline and did not launch paid jobs. This is preparation-time evidence, not the execution status of every later study. See the [research progression](../../../experiments/README.md) for completed comparisons and the [historical index](../../../docs/EXPERIMENT_HISTORY.md) for retired bundles. Frozen config contents and remote bundles are unchanged.

## Common frozen conditions

- Solver: Qwen/Qwen3.5-4B, rank 8, seed 42, unchanged base unless an explicit SFT checkpoint dependency is declared.
- Dataset: repo-qa-training-claims-v6, manifest `dc5f67caa93ad0c3ca3d19e4fc4d0c18eae1e4d110e9d68d29b6a1a355310bda`; 858 train / 117 development tasks.
- Cohorts: grpo-autoresearch/fixes-v1-cohorts.json, hash `045f39a6a05d72be45f5c42dc9fa12e3fdc2838a1fa92da436ea3f478ad6382e`; unchanged 32 selection / 85 confirmation membership.
- Protocol: experimental-reference-v4, action-alias-v1, paginate-v1.
- Judge: Qwen/Qwen3.5-397B-A17B, all-claims-v7, definition-context-v1, 65,536 context / 8,192 output tokens, one mechanical repair. Training factual reward: positive-coverage-v4 with independent-factual-coverage-v3 and citation-routing fix, except the explicitly separate Q/E reward study.
- Policy limits: 8,192 context, six responses, 512 tokens/response, 3,072 output tokens total, five tools, 3,500 observation bytes, 300-second episode loop, 120-second provider timeout.
- GRPO: 16 attempted batches / at most 16 updates, eight episodes per batch, temperature 1, zero group retries. Initial/final strict selection evaluation, temperature 0. No automatic no-signal or regression stop is silently reintroduced into the latest bounded recipe. Unresolved groups are excluded; equal rewards skip updates; hard batch/runtime/spend bounds apply.
- Routine checkpoint TTL 48 hours; best retention 14 days; 95% scoring coverage still required for best-checkpoint eligibility. Finite remote optimizer retention is not a permanent archive.

## Experiments and prerequisites

| Procedure | Configs | Current readiness |
|---|---|---|
| 1 | 01-baseline, repeat, confirmation and eight throughput arms | Ready for fresh matched measurement. Historical baseline is not a completed throughput sweep. Never overlap timed throughput arms. |
| 2 | 02-direct-grpo | Fresh current-code reference arm; a new run identity, not a resume of historical experiments. |
| 3 | Four 03-lr-*-group-* arms | Fixed 2×2 LR/group-size screen; no placeholder winning LR. Rates 5e-6/1e-5, groups 4/8, batch sizes 2/1. All have a 393,216-token worst-case rollout-output allowance. Reuse the exact matching procedure-2 control instead of paying twice when its complete specification matches. |
| 4 | 04-investigation-sft and 04-sft-then-grpo | Complete-investigation adapter and fresh-optimizer handoff implemented. **Only one current-version full investigation passed admission: these configs are smoke tests, not a sufficiently populated SFT effectiveness study.** |
| 5 | 05-quality-only and 05-efficiency | Explicit verifier-tier bridge implemented and tested. Identical base initialization, recipe, evaluator and task order. Live judge calibration remains experimental. |
| 6 | Repeated control, symbols, lexical, hybrid-subword and history arms | Integrated into the actual collection runner; all are frozen-base inference. Python AST symbol lookup, bounded retrieval and archived history are implemented. Confirmation variants are available for locked finalists only. |

Procedure 4's existing 30-example tool-prefix release now also has current-campaign configs: `04-tool-only-sft.json` and `04-tool-sft-then-grpo.json`. It is a different, completed experiment whose quality declined; do not silently call it full-answer SFT or a selected winner. The new full-investigation admission requires an exact current strict-passing judge identity/rubric, train lineage, clean observed tool actions, valid read-backed citations and native token alignment. No gold text is inserted into learner messages. See [SFT admission](investigation-sft/admission.json). More current strictly passing trajectories must be collected before a meaningful full SFT comparison. `04-collect-demonstrations.json` prepares four fresh base-policy attempts for each of the 858 training tasks, with no optimizer calls. It has its own 24-hour controller limit and must be packaged separately. After downloading its trajectories into artifacts, rerun `PYTHONPATH=. .venv-eval/bin/python scripts/prepare_current_experiments.py --output configs/experiments/current-v9` to freeze newly admitted examples and fresh run identities. No paid collection has been launched.

Procedure 5 uses actual strict-verifier tiers, not a threshold on factual reward. Failed=0; partial=0.2×supported coverage; accepted quality-only=0.9; accepted efficiency=0.9+0.1×clip(1−compute/budget,0,1). Compute=input tokens+2×output tokens+100×tool seconds, using trusted recorder metrics and the fixed per-task compute budget. Unresolved assessments remain null. Development still uses strict quality evaluation, never the shaped reward. Semantic acceptance is model-assessed and is not labeled human-calibrated.

Procedure 6 isolates three comparisons: control/symbols, lexical/hybrid-subword, and hybrid-subword/history. All retrieval arms have symbols fixed on; history fixes hybrid-subword retrieval. This prespecifies each comparison rather than assuming an earlier winner. The hybrid is explicitly **lexical overlap plus char-trigram-hash-256-v1 vectors**, not learned semantic embeddings. A pretrained neural embedding comparison remains a separate follow-up. Python references are AST identifier occurrences, not language-server binding resolution. Non-Python navigation and execution probes remain outside this source-reading intervention.

Retrieval splits only pinned repository source into 12-line chunks and returns at most 512 actual solver-tokenizer tokens, including observation metadata, under the existing byte cap. [The source/index manifest](source-index-manifest.json) pins inventories, algorithm version and parameters; query text is used only at retrieval time. There is no external embedding call or private-rubric access. Index/scoring work is performed per request and is included in tool time; no amortized cached-index speedup is claimed. History compacts old observations into source handles, keeps the newest observation intact, archives original observations per episode, and supplies bounded readback. Generated actions and sampled token/logprob records remain intact.

## Budgets and launching

[budget-plan.json](budget-plan.json) contains explicit proposed per-arm caps, computed from conservative operation+controller estimates plus a rounded 10% allowance. The old $1,000 allocation/history is retained in the aggregate plan. New run ledgers are distinct; no historical ledger is reset or cap changed. The prior hardcoded $1,000 ceiling in the packaging code is replaced with validation of the supplied explicit ceiling. Funding is no longer used to shorten or mismatch the arms. Preparation is not submission, and the plan is not a provider invoice. Recheck live prices and reconcile remote reservations at dispatch.

Some configs are optional alternatives or finalist-only confirmation controls. Do not run all 43 merely because they exist. Freeze selection decisions before any confirmation use. Comparison campaigns use four-hour controller limits (the separate demonstration collection uses 24 hours); group small numbers of jobs per campaign so sequential schedules fit that bound.

```sh
# Offline validation
.venv-eval/bin/python -m training_pipeline validate --config configs/experiments/current-v8/05-efficiency.json

# Offline frozen packaging of independent LR arms (no provider call)
.venv-eval/bin/python -m training_pipeline.remote prepare \
  --configs configs/experiments/current-v8/03-lr-5e-6-group-4.json configs/experiments/current-v8/03-lr-1e-5-group-4.json \
  --parallel-training --budget-plan configs/experiments/current-v8/budget-plan.json \
  --output artifacts/current-v8-lr-bundle
```

SFT→GRPO is a sequential `fork` from the new SFT run's coverage-eligible `checkpoints/best.json` with fresh optimizer state. Packaging rejects dependency-containing parallel campaigns. If no eligible checkpoint exists, the handoff must stop; it never silently loads last weights or an expired historical adapter. The initial base can remain selection-best, so inspect checkpoint step before interpreting any SFT effect.

[Validation/estimates](validation.json) and [study plan](study-plan.json) separate software readiness from live validation, data sufficiency and scientific outcomes. The broader REINFORCE, PPO/learned-reward, neural retrieval, runtime-probe, annotation and 9B roadmap remains deferred.
