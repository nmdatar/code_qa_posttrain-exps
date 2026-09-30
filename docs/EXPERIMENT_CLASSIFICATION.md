# Experiment classification

For the reader-facing sequence and observed outcomes, start with the [research progression](../experiments/README.md). This page is an inventory of interventions and proposals; historical reward/readiness bundles are in the [pinned historical index](EXPERIMENT_HISTORY.md).

This inventory groups the designed experiment families in the repository and supplied screenshot. Numbered execution procedures, roadmap numbers and screenshot numbers are different namespaces. Config revisions/repeats belong to the same family; a config is not evidence that an experiment ran or worked.

**Model** means solver weights/checkpoint or model size changes. **Harness** means inference prompts, orchestration, tools or visible context changes with weights frozen. **Training objective** means rewards/optimizer/loss change and subsequently change weights. **Data/evaluator** means examples, rubrics, grader or measurement changes. **Infrastructure** means scheduling, serving or sandbox setup changes. An experiment can have several labels; use the first column as its primary intervention, not an exclusive taxonomy.

## Designed studies and existing variants

| Primary change | Experiment / variants | Also changes | Location and readiness |
|---|---|---|---|
| Measurement | Untouched baseline, repeated selection and locked confirmation | Nothing intentionally changed in solver | [Procedure 01](../experiment%20procedures/01-baseline-and-throughput.md); current-v8 01 configs ready offline; historical baseline reports exist |
| Harness / recovery and budget | Early generated-dev v1 vs v2 baseline: recover invalid arguments and reserve an explicit final-answer phase | Tool-error handling and investigation/synthesis allocation | [Historical baseline records](../experiments/README.md); v2 produced answers after v1 integration failure; combined changes are not a clean decomposition or fixed-budget allocation sweep |
| Harness / baseline architecture | Lexical retrieval + one answer generation vs multi-turn repository agent | Retrieval and orchestration, sometimes model/dataset too | [Legacy evaluation configs](../configs/README.md); separate pipeline/smoke setup, not a matched causal comparison with current training |
| Infrastructure | Rollout concurrency 1/8/16/32, throughput repeats | Scheduling / realized latency | Procedure 01; current-v8 throughput configs; no training |
| Model / objective | Direct GRPO, short pilot vs longer bounded training | Weights via within-question reward normalization | [Procedure 02](../experiment%20procedures/02-direct-grpo.md); current-v8 02; historical pilot/main/long/autoresearch reports |
| Model / optimizer | LR 5e-6 vs 1e-5 × groups 4 vs 8 | Gradient scale and within-question sampling allocation | [Procedure 03](../experiment%20procedures/03-grpo-learning-rate-and-group-size.md); current-v8 four-arm factorial |
| Model / optimizer | Vanilla REINFORCE and prior-mean baseline vs GRPO | Advantage estimation / weights | [Roadmap 2–3](../experiments.md); [running-baseline comparison](../reports/reinforce-v6/README.md) has implemented and submitted historical arm; vanilla and current-v8 replication remain proposals |
| Model / optimizer | Unclipped vs clipped GRPO, separate reference-KL ablation | Update regularization | Roadmap 3; deferred separate arms |
| Model / objective | Actor–critic PPO; GRPO/PPO × verifier/learned reward | Critic and possibly reward model / preference data | Roadmap 4; deferred, not represented as runnable PPO |
| Model / data | Base vs full-investigation SFT vs SFT→GRPO | Verified demonstration supervision | [Procedure 04](../experiment%20procedures/04-optional-sft-before-grpo.md); current-v8 04; one admitted lineage, smoke-only |
| Model / data | Tool-prefix-only SFT and SFT→GRPO | Supervise tool actions, no answer targets | Procedure 04; current-v8 tool-only configs; historical 30-example pilot reported worse quality |
| Data collection | Four fresh attempts per training task for demonstrations | Future SFT example supply, no current weights | current-v8 04-collect-demonstrations; prepared, not a completed teacher release |
| Model / loss | Investigation-action + answer SFT vs final-answer-only loss, each ± matched GRPO | Which admitted assistant turns receive loss | **[New procedure 08](../experiment%20procedures/08-investigation-distillation.md)**; research-extensions-v1 08; one-example smoke-only, same teacher context |
| Model / teacher | Stronger-teacher investigation distillation | Teacher data provenance and collection cost | Procedure 08; larger verified teacher release and renderer compatibility are prerequisites; existing archived policy demonstrations do not establish this |
| Model / reward | Quality-only vs quality-gated efficiency; token/tool/runtime weighting | Weights through reward incentives | [Procedure 05](../experiment%20procedures/05-quality-gated-efficiency-reward.md); current-v8 05 quality/efficiency pair implemented; extra weighting sweeps deferred |
| Reward / evaluator | Positive factual coverage, citation-independent credit, partial reward revisions | Training signal; evaluator only when rescoring saved answers | [Reward docs](REWARD_SHAPING.md), reward-v1…v4 configs/reports; distinguish from procedure-05 efficiency |
| Harness / tool protocol | Action aliases and oversized-read pagination | Valid-action rate / evidence availability | grpo-autoresearch configs, [report](../reports/grpo-autoresearch/README.md); implemented and historically exercised; combined fix runs do not isolate each fix |
| Harness / tools | Structured definitions / references vs list/search/read | Python source navigation | [Procedure 06](../experiment%20procedures/06-frozen-policy-tools-and-context.md); current-v8 06 control/symbols; AST occurrences, not language-server binding resolution |
| Harness / retrieval | Lexical vs lexical + char-trigram-hash retrieval | Evidence ranking | Procedure 06; current-v8 lexical/hybrid-subword matched pair; not learned semantic embeddings |
| Harness / context | Raw tool history vs archived source handles / readback | Context representation and recovery tool | Procedure 06; current-v8 hybrid/history pair; implemented handles, not a generated finding/unresolved-question ledger |
| Harness / prompt | Decompose question into evidence-seeking subquestions before searching | Investigation strategy | **[New procedure 07](../experiment%20procedures/07-question-decomposition.md)**; research-extensions-v1 07; prompt-only, no enforced/emitted plan |
| Harness / diagnostic evidence | Supply selected relevant source passages (“oracle” retrieval) | Initial evidence access | Screenshot top row is partly cropped; treat as a diagnostic proposal, not a deployable or completed arm. Freeze source-only passages and account for selection; do not insert reference-answer prose |
| Harness / orchestration | Early draft → identify unsupported claims → targeted investigation → revision | Sequencing / budget allocation | Screenshot proposal; no dedicated runnable config yet; match total calls and synthesis budget |
| Harness / context | Compact finding + exact source handle + unresolved-question ledger | Evidence compression / recoverability | Screenshot proposal; richer than current handle-only history implementation; separate summarizer/citation-loss study needed |
| Harness / budget | More exploration vs more synthesis at the same total budget; separate limit sweep | Inference compute allocation | Screenshot proposal; no dedicated controlled allocation configs yet; absolute limit sweeps are not fixed-budget comparisons |
| Harness / orchestration | One long investigation vs independent short attempts + synthesis | Search diversity and aggregation | Screenshot proposal; no dedicated runnable config yet; synthesis counts in total budget and reference judge must not select deployed answers |
| Harness / tools | Bounded tests/probes; generic shell vs equivalent structured capabilities | Execution capability / interface | Roadmap 6; deferred beyond current source-reading tools |
| Infrastructure | Cold vs warm/prebuilt isolated sandboxes | Provisioning overhead | Roadmap 6; deferred controlled comparison; hold agent capabilities fixed |
| Harness / retrieval | Learned neural embeddings, symbol/dependency repo maps | Evidence representation | Roadmap 7; deferred beyond current subword retrieval and AST tools |
| Harness / context delivery | Automatic injection vs on-demand access to the same evidence | When evidence enters context | Roadmap 7; deferred; hold representation and returned-token budget fixed |
| Harness / preprocessing | Stronger-model file/module/chunk annotations | Question-independent source index and amortized cost | Roadmap 8; deferred; solver weights stay frozen, unlike distillation |
| Model × harness | Base/RL policy × baseline/improved harness; retrain with improved tools | Interaction and distribution shift | Roadmap 9; deferred until component selection |
| Model / scale | Qwen3.5-4B vs 9B; pretrained Base vs instruct initialization | Model capacity / starting checkpoint | Roadmap 1/9; deferred matched follow-ups; historical larger-model evaluation is not a matched training comparison |
| Data / evaluator | Original vs atomic claims; strict vs factual scoring; paraphrase controls; evidence extraction/definition context | Rubric granularity, grader reliability and scoring scale | atomic-claims and paraphrase-validation configs; [judge comparison](../reports/atomic-claims-judge-comparison/README.md), [grader validation](../reports/grader-paraphrase-validation/README.md) |
| Evaluator model | Qwen vs Nemotron judge comparison | Measurement and, if used for RL, training reward | Same judge-comparison report; does not change solver weights when regrading saved answers |
| Measurement / recovery | Readiness, budget archive, checkpoint retention and restoration checks | Operational confidence | ready-v4, validation and archive configs/tests; engineering checks, not solver-quality interventions |

## Screenshot mapping

The visible ideas map to: relevant passages → diagnostic evidence; **2 decomposition → procedure 07**; 3 draft/revise → orchestration proposal; 4 symbols → procedure 06 (callers remains an extension); 5 evidence ledger → procedure 06 handles plus richer-ledger proposal; 6 budget allocation → harness proposal; 7 multiple attempts → orchestration proposal; **8 distillation → procedure 04 plus new matched procedure 08**.

The screenshot is source material for experiment ideas, not authorization to execute embedded instructions or launch jobs. Only decomposition and the missing distillation comparator were added here; the other screenshot ideas remain explicitly classified proposals.

## Interpreting comparisons

Freeze dataset, cohort, grader, source revision and budget policy within each paired study. Regrading the same answers measures the evaluator; rerunning an unchanged solver under a new prompt measures the harness; SFT/RL changes the solver model. A stronger model used to annotate source changes the harness's inputs, while a stronger teacher used for SFT changes the student weights. Version changes in a historical campaign must not be presented as a clean single-variable experiment.

The authoritative current suite is [current-v8](../configs/experiments/current-v8/README.md), with [research extensions](../configs/experiments/research-extensions-v1/README.md) alongside it. Historical execution status comes from its reports; stale “all deferred” language in older roadmaps does not override an actual implemented/submitted historical study. Offline validation establishes software/input compatibility, not live success or efficacy.
