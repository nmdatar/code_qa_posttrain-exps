# Preregistered expanded model-improvement study: design and feasibility

Prepared 2026-09-29 from local frozen run artifacts; no paid calls or confirmation results were inspected. This is a design recommendation, not evidence that any proposed run has launched.

## Scientific question and arms

Compare three fresh-base pipelines, with seeds 42, 43, 44 paired across arms:

A. Broader running-baseline REINFORCE, positive-coverage-v4 reward.
B. Same REINFORCE with a frozen reward alignment rule that retains factual coverage credit and penalizes verified falsehoods/unsupported claims.
C. Verified complete-investigation SFT, followed by the same positive-coverage REINFORCE as A.

A is the shared control for B and C. C measures the whole SFT-plus-RL pipeline, not the isolated effect of a reward change. Use the same task multiset, per-seed task order, base model, rank, nominal LR, budgets, rollout temperature and grader in all RL arms. Different reward scales can change the effective optimization step; preregister both reward formula and normalization rather than silently equating nominal LR with effective training strength. SFT examples must come exclusively from training lineages; freeze the demonstration manifest, sources, verification criteria, number of supervised tokens and SFT epochs before launch. Admit only complete successful investigations with grounded final answers; never pad the demonstration dataset with bad examples to reach a size target. Report total additional SFT generation and optimization costs separately.

A-versus-fresh-base improvement, B-versus-A and C-versus-A are distinct estimands. Historical GRPO is descriptive context, not a concurrent randomized control for a new GRPO-superiority claim.

## Data and evaluation

The v6 release contains 866 raw training and 121 raw development rows; the admitted operational counts are **858 train and 117 development**. The existing immutable cohort manifest has **32 selection and 85 confirmation** task IDs, covering seven repository families. Half the selection cohort is xarray. Raw row counts must not replace admitted counts. Verify current loader identities and training lineage exclusions before launch.

Use selection only for frozen diagnostic checkpoints and a predefined eligibility rule. Choose the final scheduled checkpoint for the primary analysis, avoiding best-of-many checkpoint selection. Do not look at confirmation outcomes to alter reward, hyperparameters, demonstrations, stopping decisions or selection rules. After all models and analysis code are frozen, evaluate all three seeds of all arms on all 85 confirmation tasks, using the same sampler/judge protocol. Evaluate each fresh-base seed on the same confirmation tasks for paired baseline comparisons. The 32 selection outcomes remain development evidence and must not be pooled into the primary untouched confirmation claim.

Use a preregistered one-repair judge rule, retain all raw verdicts, and report resolved coverage separately. The previous REINFORCE final coverage was 29/32; GRPO was 31/32. Missing grades contribute zero demonstrated credit, but are not proven wrong answers. Report pessimistic/optimistic missing-grade bounds and the fully resolved matched subset as sensitivity analyses, never silently drop unresolved outputs. Require at least 95% resolved coverage for an unqualified result; otherwise label measurement incomplete and use frozen, symmetric adjudication across arms. Regrading costs need their own provision.

Primary outcome: strict demonstrated credit over all 85 tasks, macro-averaged over the three seeds. Report strict successes/denominator per seed, within-seed paired task deltas, means and seed ranges. Secondary: required-fact coverage, unsupported-assertion/citation errors, answer completion, resolved coverage, tool efficiency, and per-repository metrics. Training reward and optimizer counts are mechanism diagnostics rather than evidence of generalization.

Use task-paired bootstrap intervals that preserve all arm/seed outputs for a sampled task. Provide a repository-cluster sensitivity analysis and per-family results; seven clusters make cluster uncertainty estimates fragile. Use Holm correction for the three planned A-versus-base, B-versus-A and C-versus-A claims; report unadjusted effect intervals plus adjusted inferential decisions. Three training seeds are a minimum replication check, not a high-precision estimate of run-to-run variance. Repeated outputs for one task are not independent new tasks.

## Training scale

**Preferred substantial study:** each arm and seed sees one complete admitted training pass, 858 distinct tasks x four rollouts = **3,432 RL trajectories per job**. Batch size 16 tasks means 64 trajectories for a full optimizer batch; 53 full batches plus one final 10-task/40-trajectory batch = **54 scheduled batches per job**, **486 across nine jobs**, **30,888 trajectories** total. Do not repeat six tasks to fill the last batch. The implementation must support a deterministic partial final batch or explicitly record padding, which would change the task multiset. Freeze three per-seed task permutations shared by all arms. Log eligible/contributing trajectories and acknowledged optimizer calls; failures may lower actual updates.

**Minimum credible expanded study:** a stratified fixed set of 128 admitted training tasks, identical across arms, four rollouts each =512 trajectories per job, three seeds per arm =4,608 total. Batch16 would yield only eight updates: prefer batch4 tasks/16 trajectories and **32 scheduled updates per job** for this smaller design. This is a meaningful expanded comparison with repeated seeds, but it is still a pilot for modest generalization gains; its scientific claim must be limited accordingly. A full pass and larger optimizer batch address training exposure/stability, not evaluation power.

## Precision and power limitations

At a 25% success rate, an illustrative normal-approximation 95% half-width is about 15.0 percentage points for 32 independent tasks and **9.2 points for 85**. For a paired comparison with discordant outcome probability0.30, an approximate two-sided 5%-level, 80%-power detectable effect is **16.6 percentage points at85 tasks**, before multiple-comparison correction or repository clustering. The analogous task requirement is roughly236 for10-point effects and941 for5-point effects. These are planning approximations, not exact power guarantees; actual paired correlation, errors and family structure matter.

Three seeds reduce seed-specific noise but do not turn85tasks into255independent tasks. The available confirmation set therefore cannot confidently establish a small5–10point gain merely by training longer. For a strong small-effect claim, acquire additional independently held-out, contamination-checked tasks/repositories before outcomes are inspected. Report honest uncertainty or inconclusiveness if the existing benchmark does not distinguish arms. No training batch count guarantees a positive result.

## Empirical cost and runtime basis

The archived REINFORCE run cost **$65.753980 in incremental reservations**, not invoices, for128training+64evaluation trajectories,16updates,17checkpoint pairs and24m09s wall time. The authoritative ledger prefix matches the prior GRPO archive. Its incremental components were:

| Component | Reservation |
|---|---:|
| Episode sandboxes,192 | $10.619136 |
| Policy sampling | $1.161341 |
| Judge sampling | $36.007341 |
| Optimizer calls,16 | $0.903817 |
| Checkpoint pairs,17 | $15.542857 |
| Controller | $1.519488 |

Historical episode-associated mean is **$0.248895 per trajectory**, pooled across training and evaluation. This is a planning approximation, not a worst-case guarantee or a fixed provider price. Optimizer reservations averaged$0.056489/call; checkpoint pairs reserved$0.914286 each. Large batches have more train tokens per call, so per-call cost cannot be held constant when projecting64-trajectory batches. Save periodic checkpoints (for example every16updates plus final), retaining needed evaluation artifacts; checkpoint storage was24%of prior reserved cost. Model-internal sampler saves required by training may still incur separate cost and must be included by the real estimator.

For comparable accounting, allow each job32selection diagnostic+85final confirmation tasks and each of three base seeds32selection+85confirmation tasks: **1,404evaluation episodes**. Avoid repeatedly evaluating the fresh base separately in each arm when the immutable base and seed identity are shared. More checkpoint evaluations add cost.

| Study | RL episodes | Empirical RL episode cost | Evaluation episode cost | Planning total before binding upper-bound validation |
|---|---:|---:|---:|---:|
| Minimum128tasks x3seeds x3arms | 4,608 | ~$1,147 | ~$349 | **$1,600–$2,400 plus SFT preparation** |
| Full858tasks x3seeds x3arms | 30,888 | ~$7,688 | ~$349 | **$8,100–$12,000 plus SFT preparation** |

Ranges allow some checkpoint, optimizer and runtime variation but are not spending authorizations. The original128training+64evaluation operation bound was$288.814942, about4.4times its realized reservations; new frozen configuration-specific bounds must be computed, not extrapolated as if these empirical estimates guarantee completion. Worst-case commitments can be several times these ranges. Allocate reserve for adjudication and checkpoint export. Reference prices are those frozen in the archived Sept29configuration, not a new external quote.

At unchanged rollout/judge throughput, the full study is approximately **6–10hours per job,54–90aggregate job-hours**, versus about**1–2hours per job** for the minimum design. Three truly independent concurrent slots could theoretically reduce full-study elapsed time to18–30hours, but shared Tinker/Modal concurrency and large-batch grading can remove that speedup. Re-estimate after the first fixed tranche without changing outcome-based stop rules. The old24-minute timing should not be multiplied only by optimizer update count: most additional work is rollout and grading.

## Binding budget constraint

Latest archived GRPO-ledger reservations are **$428.323433896/$655**, leaving **$226.676566104**. The previous project ceiling is **$1,000**, with other allocations and reserves retained. Neither the minimum repeated-seed design nor the full-pass design fits the remaining GRPO allocation; even the empirical minimum estimate exceeds the original entire project ceiling. A request for bigger experiments should not silently alter an explicit spending ceiling. Prepare exact frozen plans, implementation checks and conservative estimates, then obtain a concrete increased budget authorization before paid work that cannot fit. Do not launch a handful of batches under the current cap while representing them as the requested confidence-bearing study.

## Source artifacts

- `reports/reinforce-v6/comparison.json` and `RESULTS.md`.
- `configs/experiments/reinforce-v6/run.json` and `budget.json`.
- `configs/experiments/grpo-autoresearch/fixes-v1-cohorts.json`.
- `artifacts/reinforce-v6-results/artifacts/project-budget/02-direct-grpo.json` compared with the prefix in the GRPO archive.
- `configs/experiments/current-v8/validation.json` admitted loader counts and v6release raw rows.

## Concrete campaign profiles after configuration estimation

The executable planner refined the practical option to **256tasks/seed**, batch8tasks x4rollouts:32scheduled RLupdates and1,024training trajectories per job,9,216acrossninejobs. This is more substantial than the128task minimum above. The full profile uses **batch13tasks x4rollouts x66updates =858tasks exactly**, avoiding any final-partial-batch dependency; ninejobs yield594scheduled updates and30,888trajectories. Task batches differ between profiles but are held identical across arms within a profile.

Three seeds and all three RL arms remain required in either profile. Include SFT-only checkpoint evaluations as a diagnostic within armC, and budget85confirmation tasks for each arm/seed plus three fresh-base seeds after the model-selection freeze. Up to four32task selection evaluations per RLjob and12-hour controller reservations are included in the revised operation estimates.

The coordinator's configuration-derived bounds are approximately$1,742.84 per practical RLjob and$5,430perfullRLjob. Combined campaigns including teacher/demo construction, SFT-only evaluation and final confirmation need an explicit **approximately$22,000–$25,000 practical ceiling or$60,000 full ceiling**, despite empirical expected planning costs around **$3,000–$5,000 or$9,000–$13,000**, respectively. These are revised comprehensive profiles; final generated estimator artifacts govern the exact requested cap. The large gap reflects reserved worst-case token/retry/context allowances, not an expectation that the entire ceiling will be billed. Neither profile is launchable under the existing$1,000project ceiling without explicit increased funding authorization.
