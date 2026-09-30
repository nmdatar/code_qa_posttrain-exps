# Expanded direct REINFORCE, seed 42

**Completed 32 acknowledged optimizer updates over 256 distinct training tasks and 1,024 attempted trajectories in those batches. This seed did not demonstrate improved selection quality.** It is one arm/seed of the expanded study, not the final three-arm comparison or untouched confirmation.

| Selection metric | Initial, step 0 | Intermediate, step 16 | Final, step 32 |
|---|---:|---:|---:|
| Full strict passes | 7/32 | 6/32 | 6/32 |
| Sum of fractional strict credit | 7⅔ | 6 | 7 |
| Demonstrated strict credit, denominator 32 | 23.96% | 18.75% | 21.88% |
| Resolved grades | 28/32 | 31/32 | 29/32 |
| Resolved scoring coverage | 87.50% | 96.88% | 90.63% |
| Configured coverage gate met | No | Yes | No |

Full strict passes count only scores equal to one. Fractional strict credit also counts eligible partial answers. Unknown grades receive no demonstrated credit; this is not a finding that their answers were wrong. The initial and final evaluations did not satisfy the configured scoring-coverage eligibility gate, so these results do not justify promotion.

The final-minus-initial change is **−2.08 percentage points**. A paired-task percentile bootstrap over the same 32 tasks gives **95% interval −15.63 to +11.46 points** (20,000 resamples, RNG seed 42). This interval captures task resampling only, not training-seed variation, stochastic generation/judging, or correlation between tasks from the same repository. It is not evidence of equivalence or a reliable degradation claim. All selection task IDs, cohort manifest, data identity, reward version, environment identity and scientific config hash match between the three evaluations.

Allowing every missing grade to range independently from zero to one gives a final-minus-initial sensitivity range of **−14.58 to +7.29 points**. On the 26 tasks resolved at both endpoints, the mean change is −6.41 points; that subset may be selected by grading availability and is secondary.

## Training and recovery accounting

The run used 32 batches of eight distinct task groups, with four attempts per task: 256 tasks and 1,024 attempts associated with acknowledged updates. Of those attempts, 1,017 received resolved training grades and 975 contributed nonzero advantages to updates. Five complete task groups were excluded from advantage construction; the whole-group rule can exclude otherwise resolved members. There were 133 zero-variance groups, which a running baseline can still use. Mean within-batch reward variance was 0.12377; this is a training diagnostic, not the selection outcome.

The first two updates came from `expanded-direct-seed42-v2`; `expanded-direct-seed42-v2-continued` restored its acknowledged step-2 checkpoint and completed updates 3–32. The abandoned third-batch attempt in the original failed run has 32 extra trajectories. They are excluded from the reported learned-batch/trajectory totals because no optimizer update was acknowledged for them. They must remain in financial accounting. The original and continuation records remain distinct, preserving the infrastructure failure and recovery history.

## Checkpoint, runtime and provenance

The fixed final checkpoint is `ckpt-fc0b245037464f3a937d57c737deb7bc` at optimizer step 32, not the best intermediate selection checkpoint.

- Sampler: `tinker://33ea3593-23f7-5639-b2d5-ae69e5bb0249:train:1/sampler_weights/ckpt-fc0b245037464f3a937d57c737deb7bc`
- Training state: `tinker://33ea3593-23f7-5639-b2d5-ae69e5bb0249:train:1/weights/ckpt-fc0b245037464f3a937d57c737deb7bc`
- Checkpoint metadata specifies 172,800-second remote retention. This report does not certify a permanent weight export.
- Total elapsed runtime is unavailable from the downloaded event/checkpoint records, which lack run start/finish timestamps. The three evaluation wall times were 47.28, 57.29 and 71.82 seconds; summing them is not training runtime.
- The authoritative private ledger ended at $376.22296373843363 in reservations, including the original v2 attempt, three diagnostics, and continuation. Continuation plus its controller added $321.8180274565767 above the $54.404936281856964 pre-continuation balance. These are reservations, not provider invoices.

Input JSON, checkpoint step and scientific identities were checked locally. The machine-readable report includes input file SHA-256 values. The completed archive contains 7,681 files, with zero JSON parse errors and all local SHA-256 hashes reverified. No sampler weight archive was present; remote checkpoint references are preserved, but a permanent weight export is not certified.

[Machine-readable results](direct-seed42-results.json) · [W&B continuation](https://wandb.ai/nmdatar-harvard-university/repository-qa-training/runs/expanded-direct-seed42-v2-continued)
