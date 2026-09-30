# Expanded direct REINFORCE, seed 43

**Completed 32 acknowledged optimizer updates over 256 distinct training tasks and 1,024 attempted trajectories in those batches. Final selection performance did not improve.** This is one seed of the direct arm, not the final three-arm comparison or untouched confirmation.

| Selection metric | Initial, step 0 | Intermediate, step 16 | Final, step 32 |
|---|---:|---:|---:|
| Full strict passes | 7/32 | 10/32 | 5/32 |
| Sum of fractional strict credit | 7 | 10 | 5 |
| Demonstrated strict credit, denominator 32 | 21.875% | 31.250% | 15.625% |
| Resolved grades | 28/32 | 30/32 | 31/32 |
| Resolved scoring coverage | 87.50% | 93.75% | 96.875% |
| Configured coverage gate met | No | No | Yes |

Full strict passes count scores equal to one; fractional strict credit would also count eligible partial answers. For this seed the credit sums happen to equal full-pass counts. Unknown grades receive no demonstrated credit, not an assertion that those answers were incorrect. The fixed final checkpoint is used; the better intermediate selection result does not replace it after observing outcomes.

Final-minus-initial demonstrated credit changed by **−6.25 percentage points**. A paired-task percentile bootstrap on the same 32 tasks yields a **95% interval of −21.875 to +9.375 points** (20,000 resamples; bootstrap RNG seed 42). This interval does not account for training-seed variability, generation/judge variability, or correlation among questions from the same repository. It does not establish population degradation or algorithm inferiority. Selection membership, cohort manifest, data identity, reward version, environment identity and scientific config hash match across all evaluations.

Letting every unresolved grade independently take any value from zero to one gives a final-minus-initial missing-grade sensitivity range of **−18.75 to −3.125 points**. Thus no assignment to the missing grades can produce an improvement on these particular 32 observed tasks, while sampling uncertainty about broader performance remains. The 28 tasks resolved at both endpoints show a mean change of −7.14 points; this availability-selected subset is secondary. Final scoring coverage passes the configured gate, but initial and intermediate scoring coverage do not.

## Training and recovery accounting

The 32 acknowledged batches each scheduled eight task groups with four attempts per task: 256 unique tasks and 1,024 attempted trajectories. Of these attempts, 1,009 received resolved training grades; 964 contributed nonzero advantages. Eight whole task groups were excluded from advantage construction, which can also exclude resolved members of an incomplete group. There were 119 zero-variance groups. Mean within-batch reward variance was 0.13254. These are learning-signal diagnostics, not measures of selection improvement.

`expanded-direct-seed43-v1` completed updates 1–26. `expanded-direct-seed43-v1-continued` restored the acknowledged step-26 checkpoint and completed updates 27–32. The original failed 27th-batch attempt produced 32 extra trajectories without an acknowledged update. They are excluded from the learned-batch/trajectory totals and retained in cost accounting. Original failure and continuation records remain distinct.

## Checkpoint, runtime and provenance

The fixed final checkpoint is `ckpt-526b475d98334d8585e99d96125eaa38` at optimizer step 32.

- Sampler: `tinker://795b8b20-8f3d-5a81-b3ac-c5ab312307a2:train:0/sampler_weights/ckpt-526b475d98334d8585e99d96125eaa38`
- Training state: `tinker://795b8b20-8f3d-5a81-b3ac-c5ab312307a2:train:0/weights/ckpt-526b475d98334d8585e99d96125eaa38`
- Metadata specifies 172,800-second remote retention. A local sampler adapter export is saved and verified in the completed archive.
- Total elapsed runtime is unavailable from event/checkpoint records, which lack run start/finish timestamps. Evaluation durations were 54.38, 66.13 and 72.01 seconds; these are not training runtime.
- The final authoritative ledger records $364.8522665304345 in reservations, including recovery. The continuation added $75.6193347557199. These are reservations, not provider invoices.

Input run JSON, checkpoint step and scientific identities were checked locally; input SHA-256 values are recorded in the machine-readable report. The original archive audit reports valid run JSON but a malformed raw ledger snapshot captured while the ledger was changing; that raw snapshot is not used for cost calculations. The continuation archive contains 1,765 files, zero JSON parse errors, and reverified SHA-256 hashes. The sampler tar, adapter configuration, and safetensors header/offset structure were checked.

[Machine-readable results](direct-seed43-results.json) · [W&B continuation](https://wandb.ai/nmdatar-harvard-university/repository-qa-training/runs/expanded-direct-seed43-v1-continued)
