# Expanded direct REINFORCE, seed 44

**Completed 32 acknowledged updates on 256 distinct training tasks, with 1,024 training attempts. This seed did not demonstrate better selection performance.**

| Metric | Initial, step 0 | Intermediate, step 16 | Final, step 32 |
|---|---:|---:|---:|
| Full strict passes | 9/32 | 7/32 | 8/32 |
| Sum of fractional strict credit | 9⅔ | 7 | 8 |
| Demonstrated strict credit | 30.208% | 21.875% | 25.000% |
| Resolved grades | 30/32 | 31/32 | 30/32 |
| Configured scoring-coverage gate met | No | Yes | No |

Demonstrated credit uses denominator 32, including fractional credit for eligible partial answers and no demonstrated credit for unresolved grades. Full strict passes count only scores of one. Unknown grades are not proven incorrect answers.

Final-minus-initial change is **−5.208 percentage points**. The paired-task bootstrap 95% interval is **−21.875 to +11.458 points**, using 20,000 resamples and RNG seed 42. Allowing missing grades any values from zero to one gives a change range of **−11.458 to +1.042 points**. Among 28 tasks resolved at both endpoints, the change is −5.952 points. These are selection diagnostics for one seed, not untouched confirmation or evidence of superiority/inferiority. Both endpoints fail the configured scoring-coverage gate.

All evaluation task IDs, cohort manifest, data identity, environment, reward version and scientific config hash match within this run. The report uses the fixed final checkpoint, without choosing an intermediate outcome after seeing its score.

## Training and checkpoint

Each of 32 acknowledged batches scheduled eight tasks with four attempts each. There were 1,008 resolved training trajectories and 968 trajectories with nonzero advantages contributing to optimization. Eight whole groups were excluded, and 121 groups had zero reward variance. Mean within-batch reward variance was 0.115821. All 32 optimizer updates were acknowledged. No interrupted continuation or abandoned batch appears in this seed's recorded events.

Final checkpoint: `ckpt-3eb1a5bfe97745b9ba9b3a691b242e1b`.

- Sampler: `tinker://3eb58ae5-f25f-5fc6-bc8f-21514da03a24:train:1/sampler_weights/ckpt-3eb1a5bfe97745b9ba9b3a691b242e1b`
- Training state: `tinker://3eb58ae5-f25f-5fc6-bc8f-21514da03a24:train:1/weights/ckpt-3eb1a5bfe97745b9ba9b3a691b242e1b`
- Final-checkpoint metadata specifies 172,800-second remote retention. Its manifest and evaluation are archived, but no local sampler or optimizer-weight export exists for this fixed final checkpoint.

The archived authoritative private ledger verifies **$343.0981421804351 in reservations**, not provider invoices. Total run runtime is unavailable from timestamp-free events/checkpoints; evaluation durations alone (51.11, 76.65, 99.13 seconds) do not establish training runtime.

The completed archive contains **8,339 files**, zero JSON parse errors, and all hashes reverified. The verified `sampler.tar` is **step 16 checkpoint `ckpt-b9292bdde8ec4386a94da77a1d42049e`**, the best checkpoint that met the selection coverage gate. It is **not** the fixed final step-32 checkpoint reported above. Step 16 had 21.875% credit and 31/32 resolved grades; final step 32 had 25% credit but failed the coverage gate. The best-checkpoint metadata specifies extended 14-day remote retention and unsupported archive import to Tinker. The tar contains adapter configuration, adapter weights and a completion marker, and its SHA-256 matches the checkpoint archive metadata. This local best-checkpoint sampler export does not preserve the fixed final weights or optimizer state.

[Machine-readable results](direct-seed44-results.json) · [W&B](https://wandb.ai/nmdatar-harvard-university/repository-qa-training/runs/expanded-direct-seed44-v1)
