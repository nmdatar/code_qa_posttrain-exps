# Complete-investigation SFT study

The current archive cannot support a meaningful complete-investigation SFT experiment. The offline audit found **one distinct admissible training lineage in one repository family**, containing 154 supervised target tokens. Repeating it for hundreds of optimizer steps would not provide independent training coverage.

## Evidence and admission

Audit release: `data/sft/complete-investigations-expanded-v1/`, created without provider calls or reading evaluation answers for admission. Across 962 unique training trajectories, 304 matched the current complete grader identity. Of those, 53 had perfect factual training reward, but only 15 passed strict grading. Four complete source/native-token-verified candidates survived; all represent the same matplotlib lineage. Five strict passes had no useful verified tool prefix, and six failed the clean complete-investigation requirement. Archive copies are SHA-deduplicated before counting, and admitted examples are deduplicated by training lineage.

Admission requires all of: training split and lineage binding; matching current data/public task/rubric/grader identity; resolved strict score 1; completed termination with final stop; clean useful tool observations verified against frozen source; cited source actually read; citation file hashes and line bounds verified; exact answer/submission binding; and replay of archived native generation prefixes/token text through the student renderer. No factual-only reward is treated as answer verification. These remain automatically graded policy-generated demonstrations, not human gold. A blinded semantic audit of a stratified sample is still necessary to measure false admission, even when hashes and citations pass.

The archive audit used an initial readiness threshold of 128 lineages/eight families and failed. The proposed expanded study strengthens this to **at least 200 unique lineages across at least 15 families**, ideally 300 lineages. The preparation script now defaults to that stronger gate. The gate measures diversity, not statistical power.

## New collection phase

Use the existing sampling-only benchmark operation with **Qwen/Qwen3.5-397B-A17B** as teacher, two independent attempts on **600 distinct, family-stratified training tasks**. Use the current public task, tool protocol, source snapshot, and latest strict grader; never include private references in the teacher prompt. The benchmark's task manifest deterministically allocates training tasks across families. Keep the teacher sampled-policy identity, complete raw messages, source observations, token traces, and all grading artifacts. Do not retrospectively repair a teacher answer and retain the original proof as if it generated that repaired answer.

The cached teacher and student Qwen3.5 tokenizers have identical entire vocabularies, chat templates and real transcript rendering. Compatibility evidence is in `sft-tokenizer-compatibility.json`. This makes the existing native-proof admission route usable, but each new candidate must still pass native rendering individually. No undocumented token-conversion shortcut is needed. A changed teacher revision/tokenizer invalidates this compatibility assumption until rechecked.

Freeze one passing complete trajectory per lineage, selecting the shortest supervised native token count deterministically. Use at most 10% of examples from one family where the available admission pool permits it; if this leaves fewer than 200, collect missing diversity or report the failed gate. Validate a stratified sample of at least 60 admitted investigations against source claims, citations, and unsupported extra assertions. Record mistakes and correct the admission process on training data before freezing. Do not pad the release with repeats if teacher yield is poor. Yield is currently unknown; do not promise that 1,200 attempts will yield 200 demonstrations.

## Training and comparison

Run three independent student seeds from the same fresh Qwen3.5-4B base. For each seed, perform complete-investigation SFT with assistant-only loss on tools and final answers, a fixed preregistered learning rate, and batch size eight. With 200–300 distinct examples, one pass is 25–38 optimizer batches per seed. Use one fixed epoch initially; do not call repeated epochs additional unique data. Preserve initial and final checkpoints. Compare base versus SFT before the RL continuation to isolate SFT's effect.

Continue each SFT seed with the **same factual-reward REINFORCE task order, unique task coverage, batch/group size, rollout allowance, optimizer hyperparameters, checkpoints, and evaluation cohorts as experiment 1**. Start the RL optimizer and running baseline fresh for both paths. This tests SFT+REINFORCE against matched direct REINFORCE. Keep reward-alignment experiment 2 separate. If combining stages in a single pipeline, verify stage-local stopping counts and baseline reset; otherwise fork final SFT weights explicitly with fresh optimizer state. Final-versus-best checkpoint choice must be fixed before comparing outcomes.

Preregister the primary metric as demonstrated strict credit on the same held-out cohort. Report numerator/denominator and grading coverage, resolved-only descriptive scores, seed-level changes, family-stratified uncertainty and paired task outcomes. Shared task answers from three seeds are not three independent evaluation datasets. Keep confirmation untouched until the final selection rule and analysis are locked. Three seeds improve robustness but cannot guarantee significant gains; confidence depends on effect size and held-out family/task count.

## Cost and readiness

Offline conservative collection bounds from the recorded September 29 price snapshot, including a full 24-hour controller reservation:

| Teacher collection | Episodes | Bound |
|---|---:|---:|
| 300 tasks × 2 attempts | 600 | $994.22 |
| 600 tasks × 2 attempts | 1,200 | $1,970.20 |
| 858 tasks × 2 attempts | 1,716 | $2,809.54 |

These are reservation bounds, not invoices, and exclude subsequent SFT/RL/evaluation studies and independent semantic-review costs. Price freshness must be checked before dispatch. Most of the bound is strict grading with repair allowance. Existing ~$227 remaining in the current training allocation does not cover even the first collection bound. No paid collection or training was launched by this audit.

`prepare_complete_sft_study.py` freezes a provenance manifest/audit and fails the readiness gate in its report when the pool is insufficient; it never launches providers. New tests cover factual-reward/strict-admission separation, evaluation exclusion, grader mismatch, archive duplication, and lineage deduplication. All three passed.

## Prepared collection inputs

`configs/experiments/expanded-studies-v1/teacher-collection.json` is schema-validated and binds `teacher-tasks.json`: exactly 600 distinct training lineages across all 31 training families, allocated proportionally (7–40 tasks per family). Manifest hash: `2f41473faaeffaf84ccce5d7bed9e2d9d0917976640b7df28bcf39f5276b7ca0`. The teacher and grader price snapshot was refreshed at `2026-09-29T16:24:41.949537+00:00`; the validated bound remains **$1,970.197056**, under the proposed fresh $2,200 collection ledger cap. This is prepared input, not a launched run or budget approval. Full config hash and counts are in `sft-collection-validation.json`.

The three `sft-seed42.template.json`, `sft-seed43.template.json`, and `sft-seed44.template.json` files are intentionally non-runnable study templates, with unset data manifest hash and mandatory admission gate. They are not small placeholder training runs. The complete grader identity currently incorporates the judge-price timestamp, so final teacher release and all SFT student configs must retain the same refreshed judge snapshot; silently mixing old/new snapshots fails admission even if numerical prices agree.
