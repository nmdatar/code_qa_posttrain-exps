# Qwen397 claim-grader validation

Qwen/Qwen3.5-397B-A17B is now selected in the prepared experiment configs. It substantially improved the diagnostic judgments, but exact-quote formatting failures leave scoring coverage below the readiness gate.

## Controlled comparison

Fifteen calls reused the **identical fixture file and hash** from the Nemotron claim-grader diagnostic. Candidate text, source excerpts, reference claims, weights, expected ranges, order, prompt, temperature (0), context limit (16,384), output limit (1,024), and deterministic scoring were unchanged. The model and its required renderer/tokenizer changed to Qwen397 with `hf-chat-no-thinking-v1`. Thus this tests a model/template replacement, not a model-only change under an identical tokenizer.

No retry, prompt tuning, training, rollout collection, or checkpoint saving was performed. Saved requests were checked against the frozen fixtures and accepted scores independently recomputed from the raw claim responses.

| Type | scikit-learn | SQLFluff | SymPy |
|---|---:|---:|---:|
| Strong | 1.0 | Rejected: shortened quotation | 1.0 |
| Partial (each repeated once) | Both rejected: apostrophe substitution | 0.333 both times | 0.333 both times |
| Wrong | 0 | 0 | 0 |
| Wrong plus grading injection | 0 | 0 | 0 |

All six plainly wrong/injection cases received zero. All twelve accepted grades fell within the predeclared rubric-specific ranges. This is a small known diagnostic set with assistant-authored expectations, not independently calibrated accuracy. The SQLFluff partial answer still received full credit on a compound inheritance/type claim despite mentioning only the type; the overall score remained within the diagnostic's broad range. More atomic rubrics and finer claim-level review are still needed.

## Remaining formatting failures

Two identical scikit-learn partial cases used straight apostrophes in a quote where the candidate used curly apostrophes. One SQLFluff strong case abbreviated a quote with `...`. The strict quote-presence check rejected all three and preserved null rewards. Their unaccepted verdicts looked substantively reasonable, but **they were not counted as successful grades**. No parser relaxation or retrospective score repair was applied.

Valid scoring coverage is therefore **12/15 = 80%**, below the required 95%. Select Qwen397 for further validation, but fix the quote-output protocol and test fresh cases before a research baseline or training run. It is not yet a calibrated judge.

## Cost and configuration

Elapsed time: **32.67 seconds**. Conservative token reservations: **$0.227856**, below the $1 diagnostic cap. Actual provider billing remains unavailable. Pricing was refreshed before execution: $3/million input tokens and $7.50/million output tokens.

Prepared configs under `configs/experiments/`, the generic training example, and the new `examples/training-repository-qwen397.json` now pin this model/renderer and `source-claims-v1`. Historical Nemotron examples and artifacts remain for comparison. New run/output identities avoid overwriting Nemotron results. Spending caps and shared ledger paths were preserved.

The higher judge price changes launch estimates: direct GRPO now reserves about **$961.10** against its **$650** cap; each LR screen about **$122.90** against **$80**. Those runs are blocked by their existing budget checks. Each baseline invocation has an upper estimate of **$11.17** (conservatively pricing 85 tasks), and each throughput invocation **$16.83**, including reserved retries. The complete three-baseline/eight-throughput schedule sums to about $168.15, exceeding the shared $120 baseline allocation. Individual runs fitting a cap do not mean the whole schedule fits. No budgets were increased and no experiments were launched.

Artifacts: `artifacts/grader-diagnostic-qwen397-v3/` contains fixtures, configs, estimate, raw/parsed responses, results, summary, and ledger. Updated cost estimates are in `reports/qwen397-config-cost-audit.json`.
