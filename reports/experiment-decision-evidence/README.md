# Evidence behind the experiment decisions

Collected from the local experiment archive on September 29, 2026. Charts are newly recreated from saved measurements, not screenshots of a refreshed live dashboard. This packet connects the recorded motivation, intervention, and observed outcome; it does not infer personal intent beyond the written experiment records.

## 1. Why split compound claims?

The motivating saved answer concerns xarray's `IndexVariable`, task `import-c6fc34340a25ce91bf221303`.

**Original rubric:** “Name setter blocks assignment and level extraction returns new variable.”

**Revised rubric:** two individually assessable facts: “Name setter blocks assignment” and “level extraction returns new variable.”

The answer correctly described a setter that raises `AttributeError`, `get_level_variable()` returning a new `IndexVariable`, and a `ValueError` when `level_names` is `None`. Its source citation covers `xarray/core/variable.py:2903–2916`.

Exact excerpt from the saved answer:

> When operations are applied to non-MultiIndex data, get_level_variable() raises a ValueError (line 2906) if level_names is None, indicating the absence of a MultiIndex.

The original Qwen judgment treated omission of `self.` in this prose as a substantive error. It said that `self.level_names` “is a property call, not a local variable or argument named 'level_names'.” Thus a valid explanation received only partial coverage and a material-error flag. The source actually checks `if self.level_names is None` and raises the stated error; the prose was a reasonable reference to that property.

With the split rubric, the same saved answer and source received complete coverage for both required facts. The additional assertion about `level_names` was also correctly supported. **Training reward changed 0.50 → 1.00; strict score changed 0 → 1.** The model's answer did not improve in this comparison; the measurement did.

| Frozen answer/control | Original Qwen reward | Atomic Qwen reward |
|---|---:|---:|
| Correct cited answer | 0.50 | 1.00 |
| Same correct text without citations | 1.00 | 1.00 |
| One correct fact | 0.50 | 0.50 |
| One correct fact plus one wrong fact | 0.50 | 0.50 |
| Wrong answer | 0.00 | 0.00 |
| Saved PyBryt partial answer | 0.50 | 0.333 |
| Saved abstention | 0.00 | 0.00 |

Separating facts makes missing coverage and partial credit inspectable: one of two supported facts can earn half credit without conflating independent requirements. It also exposed why aggregate scores alone are insufficient. PyBryt's 1/3 arose from two half-covered claims, not the expected one fully covered claim; paraphrase sensitivity remained. Mixed answers still receive factual partial credit, with falsehoods handled separately in strict grading. Splitting is not a blanket reward increase.

The migration reviewed 987 tasks, expanding 1,208 claims to 1,801; 344 tasks gained claims. Relations and conditions that would change meaning if split were retained. The evidence is a small diagnostic (seven correlated answers across three tasks, two rubrics, two judges), not a general accuracy benchmark. The earlier [SQLFluff diagnostic](../grader-diagnostic-qwen397-v3.md) had also shown the reverse problem: a compound inheritance/type claim got full credit when only the type was mentioned.

**Evidence:** [extracted answer, pinned source lines, and both full grades](claim-example.json); [original comparison report](../atomic-claims-judge-comparison/README.md); [all claim-level decisions](../atomic-claims-judge-comparison/claim-decisions.csv); [parent-to-child claim mapping](../atomic-claims-judge-comparison/claim-migration.csv).

## 2. Why move from GRPO to running-baseline REINFORCE?

![GRPO and REINFORCE comparison](grpo-vs-reinforce.png)

The operational motivation was **too little within-question reward variation**. GRPO centers rewards within a group of answers to the same question. When every answer has the same reward, every centered advantage is zero, so the group supplies no update signal. In the longer run, **17/32 groups** had zero variance and only **60/128 trajectories** contributed. Five of sixteen scheduled batches produced no optimizer update.

The replacement used `advantage = reward - mean(eligible rewards from earlier batches)`, initially zero, without per-question centering or standard-deviation normalization. Equal-reward groups can therefore contribute when their reward differs from the historical baseline. This does not create positive credit for failures: zero-reward answers can receive negative advantages when the baseline is positive. If both reward and baseline are zero, no signal is created.

A concrete saved comparison: batches **2 and 3** had mean reward **0 in both runs**. GRPO recorded **0 contributing trajectories** in each batch. REINFORCE recorded **8** in each, because its prior baseline was positive.

| Measurement | GRPO | Running-baseline REINFORCE |
|---|---:|---:|
| Training attempts | 128 | 128 |
| Acknowledged updates | 11 | 16 |
| Contributing trajectories | 60 | 122 |
| Mean training reward | 0.21224 | 0.22161 |
| Initial → final strict passes, same 32 selection tasks | 8 → 6 | 7 → 9 |
| Initial → final demonstrated strict credit | 25% → 18.75% | 21.875% → 28.125% |
| Initial → final resolved grades | 28 → 31 | 29 → 29 |

This supported choosing REINFORCE for further experiments because it used substantially more of the sampled data. The held-out point estimate was encouraging, but not conclusive: the difference in changes was **+12.5 percentage points**, with paired 95% bootstrap interval **[−12.5, +37.5]**. Both runs used fresh Qwen3.5-4B, rank 8, seed 42, LR 1e-5, matching training-task multiplicities and evaluation membership, and the same v6 rubrics/v7 grader. Different advantage scales mean equal nominal learning rates do not guarantee equal effective update sizes. Initial evaluations differed despite fresh base initialization; this was a sequential single-seed recipe comparison. Unresolved grades contribute no demonstrated credit, but are not proven wrong answers. Confirmation was untouched.

**Evidence:** [comparison data](grpo-reinforce-comparison.json); [GRPO batch CSV](grpo-reward-curve.csv); [REINFORCE batch CSV](reinforce-reward-curve.csv); [original results and limits](../reinforce-v6/RESULTS.md). Archived dashboard links: [GRPO](https://wandb.ai/nmdatar-harvard-university/repository-qa-training/runs/grpo-long-v6-seed42-v1), [REINFORCE](https://wandb.ai/nmdatar-harvard-university/repository-qa-training/runs/reinforce-v6-seed42-v1).

## 3. Which failures motivated SFT before RL?

The earlier `autoresearch-grpo-paginate-v2-seed42` run had **54 action-validation errors across 80 training episodes**, **15 budget exhaustions**, and only **one actual tool-execution failure**. The problem was primarily producing usable actions and finishing an investigation. Errors can occur multiple times per episode; 54/80 is not an episode failure rate.

Recounting the original trajectories gives 19 malformed/non-object JSON actions, 21 invalid-tool errors, eight citations to unobserved files, four unsupported-argument errors, and two paths outside the pinned source catalog.

Exact model outputs and observed errors:

**Malformed quoted string** (`ep-053ad9019d2f422b8e44cadfee8d02f5`):

```text
{"tool":"search_code","arguments":{"query":"hook,"glob":"*order*.py"}}
```

Harness response: `Expecting ',' delimiter: line 1 column 51 (char 50)`. The closing quote after `hook` is missing. The same episode then produced a final JSON object followed by a period (`..."citations":[]}.`), causing `Extra data: line 1 column 162 (char 161)`. The episode exhausted its budget.

**Missing tool name** (`ep-2a638ae94f3e473795983d0d5c87fdd4`):

```json
{"arguments":{"query":"_Unset","glob":"*.py"}}
```

Harness response: `Invalid tool`. The object has arguments but does not specify which tool should execute them. This episode also exhausted its budget. The pattern audit additionally found eight errors using `action` instead of `tool`; that alias problem motivated a harness fix, while absent tool names could not safely be guessed.

**Arguments for the wrong tool** (`ep-800100c3b2244384b16eb9ba3ad59be3`):

```json
{"tool":"search_code","arguments":{"glob":"*.py","query":"format","end_line":5000,"start_line":1}}
```

Harness response: `Missing or unknown tool arguments`. Line bounds belong to file reading, not this search interface.

**Ungrounded citation** (`ep-252944b559b44eee8d0204fb233e5751`): the model submitted citations to `numba/core/annotations/type_annotations.py:223–229` and `numba/core/annotations/pretty_annotate.py:184`, then received `Citation must refer to an observed file`. A syntactically valid final answer still has to bind its citations to observed source.

**Why SFT first:** terminal reward is an indirect way to teach punctuation, the action envelope, valid arguments, evidence gathering, and stopping. Verified demonstrations provide direct token-level targets for those behaviors; RL can then optimize complete investigations using outcome rewards. This is the experiment rationale, not a proved causal explanation for all errors.

![SFT formatting and task-quality evidence](sft-format-vs-quality.png)

The initial pilot supervised only 69 successful tool actions from 30 task prefixes, for four updates. On a fixed-context temperature-1 probe, valid actions rose **109/128 → 121/128**, and malformed/non-object JSON fell **8 → 2**. But within-run end-to-end strict passes fell **5/32 → 3/32**, and completions fell **32 → 28**. The after-SFT model used all five tool calls on every task; final-answer supervision had been excluded. The recorded decision was **not to initialize RL from that checkpoint**.

That result motivated **complete-investigation SFT**: teach search/read actions, source-supported final answers, citations, and stopping together, then compare **SFT + REINFORCE against matched direct REINFORCE**, with fresh RL optimizer and running baseline. The historical SFT pilot report initially proposed a GRPO continuation; the later expanded-study design specifies REINFORCE. Neither the formatting probe nor the negative tool-only pilot establishes that SFT + REINFORCE wins.

The broader SFT design requires at least 200 distinct admitted demonstrations across 15 families. The [archived billing-stop report](../expanded-studies/BILLING-STOP.md) records partial teacher collection and an unmet admission gate, not a completed SFT+RL success. This packet does not refresh live campaign status.

**Evidence:** [all 54 extracted errors with exact text and original trajectory paths](action-errors.json); [original 80-episode summary](../grpo-autoresearch/training-pagination-summary.json); [tool-pattern audit](../grpo-autoresearch/invalid-tool-patterns.json); [SFT results](../sft-tool-warmup-v1/RESULTS.md); [recounted probe counts](sft-probe-counts.json); [full-investigation SFT + REINFORCE design](../expanded-studies/sft-design.md).

**Do not conflate actor and grader errors.** Nemotron's oversized evidence ranges and invented citation IDs, and Qwen's earlier quote-format errors, were judge contract failures. They motivated grader changes. The examples above were learner action failures and motivated interface/trajectory training.

## Reproduction

`build_evidence.py` extracts the claim example, recounts 54 errors from the 80 original training trajectories, verifies contributing-trajectory totals against the comparison summary, recounts all 256 SFT probe samples, and renders SVG charts using ReportLab. PNGs are rendered from the SVGs with Sharp. No new model calls, training, or grading were performed.
