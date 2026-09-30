# Training paradigms and primary algorithms for repository Q&A agents

Research date: **28 September 2026**. This report develops the README's **Model Training** heading, with implications for **Goal**, **Dataset**, and **Product**. The [initial project prompt](/Users/ndatar/Documents/ChatGPT/action-interview/prompts/initial-project-prompt.md) explicitly calls for RL-training a small LLM as a maximally efficient code Q&A agent and serving deep-research-style answers in a full-stack product. The project does not yet specify a base model, hardware, budget, or target languages. Implementation choices and budgets below are proposals, not completed experiments or published results. Companion reports cover [background and similar projects](/Users/ndatar/Documents/ChatGPT/action-interview/BACKGROUND_RESEARCH.md) and [dataset design](/Users/ndatar/Documents/ChatGPT/action-interview/DATASET_RESEARCH.md).

## Recommendation

Build a strong retrieval-and-tools baseline, then train an open coding model on verified tool trajectories. Compare **SFT alone**, **SFT plus DPO**, and **SFT plus GRPO** under the same repository split, tools, and inference budget. Start with SFT; promote RL only when it improves independently checked correctness and citation support, rather than the training judge's score alone.

The most directly relevant reproducible starting point is **SWE-QA-Pro**, which explicitly trains repository question-answering behavior. **RepoSearch-R1** supplies a complementary cold-start, tree-search-assisted RL approach. **Search-R1** informs the mechanics of learning to search, while **SWE-Gym** and **SWE-RL** provide adjacent software-engineering evidence. Their issue-resolution results do not establish Q&A performance.

The primary hypothesis should be: *Posttraining improves evidence acquisition and grounded explanation on unseen repositories at a fixed inference cost.* Avoid claiming that a DeepWiki-like interface implies a disclosed DeepWiki training recipe; product behavior and documented weight updates are different evidence.

## 1. What is actually being trained?

Separate four interventions so the experiment can identify the source of gains:

| Intervention | Updated component | Intended improvement | Main limitation |
|---|---|---|---|
| Frozen-model RAG | Index, chunking, ranking, prompts | Put relevant code in context | Retrieval can miss multi-file relationships |
| Inference-time agent | Tool interface, execution loop, search budget | Follow references, inspect runtime behavior | More calls can increase cost without useful evidence |
| Retriever/reranker training | Retrieval encoder or ranking model | Improve evidence recall/precision | Cannot by itself teach explanation or stopping |
| Generator/agent posttraining | Model weights or adapters | Learn tool selection, query refinement, synthesis, abstention | Can overfit repositories, teachers, or judges |

These can be combined. RAG does not inherently mean a frozen model: the experimental baseline here deliberately freezes generator weights. Likewise, a tool-using scaffold is not itself proof of posttraining. SWE-agent demonstrates that interface design can materially affect software-agent behavior; hold the interface constant when attributing a gain to learning. [SWE-agent paper](https://arxiv.org/abs/2405.15793)

For this project, model the input as a question plus repository identity and commit. An episode is a sequence of assistant actions, tool observations, and a final answer. Train the policy to decide where to look, what to inspect next, when evidence is sufficient, and which claims the evidence supports. Keep repository facts in the versioned environment whenever possible; using fine-tuning as a constantly refreshed repository memory makes updates and provenance harder to manage.

## 2. Closest training precedents and what transfers

### SWE-QA-Pro: direct task match

The paper reports a two-stage Qwen3-8B recipe: 1,000 SFT trajectories followed by RL on 464 questions. In its Table 2, agent-mode scores move from **30.03** to **34.34** after SFT and **35.39** after RL; GPT-4o scores **33.08**. These are summed scores from **five judge-rated dimensions (maximum 50)**, not accuracy percentages. The benchmark has 260 questions across 26 Python repositories, and the authors acknowledge overlap between training-reward and evaluation-judge distributions as a limitation. [Paper, sections 3–4 and limitations](https://arxiv.org/html/2603.16124v1)

The released training instructions identify Claude Sonnet 4.5 as the trajectory teacher, ms-swift for SFT, and GRPO through verl-tool against a repository tool server with an LLM-judged rubric reward. The repository announces training-code and dataset release on June 24, 2026. This makes it a concrete reproduction target, while the published score comparison remains specific to its harness and judges. [Training README](https://github.com/TIGER-AI-Lab/SWE-QA-Pro/blob/main/train/README.md), [official repository](https://github.com/TIGER-AI-Lab/SWE-QA-Pro)

**Project implication:** reproduce the small-model recipe before introducing a larger custom training stack. Add independent evidence checks and an external test set before treating its measured improvement as product validation.

### RepoSearch-R1: search-assisted exploration during training

RepoSearch-R1 trains a repository QA agent with MCTS-guided rollouts and a KL-free GRPO variant. Tree selection uses an exploration-decay UCT mechanism; self-critique diversifies child actions. Rewards combine an LLM outcome judge and intermediate process signals. It masks tool observations from the optimization loss and studies a cold start without SFT distillation. “Without external supervision” should not be interpreted as reward-free learning: the method still evaluates answers against ground truth using a judge. [RepoSearch-R1 paper, sections 3–4](https://arxiv.org/html/2510.26287v1)

**Project implication:** test this after a simpler rollout implementation works. Tree search can concentrate samples on promising exploration paths, but brings branching, prefix reuse, and sampling-distribution complexity. Compare total generated tokens, environment work, and judge calls—not just optimizer steps. Keep ordinary GRPO with a reference penalty as the initial controlled baseline; removing that penalty is an ablation, not a universal improvement.

### Search-R1: the search-learning machinery

Search-R1 interleaves generated actions with actual search results, applies PPO or GRPO, and excludes retrieved tokens from policy loss. Its outcome reward uses exact matching on short-answer factual QA. This supports learning a search policy with real observations, but its natural-language QA setup does not validate long-form repository explanations or source citations. [Search-R1, sections 3–4](https://arxiv.org/html/2503.09516v4)

**Project implication:** reuse the interaction and masking principles, not its short-answer reward unchanged. Repository QA often has several equivalent descriptions and multiple necessary supporting locations.

### Adjacent coding research

SWE-Gym supplies executable software-engineering environments and studies training agents and trajectory verifiers. It is useful for learning navigation and for designing replayable training infrastructure. A patch passing tests is a different output contract from a complete, well-supported answer. [SWE-Gym](https://arxiv.org/abs/2412.21139)

SWE-RL learns from software evolution with a lightweight similarity-based reward. It illustrates scalable posttraining from developer artifacts; similarity to one reference solution can still penalize valid alternatives. That problem becomes sharper for free-form explanations. [SWE-RL](https://arxiv.org/abs/2502.18449)

FastContext proposes separating an exploration model from a downstream solver. However, its current arXiv record is **withdrawn**, with a June 30, 2026 note citing product IP issues. Treat it as a historical design lead, not a current reproducible empirical foundation. [FastContext record](https://arxiv.org/abs/2606.14066)

## 3. SFT and distillation: the first training stage

Supervised fine-tuning teaches a model to imitate accepted assistant actions and final answers. For trajectory tokens \(z_t\), history \(h_t\), and assistant-token mask \(m_t\), use:

\[
\mathcal L_{\mathrm{SFT}}=-\frac{1}{\sum_t m_t}\sum_t m_t\log\pi_\theta(z_t\mid h_t).
\]

Here \(m_t=0\) for user messages and tool observations, and 1 for supervised assistant outputs. Observations remain visible as context. Do not teach the agent to fabricate a tool response by training it as assistant-generated text. Explicitly test the tokenizer's chat template and mask boundaries before a long run.

**Proposed data construction:** have a stronger teacher answer each training question against the exact checkout using the production tool schema. Replay calls, check cited paths and lines, and review substantive claims. Retain compact successful trajectories as well as examples of recovering from failed searches. Include questions whose correct answer is limited by missing configuration, unsupported premises, or absent historical context. Distillation is the data-generation strategy; SFT is the optimization procedure.

Train observable behavior: search arguments, selected file ranges, verification actions, concise evidence summaries, and answers. Long generated reasoning narratives are not a substitute for valid evidence. A teacher's fluent answer should not pass filtering solely because another instance of the same model approves it.

**Proposed curriculum:** begin with locating definitions and explaining bounded behavior; add cross-file call flows, configuration-dependent behavior, tests contradicting documentation, and questions requiring qualified answers. Mix difficulty levels instead of finishing with only the hardest examples. Sample by repository and question family so one large codebase cannot dominate the gradient.

The main SFT risk is imitation of inefficient teacher behavior. Repeated searches, giant directory dumps, and unsupported certainty can be learned as readily as useful navigation. Track these behaviors separately from training loss. If answer-only SFT helps as much as trajectory SFT, investigate whether the gain is mostly answer formatting rather than exploration.

## 4. Preference learning: DPO

Direct Preference Optimization trains on a preferred response \(y^+\) and rejected response \(y^-\) for the same input \(x\), relative to a fixed reference policy:

\[
\mathcal L_{\mathrm{DPO}}=-\mathbb E\log\sigma\left(\beta\left[
\log\frac{\pi_\theta(y^+\mid x)}{\pi_{\mathrm{ref}}(y^+\mid x)}-
\log\frac{\pi_\theta(y^-\mid x)}{\pi_{\mathrm{ref}}(y^-\mid x)}
\right]\right).
\]

DPO avoids fitting a separate explicit reward model and running online policy rollouts during the optimization step. It still needs preference data, whose generation and verification have costs. [DPO paper](https://arxiv.org/abs/2305.18290)

**Proposed use:** create near-miss pairs: correct answer versus plausible wrong configuration assumption; complete explanation versus missing an exception path; valid citations versus irrelevant citations; calibrated uncertainty versus an invented answer. Match verbosity where possible so preference labels do not simply reward length.

For **answer-only DPO**, give both answers the identical fixed evidence context. This isolates answer quality. For **trajectory preference training**, define the probability over model-generated actions conditional on their respective tool histories; exclude environmental observations. A naive chat-concatenation implementation can optimize the likelihood of copied code instead of the choices the agent controlled. Test the implementation against a small hand-inspected batch.

DPO is attractive when reviewable preference pairs are easier to obtain than trustworthy scalar rewards. Its key limitation here is coverage: offline pairs cannot directly teach recovery from unseen states unless those states appear in the data. Refresh candidate generation from the current policy periodically and evaluate whether that helps enough to justify the added complexity.

## 5. Online reinforcement learning: PPO and GRPO

For a repository episode \(\tau\), a useful conceptual objective is expected task reward with a penalty for drifting from a reference policy. Rewards may be deterministic, human-provided, AI-judged, or mixed. **RLVR** describes verifiable reward sources; **RLAIF** describes AI feedback. Neither is a separate optimizer.

### PPO

The clipped policy surrogate uses the current-to-old action-probability ratio \(\rho_t\) and estimated advantage \(\hat A_t\):

\[
J_{\mathrm{clip}}=\mathbb E_t\min\{\rho_t\hat A_t,
\operatorname{clip}(\rho_t,1-\epsilon,1+\epsilon)\hat A_t\}.
\]

The usual actor-critic implementation learns a value function; clipping limits the incentive for overly large updates on the sampled batch. A reference-policy KL penalty is an additional design choice, not the same thing as clipping against the rollout policy. [PPO](https://arxiv.org/abs/1707.06347)

**Project assessment:** PPO is worth considering when intermediate feedback and long-horizon credit assignment justify a critic. It adds value-model memory, training, and failure modes. Do not choose it solely because it is a familiar RLHF label.

### GRPO

GRPO replaces the learned critic with relative rewards from multiple completions of the same prompt. A common outcome-based advantage is

\[
\hat A_i=\frac{R_i-\operatorname{mean}_j R_j}{\operatorname{std}_j R_j+\varepsilon}.
\]

This advantage feeds a clipped policy objective, typically with reference regularization. It saves critic-model requirements but still generates a group of rollouts and evaluates them. [DeepSeekMath, GRPO](https://arxiv.org/html/2402.03300v3)

**Proposed implementation:** sample 4–8 bounded episodes per question; run actual tools against an immutable snapshot; compute separate correctness and evidence scores; optimize assistant tokens only. This group size is a pilot choice, not an established optimum. Log group reward variance, invalid calls, truncation, KL, token counts, and repository-specific performance.

If all rollouts receive the same reward, the normalized task advantage provides no differentiating signal. Persistent all-zero groups can indicate overly difficult questions, broken tools, or an insensitive verifier. All-success groups can indicate insufficient challenge or reward saturation. Inspect examples before increasing training compute. Group normalization also changes relative weighting across questions; it does not magically make different reward rubrics comparable.

Use rollout-policy log probabilities, not reference-policy probabilities, in importance ratios. Keep the tokenizer, tool protocol, sampling policy, and observation truncation rules versioned. Changing these midway can turn an apparent learning effect into an environment change.

## 6. Reward design and credit assignment

**Proposed reward contract:** optimize grounded correctness under a cost constraint. Keep the following components separately logged even if a weighted scalar is needed for training:

| Component | Possible check | What the check cannot establish |
|---|---|---|
| Citation validity | Commit, path, and line range exist | The cited text supports the claim |
| Claim support | Compare each claim with its cited code and dependencies | Global completeness or all runtime cases |
| Answer correctness | Expert rubric, execution for specific behavioral claims | Correctness of unrelated prose |
| Coverage | Required subquestions and exception paths addressed | Every valid alternative explanation |
| Calibration | Appropriate uncertainty on genuinely unanswerable cases | That broad refusal is useful |
| Efficiency | Tool calls, generated tokens, latency | That fewer calls mean better reasoning |

A candidate scalar is \(R=g(w_cC+w_sS+w_vV+w_aA)-\lambda K\), where \(g\) gates serious unsupported answers, \(C,S,V,A\) represent correctness, support, coverage, and calibration, and \(K\) is bounded cost. This is a design proposal. Tune weights on development examples with expert review, and retain the raw components so tradeoffs remain visible.

Avoid rewarding citation count, passing unrelated tests, or explanation length. File existence is a cheap validation gate, not proof of semantic support. Runtime tests can establish a specific behavior for a tested input; they cannot verify an architectural explanation wholesale. Keep evaluators and answer keys inaccessible to the agent's repository tools.

Outcome rewards assign the same terminal signal to both useful and wasted actions. Process supervision can instead label whether an inspection resolves a relevant uncertainty, whether a citation supports a claim, and whether a conclusion is premature. Evidence from mathematical reasoning favors process supervision in that domain, but direct transfer to repository navigation requires testing. [Let's Verify Step by Step](https://arxiv.org/abs/2305.20050)

For a pilot, annotate observable action quality on a small subset. Do not pay a positive reward merely for searching a new file: the model can collect that reward while avoiding the answer. Prefer verified milestones and compare against outcome-only training. A learned verifier should be tested on fabricated citations, omitted branches, plausible nonsense, and verbose irrelevant answers before it guides policy updates.

## 7. Parameter efficiency and model choice

**Proposed starting model:** an open 7B–8B coding-capable instruction model with a compatible license, reliable tool serialization, and a context length adequate for the intended observations. Qwen3-8B is a reproduction anchor because the direct-task recipe uses it; this is not a claim that it is the best available model in September 2026. Select by a short baseline bakeoff rather than a generic coding leaderboard.

LoRA freezes base weights and trains low-rank updates \(\Delta W=BA\). QLoRA combines low-rank adapters with a frozen quantized base to reduce memory requirements. Neither removes activation memory, long-context attention costs, or rollout generation costs. [LoRA](https://arxiv.org/abs/2106.09685), [QLoRA](https://arxiv.org/abs/2305.14314)

Start with adapter SFT if hardware is limited. Record target modules, rank, precision, packing, and context length. Compare a small full-fine-tuning run only if adapter capacity appears to limit gains. Validate the deployed quantized configuration independently: training precision and serving precision are not necessarily equivalent. QLoRA support in an SFT library does not establish compatibility with the selected GRPO rollout engine, distributed trainer, or adapter synchronization path; verify those together in a short end-to-end run before budgeting RL memory savings.

A separate explorer is a reasonable optional architecture: it returns evidence locations to a larger answer model. It can isolate retrieval costs, but should be judged by final answer quality as well as retrieval recall. Treat this as a later ablation; beginning with two trainable agents makes diagnosis more difficult.

## 8. Feasible staged experiment and budget accounting

The following is an **illustrative pilot**, not a resource commitment or published recipe. Use the dataset report's repository-level split rules.

| Stage | Proposed scale | Required decision |
|---|---|---|
| Baselines | 200 development questions, repository-held-out | Identify retrieval versus synthesis failures |
| SFT pilot | 1,000 verified trajectories, then 5,000 if justified | Does tool validity and grounded accuracy improve? |
| Preference pilot | 1,000–3,000 reviewed pairs | Does DPO help at matched answer length? |
| RL pilot | 500–1,000 training questions, 4–8 rollouts | Does reward improvement survive independent evaluation? |
| Final comparison | Frozen independent repository test split | Which checkpoint lies on the quality/cost frontier? |

At an assumed **8,000 total context-and-target tokens** per SFT trajectory, 1,000 examples are **8 million tokens per epoch**; three epochs process **24 million**. The trainable-target count is smaller when observations are masked, although the model still processes those observations. This arithmetic does not predict wall time.

At an assumed **1,000 RL questions × 8 rollouts × 2,000 generated tokens**, one pass generates **16 million assistant tokens**, plus observation processing and repeated prefix prefills. Every additional pass repeats substantial serving work. Separately budget teacher generation, retries, sandbox initialization, evaluation, and judge calls.

Measure 50–100 representative episodes and a short training run before requesting larger compute. Estimate GPU time from measured effective throughput and include idle/environment overhead. Do not infer “fits on one GPU” from parameter count or cite old QLoRA demonstrations as a hardware guarantee for this agent workload.

## 9. Evaluation and promotion criteria

Use the same base checkpoint, tools, repository snapshots, context limits, and token/call budgets to compare:

1. Frozen model with no repository access, as a leakage and prior-knowledge diagnostic.
2. Frozen single-pass retrieval and frozen iterative tool agent.
3. Answer-only SFT versus full-trajectory SFT.
4. SFT plus DPO versus SFT plus GRPO.
5. Outcome-only reward versus outcome plus citation/process checks.

A no-repository result is not necessarily cheating: some questions legitimately concern public knowledge. Report those separately from tasks selected specifically to require code inspection. Hold out repositories, related forks, and near-duplicate question templates; use a temporal slice where feasible. Training on the test question's earlier commit can still leak the answer.

Report human-audited correctness, supported-claim rate, citation coverage, calibrated abstention, tool failures, and latency/cost distributions. Bootstrap uncertainty by repository, not only by question, because questions from the same codebase are correlated. Repeat stochastic runs for close comparisons.

Use a judge distinct from the training reward provider and retain a blinded human sample. Compare the trained model with best-of-N inference from the original checkpoint at matched cost: an apparent training gain may instead reflect additional sampling. Monitor an unrelated instruction-following/coding holdout for regressions.

**Promotion gate:** the trained checkpoint should improve grounded answer quality on unseen repositories without exceeding the agreed latency/cost envelope or increasing unsupported confidence. A higher rubric score alone is insufficient. If SFT already meets the gate and RL does not, ship the SFT agent and invest in data and evidence retrieval before expanding the optimizer stack.
