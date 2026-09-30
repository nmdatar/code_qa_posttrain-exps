# Background research: repository Q&A and post-trained coding agents

Research date: **September 28, 2026**. This report expands the README's **Goal** and **Product** headings and provides context for **Dataset** and **Model Training**. The [original project prompt](/Users/ndatar/Documents/ChatGPT/action-interview/prompts/initial-project-prompt.md) specifies RL-training a small LLM for maximally efficient code Q&A and serving deep-research-style answers in a full-stack product. The README is intentionally a scaffold. Recommendations below operationalize that goal; they are not claims about an already built system.

Companion reports: [Dataset research](/Users/ndatar/Documents/ChatGPT/action-interview/DATASET_RESEARCH.md) and [Training paradigms and algorithms](/Users/ndatar/Documents/ChatGPT/action-interview/TRAINING_RESEARCH.md).

## 1. Main finding and project framing

A useful project would answer developer questions about an unfamiliar repository, collect the evidence needed to support each answer, and show precise source citations. The most defensible research question is:

> Can post-training improve repository exploration and evidence-grounded answers over a strong retrieval-augmented agent using the same base model, tools, and inference budget?

**DeepWiki is a strong product analogue, but not a publicly documented recipe for Q&A-specific post-training.** Cognition documents repository indexing, generated wikis, source links, diagrams, and question answering. Those facts establish product behavior; they do not establish the particular dataset, loss function, or reinforcement-learning algorithm behind it. The closest direct training precedents found are **SWE-QA-Pro** and **RepoSearch-R1 / RepoQA-Agent**. [DeepWiki documentation](https://docs.devin.ai/work-with-devin/deepwiki), [SWE-QA-Pro](https://arxiv.org/abs/2603.16124), [RepoSearch-R1](https://arxiv.org/abs/2510.26287).

Three distinctions should govern the project:

- **Repository knowledge versus model behavior.** An index or generated wiki stores changing repository information; post-training teaches an agent how to search, inspect, synthesize, and abstain. Updating one does not automatically update the other.
- **Retrieval versus reasoning.** Finding the right file is a necessary subproblem, but explaining a cross-file execution path requires connecting evidence and handling conditions and exceptions.
- **Code repair versus Q&A.** A patch can be checked with tests. A natural-language answer needs claim-level evidence and coverage checks; successful repair training does not establish Q&A accuracy.

These are analytical distinctions used in this report, not proprietary architecture claims.

## 2. Product and implementation landscape

| System | Documented contribution | Public evidence of relevant training | Best use in this project |
|---|---|---|---|
| Cognition DeepWiki / Ask Devin | Generated repository documentation and source-grounded Q&A | No specific DeepWiki Q&A training recipe found in reviewed official materials | Product reference and external qualitative comparison |
| AsyncFuncAI DeepWiki-Open | Independent, open wiki-generation implementation | Reviewed README describes application features, not a validated agent-training recipe | Inspectable product prototype reference |
| Sourcegraph Cody | Codebase context retrieval for chat | Context documentation describes search and prompt augmentation | Retrieval baseline and context UX reference |
| Sourcegraph Deep Search | Iterative code search/navigation agent with source trace | Documentation mentions frontier and specialized models but does not disclose a reproducible QA training recipe | Agentic retrieval and enterprise product reference |
| Cursor semantic search | Custom embedding model trained using agent-session signals | Explicit first-party retrieval-model training disclosure | Evidence that learning the retriever can matter separately from training the answerer |
| Aider repository map | Compact structural repository context | Structural context mechanism, not a QA post-training result | Cheap repository-orientation baseline |
| SWE-QA-Pro | Repository-QA benchmark and SFT/RLAIF training workflow | Directly relevant published training recipe | Primary research starting point |
| RepoSearch-R1 | MCTS-driven RL for a repository-QA agent | Directly relevant published training recipe | Advanced exploration-training comparison |

Sources and evidence limits for each row are detailed below. This is a comparison of mechanisms, not a performance leaderboard: these systems have not been evaluated here under a common protocol.

### 2.1 DeepWiki and Ask Devin

Cognition's May 2025 launch describes DeepWiki as the public version of Devin Wiki and Devin Search, with over 50,000 public repositories indexed at launch. That is a historical launch figure, not a current coverage estimate. [Cognition launch announcement](https://cognition.com/blog/deepwiki).

The current documentation describes automatically generated architecture diagrams, documentation, and links to source code. Ask Devin uses wiki information to locate relevant context. Public DeepWiki offers basic documentation and Q&A, while the Devin application adds a broader search/planning/session workflow. A repository can steer wiki generation through `.devin/wiki.json`; specifying pages replaces default cluster-based planning with the requested page set. These are useful concrete signs that coverage and document planning are product concerns, not just model-quality concerns. [DeepWiki documentation](https://docs.devin.ai/work-with-devin/deepwiki).

The official MCP announcement exposes three tools: `ask_question`, `read_wiki_contents`, and `read_wiki_structure`. This establishes two consumption paths: a person browsing a wiki and another agent consuming its context. [DeepWiki MCP announcement](https://cognition.com/blog/deepwiki-mcp-server).

**Project implications:** Build a minimal question-and-evidence interface before attempting a complete wiki generator. Keep optional architecture summaries separate from authoritative source excerpts. An answer should identify the repository revision it inspected; otherwise a polished summary can silently describe an older implementation. Consider an MCP interface only after the evidence contract is stable.

**Evidence gap:** The reviewed materials do not identify a DeepWiki-specific QA dataset, teacher model, SFT objective, RL reward, or reproducible benchmark protocol. It would be inaccurate to label a proposed SFT-plus-RL implementation a reproduction of DeepWiki's training.

### 2.2 DeepWiki-Open

AsyncFuncAI's repository explicitly describes itself as an independent implementation attempt. Its current README describes code-structure analysis, documentation, diagrams, navigable wiki pages, and code-guided tours; it identifies the current iteration as DeepWiki-Open / Grok-Wiki and lists an MIT license. It should not be represented as Cognition's released source code. [DeepWiki-Open repository](https://github.com/AsyncFuncAI/deepwiki-open).

**Project implications:** This is useful for studying the surrounding application: repository ingestion, documentation generation, navigation, and presentation. Treat an application scaffold and a trained policy as different assets. Before adapting code, pin a commit, inspect the actual implementation and dependencies, and verify the selected version's behavior. This research reviewed the public repository description; it did not run or audit the application.

### 2.3 Sourcegraph Cody and Deep Search

Cody's documentation explains a retrieval-augmented chat workflow: search for relevant code, include it in the model's context, and identify files read. It supports questions that require multiple files, such as tracing data through components. [Cody chat documentation](https://sourcegraph.com/docs/cody/capabilities/chat).

Deep Search goes further: a model repeatedly invokes Sourcegraph search and navigation tools, refining its understanding before answering. The response includes searches and files contributing to the answer. Current documentation also describes scoped searches and structured analysis outputs, and states that the service selects frontier and specialized models. These disclosures do not establish which models were trained for which QA objectives. [Deep Search documentation](https://sourcegraph.com/docs/deep-search).

**Project implications:** A strong comparison must include iterative tool use, not only a single vector-search request. The user interface should expose the evidence collected rather than just a confidence-sounding answer. Multi-repository questions are a valuable later extension, but introduce identity, version, and dependency-alignment problems that complicate initial evaluation.

### 2.4 Cursor: train the retrieval component

Cursor reports training a custom embedding model for semantic code search using signals from agent sessions, where the agent searches and reads files while solving tasks. Its discussion places semantic retrieval alongside regex-based search, rather than treating the two as mutually exclusive. This is explicit evidence of training a retrieval component, not proof of a particular Q&A-answer policy or a universally superior search strategy. [Cursor semantic-search research](https://prod.cursor.com/blog/semsearch).

**Project implications:** If failures mainly come from missing relevant files, improve retrieval first. Candidate interventions include stronger embeddings, query rewriting, reranking, and symbol-aware chunking. Answer-policy RL is a less direct fix when the evidence never reaches the model. Conversely, high retrieval recall with poor answers points toward synthesis, verification, or training-data problems.

### 2.5 Aider: a low-cost structural baseline

Aider provides a compact repository map containing important definitions and signatures, selecting useful portions within a token budget. This offers orientation without placing the entire repository into context. [Aider repository-map documentation](https://aider.chat/docs/repomap.html).

**Project implications:** Include a repository-map-plus-search baseline. A result that beats plain prompting but fails to beat this inexpensive structural context is a weak justification for complex training. Maps can guide where to look, but the agent should still inspect the actual implementation before making behavior claims.

## 3. Research most directly connected to Q&A post-training

### 3.1 SWE-QA-Pro: strongest starting point

SWE-QA-Pro, submitted in March 2026, combines an evaluation benchmark with synthetic data generation and a two-stage SFT followed by reinforcement learning from AI feedback recipe. Its benchmark emphasizes long-tail repositories, topical coverage, executable environments, and filtering questions answerable without repository exploration. The authors report a trained Qwen3-8B exceeding GPT-4o by 2.3 points on their benchmark. That is a result under the paper's evaluation configuration, not evidence of general superiority over all proprietary coding agents. [SWE-QA-Pro paper](https://arxiv.org/abs/2603.16124).

**Why it matters:** This directly connects the task, dataset design, and training intervention the proposed project needs. It is more closely matched than using a code-completion corpus or treating a repair benchmark as an answer-quality benchmark.

**Research caution:** Filtering out direct-answer successes deliberately emphasizes questions requiring exploration. Report results on that challenge distribution and on a naturally sampled question set. Otherwise the experiment may overstate how frequently agentic search is needed in routine usage.

### 3.2 RepoSearch-R1 / RepoQA-Agent: training exploration

The October 2025 RepoSearch-R1 paper describes a repository-QA agent trained with MCTS-guided reinforcement learning. Its method includes KL-free group-relative policy optimization, intermediate process rewards, judge-based outcome rewards, and an SFT-free cold start. The paper discusses generalization limitations and reward hacking. [RepoSearch-R1 full text](https://arxiv.org/html/2510.26287v1).

**Why it matters:** It targets the decision process of choosing tools and following evidence, making it a direct precedent for training repository navigation. It also motivates comparing simpler rollout generation against tree search.

**Research caution:** The abstract's compliance language should not be adopted as a general guarantee. Avoiding teacher-trajectory distillation does not resolve repository licensing, benchmark reuse, judge-provider restrictions, or permission to train on private code. Separately, MCTS increases exploration compute; compare total training and serving costs, not only the number of optimizer steps.

### 3.3 DeepRepoQA: useful inference-time comparison

DeepRepoQA, submitted in August 2026, describes MCTS-guided repository exploration for multi-hop QA and evaluates on SWE-QA. Its abstract establishes a tree-search agentic framework; it should not be cited as evidence that the proposed project must post-train model weights. [DeepRepoQA](https://arxiv.org/abs/2608.24221).

**Why it matters:** A research result should separate gains from better inference-time search from gains due to learning. Run the same untrained base model with a stronger search procedure before attributing improvement to post-training. Tree search is most worth testing on questions where early file-selection errors send a linear agent down the wrong path.

### 3.4 FastContext: historical idea with a material status caveat

FastContext proposed separating exploration into a specialized subagent that returns file paths and line ranges, using reference-model trajectories and task-grounded rewards. **The paper was withdrawn on June 30, 2026; its current arXiv record states product-IP issues and a need for reapproval.** Treat it as a historical architectural idea, not an available, verified reproduction target or a current performance recommendation. [FastContext current arXiv record](https://arxiv.org/abs/2606.14066).

**Project implication:** An explorer/answerer split is a testable design choice. Preserve actual evidence spans across the boundary so that the answerer can verify the explorer's summary. Do not assume delegation improves quality simply because it reduces the main model's context length.

### 3.5 Evidence against assuming more agents are better

An August 2026 empirical study compares semantic search with delegated deep agentic search on SWE-QA. It reports 65.2% versus 46.2% correctly answered questions in its setup, with semantic search producing correct answers at less than half the cost. It attributes a substantial share of delegated-search failures to planner/subagent handoffs. The paper is listed as under journal review. [Deep Agentic Search empirical study](https://arxiv.org/abs/2608.01507).

**Interpretation:** This is evidence for including a strong semantic-search baseline and explicitly measuring handoff failures. It is not evidence that all subagents or all learned exploration policies are worse. Model selection, tools, budgets, indexing, task distribution, and scoring can change the outcome.

## 4. Adjacent work: useful but not task-equivalent

**CoReQA** grounds repository questions in GitHub issues and comments across 176 repositories and four programming languages, with multidimensional model-judged evaluation. It helps establish that repository QA is broader than code-comment paraphrasing. [CoReQA](https://arxiv.org/abs/2501.03447).

**SWE-QA** reports 576 curated QA pairs and an agentic baseline in its original paper, covering repository-level reasoning beyond isolated snippets. It is useful for continuity with several exploration papers, but a narrow set of popular repositories should not be the only held-out evaluation. [SWE-QA](https://arxiv.org/abs/2509.14635).

**SWE-agent** studies the agent-computer interface: tools and interaction formats affect repository navigation and editing performance. This supports controlling the tool interface when comparing trained and untrained agents; changing tools and weights simultaneously obscures the source of gains. [SWE-agent](https://arxiv.org/abs/2405.15793).

**SWE-Gym** supplies real repair tasks, executable environments, and test verification, and demonstrates agent training from interaction trajectories. It is useful methodology for collecting and replaying tool-use experience, but its repair rewards cannot simply score explanatory answers. [SWE-Gym repository and paper links](https://github.com/SWE-Gym/SWE-Gym).

**SERA** provides a public data-generation/training implementation for soft-verified repository agents. It is relevant to generating affordable agent trajectories and adapting to repositories, with public code and datasets; its coding-task results do not substitute for grounded Q&A evaluation. [SERA repository](https://github.com/allenai/SERA), [SERA paper](https://arxiv.org/abs/2601.20789).

**Repo4QA** retrieves a GitHub repository as the answer to a programming question. Despite the similar name, this is different from answering a question about a specified repository. Treat it as repository discovery work, not a direct benchmark of cross-file explanatory reasoning. [Repo4QA paper](https://aclanthology.org/2022.coling-1.136.pdf).

## 5. Recommended project definition and experiment

The following is a proposed design derived from the comparison, rather than a claim from any single source.

### Product scope

Start with a read-only assistant over one pinned repository revision. Accept a question and return a concise explanation, file/line evidence, relevant caveats, and an explicit indication when the repository cannot establish the requested fact. Prioritize:

1. Finding an implementation or configuration entry point.
2. Explaining an execution or data-flow path across files.
3. Explaining conditions, defaults, errors, and exceptions.
4. Comparing two implementations or configuration paths.
5. Identifying what cannot be inferred from code alone, such as undocumented author intent or deployment-specific behavior.

Do not conflate code's observable behavior with historical rationale. If the question asks “why,” code may show the mechanism while a design document or issue supplies the rationale. Store those evidence types separately.

### Baseline ladder

| Baseline | Purpose |
|---|---|
| Direct answer, no repository tools | Detect prior knowledge and potentially answerable-without-context questions |
| Full repository context when it fits | Check whether exploration is needed at this scale |
| Lexical/symbol retrieval plus generation | Establish a cheap, inspectable baseline |
| Hybrid lexical and semantic retrieval plus reranking | Establish a competitive retrieval baseline |
| Iterative search/read agent with unchanged model | Measure the value of tool interaction |
| Same agent with SFT | Isolate the effect of learned tool use and answer format |
| Same SFT agent with RL | Measure additional benefit from optimizing outcomes |
| Optional explorer subagent or tree search | Measure architecture/inference effects separately |

Use the same repository snapshots, question set, answer budget, and evidence format wherever possible. For external products where budgets and internals cannot be controlled, report a separate product comparison with that limitation.

### What would make the result convincing

Measure answer correctness, important-point coverage, citation support, missed evidence, unsupported assertions, and appropriate abstention. Pair those with tool calls, tokens, latency, and cost per accepted answer. A long answer with many correct statements can still be worse if it invents the critical condition the developer asked about.

Split repositories before producing training questions or trajectories. A random split of questions from the same codebase can reward memorized repository structure. Hold out repository families and include changed revisions, renamed symbols, and stale-document cases. Use a small expert-reviewed evaluation subset to check whether automated judges reward verbosity or familiar phrasing.

The central experiment should change one thing at a time. If SFT improves formatting but not evidence discovery, report that. If RL improves a judge score but harms human-reviewed factuality, treat it as a failure. If hybrid retrieval matches the trained agent at lower cost, that is a useful research conclusion rather than a reason to weaken the baseline.

### Recommended positioning

Position the work as **a reproducible, evidence-grounded repository-Q&A agent with a measured post-training contribution**. The novelty must come from demonstrated improvements in evidence acquisition, citation accuracy, generalization, or efficiency. A wiki-shaped interface alone is already represented by existing products; an unqualified claim to reproduce DeepWiki's undisclosed training would not be supportable.

## 6. Reading order and remaining evidence gaps

Read SWE-QA-Pro first for the closest task/training connection; RepoSearch-R1 next for exploration-oriented RL; SWE-QA and CoReQA for task/evaluation framing; and the semantic-versus-delegated-search study to design adversarially strong baselines. Use DeepWiki and Sourcegraph documentation for product requirements, and Aider/Cursor for structural and learned retrieval alternatives.

Before implementation, inspect the exact released code, model and dataset licenses, repository snapshots, train/test partitions, and judging prompts of any selected research artifact. Public availability is not proof that an artifact reproduces every result in a paper. This report performed source research, not benchmark execution, model training, product testing, or a comprehensive code audit. All numerical results above are attributed author reports, and no cross-paper score should be interpreted as a controlled ranking.
