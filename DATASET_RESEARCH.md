# Dataset research: post-training a repository Q&A agent

Research date: September 28, 2026. This report develops the README's **Dataset** heading for the confirmed goal in the [initial project prompt](/Users/ndatar/Documents/ChatGPT/action-interview/prompts/initial-project-prompt.md): RL-train a small, efficient code Q&A model and serve deep-research-style answers in a full-stack product. The proposed dataset targets checked-out repositories and verifiable evidence. Published facts are linked to primary sources; proposed counts, mixtures, and acceptance thresholds are planning estimates, not published results.

Companion reports: [Background and similar projects](/Users/ndatar/Documents/ChatGPT/action-interview/BACKGROUND_RESEARCH.md) and [Training paradigms and algorithms](/Users/ndatar/Documents/ChatGPT/action-interview/TRAINING_RESEARCH.md).

## Recommendation

Build a versioned dataset of **questions, repository snapshots, evidence-backed answers, and reproducible tool trajectories**. Use public repository-Q&A benchmarks mainly for external evaluation, and construct training examples from a separate pool of repositories. Start with supervised trajectories and a small, carefully reviewed evaluation set before scaling synthetic generation or reinforcement learning.

The learning target is not simply “explain this function.” It is: identify what the user needs, inspect the correct repository version, follow dependencies, verify behavior when necessary, and produce an accurate answer whose substantive claims have support. Training only on code–comment pairs or issue-resolution patches leaves substantial gaps in this target.

## Existing datasets: what each actually measures

Counts refer to the cited release or paper, not a guarantee that every downloadable record is usable. Dataset names are unusually easy to confuse in this area.

| Dataset | Published scope and supervision | Appropriate role | Important limitation |
|---|---|---|---|
| **SWE-QA-Pro** (Cai et al., 2026) | 260 benchmark questions from 26 repositories; repository snapshots and natural-language answers. The paper describes issue-derived topic coverage, human review, and filtering questions answerable without repository access. | Strong external evaluation for agentic navigation and grounding; closest published dataset/training design to this project. | Small test set; difficulty filtering changes its distribution relative to ordinary user questions. [Paper](https://arxiv.org/html/2603.16124v1), [dataset](https://huggingface.co/datasets/TIGER-Lab/SWE-QA-Pro-Bench). |
| **SWE-QA** (Peng et al., 2025/2026) | Original paper reports 576 questions, including cross-file and multi-hop understanding. The maintained repository supplies commit-pinned QA and agent baselines. | Evaluate open-ended repository comprehension; borrow question categories. | Version-sensitive counts: paper describes issue collection from 11 repositories; maintained README describes v1 over 12 projects and v2 adding three. Record exact artifact revision. [Paper](https://arxiv.org/abs/2509.14635), [repository](https://github.com/peng-weihan/SWE-QA-Bench). |
| **CoReQA** (Chen et al., 2025) | Repository questions constructed from issues and comments across 176 repositories and four programming languages; five-aspect LLM judging. | Direct repository-Q&A resource and the task source used by RepoSearch-R1. | Audit task/repository overlap before reusing it for training and evaluation; issue-derived answers require snapshot validation. Distinct from CodeRepoQA and RepoQA. [CoReQA paper](https://arxiv.org/abs/2501.03447), [RepoSearch-R1 dataset description](https://arxiv.org/html/2510.26287v1). |
| **RepoProbe** (2026) | 500 discussion-derived questions across 50 repositories and 15 languages. Answers use weighted technical checklists covering architecture, business logic, and implementation. | Particularly useful for answer completeness and architecture-level evaluation. | Checklist scoring still uses an LLM judge; it is more inspectable than a single holistic score, not an infallible oracle. [Official repository](https://github.com/Tencent-Hunyuan/RepoProbe). |
| **SWE Atlas – Codebase QnA** (2026) | 124 tasks over 11 repositories in Go, Python, C, and TypeScript; expert reference answers and rubric-based assessment, with executable environments. | Evaluate runtime investigation, execution tracing, and demanding multi-file explanations. | Small, difficult suite; latency and environment failures must be separated from comprehension errors. [Official dataset](https://huggingface.co/datasets/ScaleAI/SWE-Atlas-QnA), [Scale overview](https://scale.com/blog/swe-atlas), [harness](https://github.com/scaleapi/SWE-Atlas). |
| **StackRepoQA** (2026) | 1,318 real developer questions and accepted answers mapped to 134 open-source Java projects. | Natural question styles and retrieval/memorization diagnostics. | Accepted public answers can be obsolete or memorized. Authors report evidence of verbatim answer reproduction; high answer similarity does not establish repository reasoning. [Paper](https://arxiv.org/abs/2603.26567), [full text](https://arxiv.org/html/2603.26567v1). |
| **CodeRepoQA** (Hu et al., 2024/2025) | 585,687 issue-derived multi-turn entries across 30 repositories and five languages; reported mean 6.62 dialogue turns. Collection occurred in August 2024. | Candidate source of realistic developer topics, follow-ups, and question seeds. | Issue conversations are not automatically verified QA, commit-aligned evidence, or successful agent trajectories. Large raw volume should not be mistaken for equivalent high-quality supervision. [Paper](https://arxiv.org/abs/2412.14764), [dataset repository](https://github.com/kinesiatricssxilm14/CodeRepoQA). |
| **RepoQA / Search Needle Function** (2024) | 500 tests: five languages × ten repositories × ten target functions. Given long code context and a description, retrieve the target function. | Isolate long-context localization and retrieval. | Despite the name, this task is not unrestricted explanatory Q&A; retrieving a function does not test a complete architectural answer. [Official implementation](https://github.com/evalplus/repoqa). |
| **CodeQA** (2021) | 119,778 Java and 70,085 Python question–answer pairs generated from comments using syntactic and semantic transformations. Input is a code snippet plus question. | Optional auxiliary supervision for local code comprehension and answer formatting. | Snippet-scale, comment-derived supervision lacks repository navigation and may teach comment paraphrasing. Original automatic metrics include BLEU, ROUGE-L, and METEOR. [Official repository](https://github.com/jadecxliu/codeqa), [paper](https://aclanthology.org/2021.findings-emnlp.223/). |
| **CodeSearchNet** (2019) | Approximately two million comment–code pairs in six languages, plus human relevance judgments for code search. | Retriever training or retrieval-only diagnostics. | Documentation proxies are not complete QA answers or tool-use supervision; code licenses are recorded separately from the project's MIT tooling license. [Official repository](https://github.com/github/CodeSearchNet). |
| **SWE-bench** (2023 onward) | Original set: 2,294 issue-resolution problems from 12 Python repositories; Verified: 500 human-filtered tasks. | Auxiliary environment engineering and code-investigation practice. | Primary target is a patch evaluated through tests. Patch success cannot substitute for explanation correctness or citation quality. [Paper](https://arxiv.org/abs/2310.06770), [official suite](https://www.swebench.com/). |

**Name collision:** Elkoussy and Perez's *SWE-QA: A Dataset and Benchmark for Complex Code Understanding* is a different 2026 work: 9,072 multiple-choice questions generated from 12 Python repositories. It can diagnose multi-hop understanding, but multiple-choice accuracy and open-ended, cited answers are different outputs. Do not merge its statistics with Peng et al.'s SWE-QA. [Primary paper](https://arxiv.org/abs/2604.24814).

**Training release and reproducibility:** SWE-QA-Pro reports 1,464 training questions, split into 1,000 SFT trajectories and 464 RL questions. The same paper reports training-set coverage of 1,484 repositories, which is difficult to reconcile with those question counts; verify manifests before reproducing that statistic. [Paper, §§2.2 and 3.2](https://arxiv.org/html/2603.16124v1). The project announced its training release on June 24, 2026. Its [SFT dataset](https://huggingface.co/datasets/TIGER-Lab/SWE-QA-Pro-SFT-Trajectories) exposes 1,000 trajectories in Hermes tool-calling format, with `tools` and `messages` fields and an MIT license tag. The [training README](https://github.com/TIGER-AI-Lab/SWE-QA-Pro/blob/main/train/README.md) documents SFT download instructions and RL parquet files in the repository. This is a concrete starting point, subject to tool-format conversion, provenance checks, and train/evaluation overlap auditing; it has not been downloaded or replayed in this research task.

## Separate three data products

1. **Repository corpus:** immutable source snapshots, documentation, test fixtures, dependency manifests, and optional build environments. This supplies the environment and retrievable evidence.
2. **Task corpus:** questions, conditions, accepted factual claims, evidence ranges, answerability labels, and evaluation rubrics. This defines what successful answering means.
3. **Trajectory corpus:** actual tool calls, their observations, final answers, verifier results, and cost measurements. This teaches the agent how to investigate.

A QA pair can support answer fine-tuning without teaching exploration. A trajectory can teach exploration while ending in an incorrect answer. Keep both quality dimensions explicit. Store repository snapshots once and reference them by immutable identifiers, instead of duplicating entire repositories into every example.

## Proposed construction pipeline

### 1. Define coverage before gathering volume

Begin with Python and TypeScript/JavaScript if no deployment language has been chosen; expand based on actual intended users. These are proposed priorities, not a claim of universal superiority. Sample libraries, web services, command-line tools, build systems, and data infrastructure, across repository sizes and documentation quality. Cap examples per repository so a prolific project cannot dominate.

Use a task taxonomy with localization, call/data flow, configuration precedence, API behavior, architecture, observed failure explanation, tests/invariants, and version changes. Include questions that require two or more files, but retain straightforward questions: production traffic will not consist exclusively of benchmark-hard puzzles.

Label **lookup** and **deep investigation** separately, and assign explicit tool, token, and time budgets. An efficient small model should stop after one decisive lookup when sufficient, while spending more on multi-file or runtime investigation when justified. Preserve successful alternatives at different budgets to train and evaluate this tradeoff without treating minimal tool use as success by itself.

Distinguish **implementation evidence** from **design intention**. Code can show that a cache expires after a condition, while “why the maintainers chose that condition” may require a design document or discussion. Annotate unsupported intention questions as uncertain rather than inventing motives.

### 2. Freeze environments and provenance

Record repository URL, full commit hash, submodule hashes, dependency lockfiles, build image digest where applicable, and file hashes. Snapshot documentation together with code. For runtime questions, preserve command, arguments, working directory, relevant environment settings, input fixture, exit status, and output artifact hashes.

Resolve historical issue questions against the version being discussed. A merged fix or later release may invalidate the original answer. If the relevant version cannot be established, rewrite the question to target the pinned snapshot and revalidate it, or exclude it.

### 3. Generate questions from complementary sources

| Source | Construction method | Quality control |
|---|---|---|
| Maintainer discussions and support issues | Extract concrete questions and follow-ups; use historical answers as leads. | Re-answer against the chosen commit; reject unsupported or version-ambiguous claims. |
| Code structure | Traverse declarations, callers, configuration readers, interfaces, and tests to propose questions. | Require a human-readable task; reject questions that expose the answer through a target symbol's exact name unnecessarily. |
| Executable behavior | Create deterministic inputs that reveal ordering, fallback, serialization, error handling, or boundary conditions. | Reproduce the result; preserve the environment and ensure the answer explains the mechanism. |
| Documentation and examples | Ask whether documented behavior matches implementation and identify relevant code. | Verify against source; include carefully labeled stale-document cases. |
| Authorized user questions | Collect representative usage and clarification patterns. | Consent/provenance tracking, secret removal, and independent answer validation. |

Generate a question and evidence plan first, then let a solver investigate without being handed the gold evidence. This prevents trajectories from merely walking through answer locations leaked by the generator. Record generator and solver identities, prompts, versions, and access to reference information.

### 4. Verify answers at the claim level

Break the answer into atomic claims and assign each claim one or more supporting code ranges or runtime observations. Validators should check that paths exist at the pinned commit, ranges contain the intended symbol, quotations match, and the cited material supports the particular claim. A valid link alone is insufficient.

For runtime claims, prefer executable checks when feasible. For architecture claims, independently trace relevant connections. Static call graphs are helpful candidates, but reflection, dynamic dispatch, dependency injection, and generated code can invalidate a simplistic graph.

Use a second verifier with access to the code and question. Send disagreements and consequential claims to a human reviewer. Freeze the rubric before scoring candidate models. Maintain `accepted`, `needs_review`, and `rejected` states with reasons; do not silently keep fluent answers when evidence is missing.

### 5. Collect replayable trajectories and useful failures

Record visible actions and observations: search queries, directory listings, file reads, symbol lookups, bounded execution, and the final answer. Do not require private model reasoning traces. Include short decision summaries only where useful for training and permitted by the producing system.

For supervised learning, prefer successful, reasonably efficient traces that match the deployed tool interface. For preferences or RL analysis, retain alternative answers and trajectories labeled for factual errors, omitted conditions, fabricated citations, excessive cost, or premature stopping. Verify that preference pairs concern the same task and snapshot; do not reward a longer answer merely because it sounds more thorough.

Include recoveries from empty search results, truncated output, unavailable dependencies, and ambiguous names. Add examples where the correct response requests clarification, states insufficient evidence, or says that a claimed symbol is absent from the inspected snapshot. These teach calibrated behavior instead of mandatory answer generation.

## Suggested example schema

This abbreviated JSON is illustrative; its repository, commit, paths, and answer are placeholders. Production records should use complete hashes and separate large observation artifacts.

```json
{
  "id": "qa-000123",
  "schema_version": "1.0",
  "repository": {
    "url": "https://example.org/owner/project",
    "commit": "FULL_COMMIT_HASH",
    "family_id": "project-family-17",
    "license_record_id": "license-17"
  },
  "question": "When both config and environment specify a timeout, which wins?",
  "category": "configuration_precedence",
  "answerability": "answerable",
  "reference_answer": "The environment value takes precedence in this snapshot.",
  "claims": [{
    "id": "c1",
    "text": "Environment overrides the file value.",
    "evidence": [{
      "path": "src/config.py",
      "start_line": 81,
      "end_line": 94,
      "file_sha256": "FULL_FILE_HASH"
    }],
    "verification": "code_review_and_runtime_check"
  }],
  "trajectory": [{
    "tool": "search",
    "arguments": {"query": "TIMEOUT"},
    "observation_ref": "artifacts/qa-000123/search-01.json"
  }],
  "rubric": [{"claim_id": "c1", "required": true, "weight": 1}],
  "provenance": {
    "source_kind": "synthetic_from_code",
    "generator_version": "RECORDED_VERSION",
    "human_reviewed": true
  },
  "split": "train",
  "quality_status": "accepted"
}
```

Add environment digest, command evidence, source thread identifiers, attribution, permissions, annotation history, token counts, tool budget, and retrieval-index version in the full schema. Keep gold answers and hidden rubrics outside the agent's filesystem and retrieval index.

## Splits, contamination, and leakage

Split by **repository family before generation**, grouping forks, mirrors, renamed repositories, and closely related packages. A random QA-row split permits nearly identical code and question patterns on both sides. Group near-duplicate code, copied documentation, and questions using both normalized hashes and semantic similarity, then manually inspect boundary cases.

Maintain three evaluation slices: unseen repository families; later snapshots of known repositories; and task categories or languages underrepresented in training. These measure different generalization questions and should not be combined into one opaque score. Deduplicate against every chosen public evaluation suite before admitting training sources, including derived trajectories and paraphrases.

Base-model pretraining contamination cannot generally be ruled out for public repositories. Compare closed-book, documentation-only, retrieval-assisted, and tool-agent baselines. Add newly authored questions on private or recently changed code where permission permits, plus controlled code modifications that alter the correct answer. These are diagnostics, not proof of zero contamination. Identifier renaming alone is especially weak when the algorithm and answer remain recognizable.

Do not provide source discussions containing accepted answers during a code-grounding evaluation unless that access is explicitly part of the task. For version-change questions, specify exactly which commits are visible. Enforce the chosen web-access policy consistently so one agent cannot look up the answer while another only reads code.

## Evaluation contract

Use a small set of interpretable measures rather than lexical overlap as the headline score:

| Measure | Proposed definition |
|---|---|
| Factual completeness | Weighted fraction of required rubric facts correctly stated. |
| Strict success | All essential facts satisfied and no critical contradiction. |
| Citation precision | Supported claim–citation links divided by all submitted links. |
| Citation coverage | Substantive verifiable claims with adequate support divided by substantive verifiable claims. |
| Runtime validity | Required experiments reproducibly support the answer. |
| Calibration | False-answer rate on unanswerable tasks, alongside correct abstention and clarification rates. |
| Efficiency | Input/output tokens, tool calls, wall time, and cost per successful answer. |

Report correctness and citation metrics separately: an answer can be correct but unsupported, or meticulously cited but wrong. Distinguish missing evidence from invalid paths. Evaluate within equal tool/context budgets and record harness versions; differences in tools and runtime availability can dominate model comparisons.

Calibrate automatic judges against double-reviewed human examples, using blind candidate identities and randomized answer order for pairwise judgments. Measure agreement and inspect disagreements, especially confident incorrect answers. Prefer item-level findings to a single rating of “quality.” Report uncertainty using repository-clustered resampling; many correlated questions from one repository do not supply independent evidence.

## Licensing, privacy, and release readiness

Treat permissions as data fields, not an assumption that public visibility permits every downstream use. Distinguish source-code licenses, discussion text, dataset annotations, and generated derivatives. Preserve attribution and original notices, and resolve unclear records before training or redistribution.

Concrete source checks illustrate the distinction. CodeSearchNet ships per-language license metadata for underlying source code. RepoProbe identifies Apache-2.0 for code and CC BY 4.0 for benchmark data, while its notice clarifies the provenance and retained ownership of discussion text. SWE-QA-Pro's benchmark card declares MIT. These declarations should be inspected alongside upstream artifacts; they do not establish blanket permission for every nested source. [CodeSearchNet metadata](https://github.com/github/CodeSearchNet/blob/master/resources/README.md), [RepoProbe](https://github.com/Tencent-Hunyuan/RepoProbe), [SWE-QA-Pro card](https://huggingface.co/datasets/TIGER-Lab/SWE-QA-Pro-Bench).

Scan for secrets, personal data, and accidental proprietary files. Maintain deletion and exclusion manifests and ensure those exclusions propagate to cached observations and generated derivatives. Execute collected projects only in isolated, resource-limited environments; repository contents and comments are untrusted inputs, not instructions to the collection agent. For public release, publish a dataset card explaining collection, transformations, licenses, exclusions, split design, annotation quality, and intended limitations.

## Practical first release and scaling gates

The following is a **proposed planning range**, not a validated minimum dataset size:

| Phase | Proposed size | Deliverable and decision gate |
|---|---|---|
| Pilot | 30–50 repository families; 300–500 reviewed QA items; 500–1,000 candidate trajectories | Reproducible snapshots, stable taxonomy, functioning verification, and a measured acceptance rate. |
| First training run | 100–200 training families; 3,000–10,000 accepted trajectories | Compare trajectory SFT with the same base model and tools; inspect error categories rather than just mean score. |
| Independent evaluation | 20–40 held-out families; 300–600 fully reviewed questions, plus external suites | Freeze before training selection; include answerability, citation, multi-file, and runtime slices. |
| Expansion | Scale only weak categories and underrepresented repositories | Demonstrate gains on untouched evaluation data at an acceptable cost per verified example. |

Do not count the pilot as an untouched final test after using it to refine prompts and filters. Human validation is likely to dominate the dependable data budget: for example, 500 evaluation items at an assumed 20–40 minutes each require about 167–333 reviewer-hours for one pass, before independent second review and adjudication. Measure actual review time in the pilot before committing to a larger collection.

The first release should include a manifest, immutable snapshot references, accepted tasks, replayable observations, claim-level rubrics, split assignments, quality reports, and a small hand-audited set of positive and negative examples. The key asset is an auditable connection between **question → investigation → evidence → answer**, not the largest possible number of scraped conversations.
