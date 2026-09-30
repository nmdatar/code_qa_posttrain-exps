# Repository Q&A task authoring protocol

Inputs: preassigned repository-family split, exact commit, ready environment recipe/image, existing task inventory, and collection coverage priorities. Never change family splits to increase training counts.

1. Inspect source, tests and available real question provenance at the pinned revision. Select a mechanism absent from the existing inventory. Record whether the prompt is adapted from a user question or newly synthesized from source/tests; do not manufacture provenance.
2. Draft a bounded prompt with explicit setup, operations and requested observations. Avoid naming the expected behavior or supplying the answer in the premise. Include enough information for a second investigator to reconstruct the experiment. Prefer interactions and discriminating interventions over isolated API trivia.
3. Construct a private reference: atomic required claims, exact source ranges, critical errors, and executable assertions. A claim must answer the prompt; implementation details are not automatically required simply because they are citable.
4. Make the fixture discriminate between plausible explanations. Include counterexamples, boundary cases, alternate flags/orderings or control interventions. Use explicit events/clocks rather than fragile sleeps, and keep all state for a scenario inside one sandbox operation.
5. Execute only in the isolated task image. Capture stdout, stderr, exit status and sandbox identity. When a result differs, re-read source and fix the prompt, fixture or reference with an explanation. Never automatically replace expected output with observed output.
6. Give a separate reviewer the prompt, claims, source and execution evidence. Require a reason for every supported/unsupported claim; return underspecified or incorrect tasks for revision. Reviewers do not silently edit authors' expectations.
7. Run a fresh-context question-only control on exact public prompts when measuring dependence on repository access. Grade it separately against the same claims. One control is advisory, not a difficulty benchmark. Citations alone do not establish difficulty.
8. Export only after schema, source-binding, split and assertion checks pass. Keep private references outside solver inputs. Executable tasks, independently reviewed drafts, human-approved records and completed solver trajectories are distinct states.

Parallelize by repository for authoring and by different repository for review. Reuse exact immutable environments; rerun all new assertions. Retain intermediate failures and review corrections. Report task counts by quality state, elapsed time, image builds avoided, coverage, and unknown costs honestly. Do not produce many paraphrases or arbitrarily changed constants to inflate size.
