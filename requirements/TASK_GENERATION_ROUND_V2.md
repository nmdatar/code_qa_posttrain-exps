# Task generation: round v2

Nine newly synthesized, source-backed tasks across the existing three pinned repositories. Together with the first three tasks, there are **12 runnable development tasks**. These are independently reviewed reference drafts, not human-approved gold, accepted training data, or nine newly collected solver trajectories.

The final nine tasks have **49 required claims** supported by independent reference review. All nine executable matrices passed twice in fresh Modal sandboxes, with matching outputs, exit status and timeout/truncation status. Environment images and prior image-bound isolation/source-attestation evidence were reused; these environment checks are not represented as newly executed in this round.

## Task catalog

| Task ID | Investigation | Required claims |
|---|---|---:|
| `pydantic-typed-extras-runtime-policy` | Typed extras, per-call policy overrides, storage and serialization | 5 |
| `pydantic-factory-data-signature-order` | Factory signatures, field order, validated input and default validation | 6 |
| `pydantic-alias-choice-unused-inputs` | Alias precedence, unused inputs, name fallback and output aliases | 6 |
| `tanstack-signal-consumption-unsubscribe` | Reading an abort signal changes last-observer cancellation behavior | 5 |
| `tanstack-manual-cache-edit-cancel-revert` | Public cache edits versus direct state changes during cancellation | 5 |
| `tanstack-inflight-options-retry-capture` | In-flight deduplication and changing query functions versus retry policy | 5 |
| `sqlalchemy-savepoint-preflush-state` | Savepoint preflush, selective rollback and failed preflush recovery | 5 |
| `sqlalchemy-json-joinedload-uniquing` | Legacy versus modern result uniqueness, JSON values and lossy uniqueness keys | 6 |
| `sqlalchemy-none-default-json-matrix` | Omission, Python None, SQL NULL, JSON null and server defaults | 6 |

Read the exact solver-visible prompts, system prompts, repository commits and budgets in [the public task records](../data/releases/repo-qa-generation-v2/evaluation/development/tasks.jsonl). Read the [environment references](../data/releases/repo-qa-generation-v2/evaluation/development/environments.jsonl) and [hashed release manifest](../data/releases/repo-qa-generation-v2/manifest.json).

The question scenarios are synthesized from pinned code/tests and recorded source-question motivations where applicable; they are not presented as nine verbatim user reports. New task mechanisms were compared with the first batch and with one another. This local diversity screen is not a full semantic benchmark overlap audit.

## Review decisions

- **Query:** remove leading sentences that hint at cancellation and retry conclusions. Behavioral assertions were already passing, illustrating why passing assertions do not establish prompt quality.
- **SQLAlchemy:** fix a quoted default that had different semantics in prompt and fixture; recreate the specified fresh database in a failure control. Retain first drafts and initial review findings, then verify corrected versions.
- **Pydantic:** independent review supported all 17 claims, with no material correction required.

Detailed reviews: [Pydantic](../reports/task-generation/pydantic-independent-review.json), [Query](../reports/task-generation/query-independent-review.json), [SQLAlchemy](../reports/task-generation/sqlalchemy-independent-review.json). SQLAlchemy preserves its initial `needs_revision` status in the historical section; `current_revision_status` and the final revision follow-up record the resolution. Review hashes are checked against the final source specifications before packaging.

The exact-prompt, fresh-context question-only control recovered all **49 substantive claims across all nine tasks**. Retain these as development/regression tasks; no hard-reasoning label is justified by this round. See the [claim-level assessment](../reports/task-generation/question-only-assessment.json). Missing citations and lack of execution were not graded as substantive errors.

An exact-prompt, fresh-context question-only control is recorded separately from reference verification. Its result is advisory; this batch is not a measured hard benchmark. [Control inputs](../reports/task-generation/question-only-inputs.json) contain only public prompts. No citations or executed evidence are invented for the control.

## Reproduce and extend

Worktree: `dataset-generation`. Final specs and bundle paths are in [generation-batch-v2.json](../examples/dataset_sources/generation-batch-v2.json).

```sh
# Structural checks and unchanged prepared-bundle reuse.
.venv-dataset/bin/python -m dataset_builder.batch \
  --config examples/dataset_sources/generation-batch-v2.json --phase prepare

# Run a task family's private behavior assertions again in clean Modal sandboxes.
.venv-dataset/bin/python -m dataset_builder.build verify \
  --bundle data/generated/tanstack-query-generation-v2-reviewed
```

To author another version on the same repository/recipe, prepare a new empty bundle and run `python -m dataset_builder.reuse --source EXISTING_BUNDLE --target NEW_BUNDLE`; then execute its new assertions. Reuse requires identical environment records and valid image-bound evidence. It never copies an old task-verification pass. Keep private source specs, references and assertions away from solver inputs.

For actual solver investigations, use `dataset_builder.investigate start/tool/finish/replay` with the bundle and task ID; see [the runtime instructions](DATASET_EXECUTION_STATUS.md#repeat-the-workflow). This generation round did not collect nine new solver trajectories.

See [optimization log](TASK_GENERATION_OPTIMIZATIONS.md), [reusable authoring protocol](../prompts/dataset_task_author.md), and [machine-readable round results](../reports/task-generation/round-v2.json).

## Admission limits

All existing repository families stay in development. Training, preference and final-test exports are empty. Human review and a broader benchmark/family/semantic overlap audit remain outstanding. A future training collection needs additional preassigned repository families; development tasks must not be relabeled as training. Infrastructure provider and collection model choices remain configurable at the architecture level.
