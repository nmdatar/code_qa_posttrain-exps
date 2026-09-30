# Training and evaluation acceptance scenarios

These are the original acceptance specifications. Current offline and live machinery validation is recorded in [the training smoke report](../reports/training-pipeline-smoke.md). Repository data/calibration acceptance remains separate.

## Deterministic contract and integration checks

Use a fake backend with inspectable optimizer state plus a tiny deterministic environment. Test externally observable behavior rather than duplicating implementation details.

| ID | Scenario | Required evidence |
|---|---|---|
| A01 | Supported base model and compatible checkpoint resolution | Correct purpose-specific handles and tokenizer/renderer identity; unsupported model/capability fails before sampling or update |
| A02 | Answer-only and multi-turn SFT examples | Intended assistant targets receive loss; user/tool tokens are masked; malformed alignment and empty targets fail with reasons |
| A03 | Optional/repeated stages | Base → GRPO and SFT → checkpoint → GRPO work; repeated stages retain lineage and separate stage counters |
| A04 | Generic environment boundary | Repository Q&A and deterministic tasks use the same runner without repository fields in the generic task contract |
| A05 | Episode isolation | Two same-task episodes have independent mutable tool state; private reference content never reaches policy observations |
| A06 | Multi-turn log-probability attribution | Every action token uses its actual sampling context and pinned policy; earlier turns reused as prompt are not trained again |
| A07 | Group arithmetic | Rewards [0, 1] give advantages [-1, 1] with population standard deviation; equal rewards give zeros; all-zero contribution causes no optimizer step |
| A08 | Invalid group | A missing reward, nonfinite log probability, mixed policy, mixed task, or version mismatch prevents the group entering an update |
| A09 | Retry and tool failures | One full-group retry for unresolved infrastructure failures, then quarantine; valid wrong answers remain scored; invalid tool arguments are policy outcomes; cleanup runs |
| A10 | Sampling refresh | Next batch uses the new snapshot only after a successful update; no mixed old/new sampling policy within a batch |
| A11 | Evaluate checkpoint | Sampling artifact loads without optimizer allocation and all results carry immutable checkpoint identity |
| A12 | Fork checkpoint | New run loads weights with fresh optimizer, counters and lineage; sampler-only references do not falsely qualify for training |
| A13 | Resume checkpoint | Stored optimizer, strategy state, data order/cursor and client RNG restore; configuration mismatches fail; discarded in-flight work is not claimed as completed |
| A14 | Partial save and ambiguous update | Partial artifacts never publish a usable manifest or replace the prior checkpoint; uncertain remote updates are not blindly retried |
| A15 | Periodic evaluation | Training pauses at the configured committed step; development evaluation uses the saved snapshot; transient failures persist incomplete coverage before training continues |
| A16 | Split protection | Training rejects held-out tasks and conflicting family/lineage splits; final test is excluded from periodic evaluation and selection; pilot-test provenance remains explicit |
| A17 | Reporting semantics | Strict verifier and reference-rubric reports remain separately labeled; unresolved cases retain failure counts and coverage; synthetic results are labeled synthetic |
| A18 | Tracking failure | Local events and complete traces survive W&B failure; reconnection does not repeat optimizer updates; event identity prevents duplicate logical records |
| A19 | Extension boundary | Register a fixture strategy with a distinct input/auxiliary capability requirement without modifying the core orchestrator; unsupported capability fails clearly |
| A20 | Existing evaluator compatibility | Existing dataset, grading, report and CLI tests continue to pass; generated reports preserve prior score meanings |

## Live end-to-end smoke test after design review

Choose an available Tinker model and a reviewed, split-isolated data release. Configure bounded steps, generation/tool budgets, learning rates and spend before execution. This smoke test checks connectivity and behavior; it does not require a reward increase or claim improved model quality.

1. Run a base-model development evaluation and record the resolved identity/configuration.
2. SFT on approved examples, commit a checkpoint, then initialize GRPO from those weights with a fresh optimizer.
3. Collect real multi-turn repository rollouts, verify them, perform a bounded update, and inspect masked-token/log-probability alignment.
4. Observe live W&B scalars, selected complete rollout views, verifier explanations, measured usage, and links to full durable traces.
5. Commit and periodically evaluate a GRPO snapshot; demonstrate a separate standalone evaluation against the same snapshot and settings.
6. Demonstrate a fresh fork and interruption/resume with distinct optimizer semantics and preserved lineage. Do not expect remote sampled text to match bit-for-bit.

## Design-package checks performed now

- Three editable scenes exist and screenshots have been inspected for labels, arrows and layout.
- Local `.excalidraw` exports contain editable shapes/text/arrows and valid binding references.
- Index links connect the scenes, exports, architecture, contracts, decisions and scenarios.
- User-confirmed scope is separated from proposed algorithm defaults and unresolved experiment inputs.
- Existing dataset roadmap and runtime source remain unchanged by this deliverable.

Review completion means the user has worked through the overview, GRPO loop and checkpoint lifecycle and accepted or revised the technical proposals. Artifact creation alone does not satisfy that review gate.
