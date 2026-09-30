# Trace product implementation

Implemented the local repository Q&A application from the
[plan](../requirements/PRODUCT_PLAN.md) and its
[editable Excalidraw reference](../requirements/diagrams/product-flow.excalidraw).

## Delivered

- React/TypeScript three-panel workspace: repo tree, agent activity, code/output inspector.
- Pinned Pydantic and responses repositories from the existing dataset release.
- Base-model selection and local project checkpoint discovery with Tinker reconciliation.
- Inference-only JSON protocol adapter, capability-aware tool descriptions, and template identity checks.
- Bounded subprocess runs, isolated execution through the existing Modal backend,
  public SSE events, reconnect/replay, cancellation, crash recovery, and single-server ownership.
- Source-range citation validation; no correctness-grade claim or invented reasoning.
- Clearly labeled scripted preview with real pinned source reads and synthetic execution output.
- Local run history, responsive panels, keyboard-accessible controls, and setup documentation.

## Validation performed

- 45 product/harness/security/Tinker regression tests passed together.
- One additional inference-history/usage adapter test passed separately (46 targeted tests total).
- TypeScript checking and Vite production build passed.
- Pydantic environment manifest verified against all 978 source-file hashes.
- Browser preview completed four tool calls and a cited answer.
- Reload restored exactly four steps without duplication.
- Clicking the citation opened `pydantic/main.py`, highlighted lines 96–115,
  and displayed the correct pinned commit.
- Desktop and 390px-wide mobile inspector layouts reviewed.
- Browser error check was empty.
- Final desktop axe audit: 39 passes, zero violations; one incomplete category
  for contrast on offscreen/partially obscured tree nodes requires manual review.
- Excalidraw JSON checked: 27 unique elements with valid arrow bindings.

Visual verification files are under `artifacts/product-verification/` (ignored runtime artifacts).

## Live-service verification

Follow-up verification found the existing SDK login in `~/.tinker/credentials.json`.
The product readiness check now recognizes saved SDK credentials, environment API
keys, and configured dynamic credential commands. Credentials stay in the SDK;
no key was copied into product artifacts or browser state.

- Tinker catalog authentication succeeded and returned available project checkpoints.
- The first base-model run returned an unverified answer without tools. This remains
  recorded as evidence that an untrained model may skip research on an ordinary prompt.
- A second base-model run with explicit tool-use instructions completed in 26.3 seconds:
  pinned `read_file`, real Modal `python_probe`, then an answer citing observed lines.
- Probe exit code was 0, stderr empty, and the result confirmed `ValidationError`
  with `frozen_instance` when assigning a field on a frozen Pydantic model.
- Run ID: `828c9d866d614448bdb0549003bdfb89`; 3 model turns, 2 tool calls,
  5,256 input tokens and 331 output tokens. Cost remains unmeasured.
- Browser verification confirmed two tool cards, a completed answer, and an enabled
  verified-citation link. Screenshot: `artifacts/product-verification/live-base-demo.png`.
- All 11 product tests passed, including the new stored-credential regression test.

Live checkpoint sampling is still unverified; checkpoint catalog availability is
not evidence of checkpoint answer quality. Source citation validation establishes
observed location ranges, not semantic correctness of the model's explanation.

The existing Pydantic image supports Python probes; its installed dependencies determine
whether a selected pytest invocation can succeed. The product does not install extra
packages into an immutable environment automatically.

## Repository switching and autonomous-question follow-up

Fixed the initial two-repository catalog and stale previous-run explorer behavior.
All 44 pinned dataset environments now appear, distinguished by commit. Selecting
another repository clears the previous investigation and immediately loads its file
tree; source browsing works before starting inference. Saved runs remain available.
All 44 repository catalogs opened successfully against their pinned snapshots.
Browser verification switched from Pydantic to Requests: 140 files loaded, the prior
answer disappeared, and the question composer was cleared.

Removed the need for a user-authored tool procedure. Generic system instructions
expose tools and require source grounding; they supply no repo-specific paths,
line ranges, or investigation sequence. The product opts into bounded submission
feedback: unsupported citations return to the model for up to two repairs within
the existing run budget. Default behavior for other harness consumers is unchanged.

Live question: “How does Pydantic prevent assignment to a frozen model?”
Run `adeca86a1cc945089b704f6b75d39bbd` completed in 17.4 seconds using the base model.
It independently chose list_files, search_code, and two read_file calls, repaired
citations twice, and returned an answer with an observed source citation. No execution
was requested or needed by the chosen investigation. This demonstrates autonomous
tool choice, not independently graded answer correctness.

Validation: 47 targeted product/harness/tool/Tinker tests passed; TypeScript and
Vite production build passed. Added tests for full catalog discovery, pre-run source
browsing/path boundaries, feedback-based repair, token accounting, and repair limits.

## Executable plain-question test

Question: “How does Pydantic prevent assignment to a frozen model? Verify the behavior
with an executable example.” No source paths, line numbers, probe code, or ordered
procedure were supplied.

Both base-model attempts failed at the first action-format validation, before any
tool calls or sandbox execution:

- `16b3a68ea0374a0cb3a79101abb42018`: agent_error, 6.6 seconds, 96 output tokens.
- `f6df37cb055b43e49b050aaf260f7883`: agent_error, 6.0 seconds, 77 output tokens.

This test did not validate autonomous executable investigation. The earlier guided
execution demo succeeded, but that result must not be treated as evidence that the
base model reliably handles executable questions without procedural guidance.

### Runtime action recovery and checkpoint demo

Malformed product JSON actions now preserve the decoded response in the private
trajectory and return specific format feedback to the model, with at most two
repairs per run. Repairs consume the original token/step/time budgets; provider
failures are not retried. The UI displays correction notices. Shared harness
consumers retain zero action repairs unless explicitly enabled.

A live retry test (`e9fff3a2a83d485d9a0df639b3876118`) exercised all three
attempts and revealed repeated model errors requesting a repository URL. Product
prompts now include the selected repository name and commit and explain that the
tools already access its checkout (no investigation recipe added).

Checkpoint `ckpt-f1ee215de01d4d6682c40047ca9a5046` then completed the original
Requests header-merging question in run `f19f400f38d74fe28751b6fe000f8329`:
45.3 seconds, seven model turns, four source tools (list, search, two reads),
and verified citations to `src/requests/sessions.py` lines 59–88 and 490–493.
That successful run used two source-evidence corrections and no format repairs.

Validation: 48 product/harness tests passed, frontend build passed, browser
confirmed format-correction notices. Tests cover repair history, preserved raw
output, token accounting, retry exhaustion, and budget exhaustion.

### Token-aware demo context handling

Product inference now fits the fully rendered tokenizer prompt plus a reserved
output allowance into the checkpoint context window. It replaces older tool
observations with existing artifact references, retaining the original question
and latest observation. The full trajectory is unchanged. Irreducible overflow
is reported as budget exhaustion with zero usage for the unsent request, instead
of a misleading provider failure. Read-range feedback states the exact limit.
Action repair counts now reset after valid actions, bounding consecutive failures
while retaining the global episode budgets.

Offline replay of the failing `50b7402610c14e6d8ae2bb00be9a93aa` prompt shrank
8,521 tokens to 3,046 plus 1,536 reserved output tokens, preserving the latest
source read. A first live test (`f47bd5226848477893b74a4d580a9b74`) ran 14 turns
without context overflow but exhausted the old cumulative format-repair limit.
After resetting repair streaks on valid actions, live checkpoint run
`deda5deeada2470aa48afe8f448690e9` completed the unchanged Requests question in
88.3 seconds, 12 turns, eight source tool calls, and 630 generated tokens.
Citation ranges passed the observed-source check; this is not semantic grading
or evidence of a measured success rate. No model weights or question recipe changed.

Validation: 49 product/harness tests passed, including token compaction, latest
observation retention, irreducible overflow classification, usage accounting,
consecutive repair limits, and resetting repair streaks after a valid tool call.

### Evidence, follow-ups, investigation map, and public repository import

Added paragraph-level claim citation links when the model supplies an exact claim
sentence, with explicit separation between observed source and model-attributed
support. Unmapped/legacy citations remain labeled additional sources rather than
being assigned to claims heuristically. Exact mapping enforcement was tested but
removed after the checkpoint repeatedly paraphrased the claim: optional mappings
must not prevent otherwise valid answers.

Follow-ups keep the repo and checkpoint, link to their parent, and deliver bounded
prior answer/source excerpts from the pinned commit. Browser-created follow-up
`cedf230514de48d2b589785595c04432` exposed an oversized tool result; product tool
outputs now cap at 10 KB with larger outputs available through artifacts. Retry
`80356116089448c59f5e5ea63aec4ddc` completed in 43.6 seconds with four reads and
observed-source citations. A subsequent strict-claim trial failed; no reliability
claim is made for that model. The exact-range citation button opened its source
in the browser. Screenshot: `/tmp/product-followup-evidence.png`.

The investigation map groups encountered files in discovery order, marks search
matches/read attempts/reads, and opens source on selection. Public GitHub import
supports branch/tag/commit, persists across startup, pins a Git commit, and never
checks out or runs repo code. Successfully imported octocat/Hello-World at
7fd1a60b01f91b314f59955a4e4d4e80d8edf11d and verified its one-file tree in the UI.
Private repository authentication and imported execution environments remain
unsupported and are documented.

Validation: 52 product/harness tests passed; production frontend build passed;
browser checks covered parent navigation, map rendering, import catalog/source
browsing, citation navigation, and no horizontal overflow at desktop width.
