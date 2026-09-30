# action-trace — local repository research

Design: [product plan](../requirements/PRODUCT_PLAN.md), with its
[editable Excalidraw reference](../requirements/diagrams/product-flow.excalidraw).

## Run

From the repository root:

```sh
python3 -m venv .venv-product
.venv-product/bin/python -m pip install -e '.[training,modal]' -r requirements-product.txt
cd product_web
npm ci
npm run build
cd ..
# Reuse an existing saved Tinker login, or authenticate once:
.venv-product/bin/tinker auth login
.venv-product/bin/python -m product_api
```

Open http://127.0.0.1:8000. Modal execution uses the existing Modal authentication
on the host. For frontend development run `npm run dev` in `product_web` while the
API runs on port 8000; Vite proxies `/api`.

The backend and run workers use the SDK's saved default credential from
`~/.tinker/credentials.json`. `TINKER_API_KEY` and `TINKER_CREDENTIAL_CMD` are also
supported through the SDK. No key is copied into product artifacts or sent to the
browser; never use a `VITE_` variable for credentials.

The app reads project checkpoint manifests from `artifacts/**/checkpoints/*.json`,
then reconciles the owning project runs with Tinker's paginated checkpoint list.
The picker shows only remotely verified, unexpired sampling checkpoints with a
supported renderer. Fake/local-only manifests, training-only checkpoints, and
unverified checkpoints are hidden. Missing/expired sampler weights never fall
back to base weights. Checkpoint discovery requires local manifests;
remote-only project runs must have their manifests downloaded first.

The repository selector lists all pinned environments from the existing dataset release.
Malformed model actions receive specific format feedback and up to five consecutive correction
attempts within the original run budgets. Rejected output is retained in the local
trajectory for diagnosis; the timeline shows correction notices. Provider failures
and exhausted budgets still stop the run. The model receives the selected repository
name and pinned commit, with tools already bound to its source checkout.
Before sampling, the adapter measures the fully rendered prompt with the model's
tokenizer and reserves up to 8,192 output tokens. Older tool observations are
replaced with references to their saved artifacts until the prompt fits; the
question and latest tool observation are preserved. An irreducible context
overflow terminates as a budget limit with zero usage for that unsent request,
not as a Tinker transport failure.
Selecting one immediately opens its source tree and starts a fresh investigation
view; prior runs remain in Recent runs. Source browsing does not start inference.

The repositories come from the existing dataset release. Source locations
are resolved from the standard local repo cache, then the original dataset source
record. No repo is cloned or environment rebuilt implicitly. Source is checked
against the pinned manifest hashes, including explicitly materialized files.
The live answering adapter samples only and never creates a trainer. Known-answer comparisons run a separate sampling-only grader after both investigations finish.

## Preview and activity

“Explore a scripted preview” exercises actual pinned source tools and a deterministic
scripted policy. The execution result is explicitly synthetic and no code is run.
It does not require provider credentials and is never a model quality result.

Real runs execute Python and tests in fresh network-blocked Modal sandboxes using
the existing environment backend. Environment readiness and source hashes are
verified. Test dependencies must exist in the selected image; a missing test runner
is reported as a failed execution, not a successful verification.

The UI streams completed model turns and tool lifecycle events, not sampled tokens
or private reasoning. Clicking tools, files, or citations opens pinned evidence.
The user supplies an ordinary question, not a research procedure. The agent chooses
its own files, tools, and investigation strategy. A product-specific evidence check
rejects final answers without observed source citations and permits at most five
repair attempts within the existing token, turn, and time budgets. Repeated failure
ends the run without presenting an unsupported final answer. Other harness consumers
retain their existing behavior unless they opt into submission feedback.

Only observed source lines receive verified citation badges. “Verified” means source
location validity, not semantic correctness of the answer.

## Persistence and limits

Run data lives under `artifacts/product-runs` (override with `PRODUCT_RUNS_DIR`).
Keep one backend process for that directory. Up to eight investigation workers can
run concurrently across chats and browsers; each comparison uses two slots.
Grading continues independently. At capacity, new submissions receive HTTP 429
without launching either comparison side. New chat / New comparison leaves
existing work running and available in history.
Each run is bounded by 80 model turns, 64 tool calls, 64,000 generated tokens, and 900 seconds.
The adapter caps each generation at 8,192 tokens within the model context window.
The supervisor terminates workers that exceed the saved run deadline plus 60 seconds (960 seconds by default). Stop sends SIGTERM so
sandbox cleanup can run; a forced process kill may leave a remote operation alive
until its own bounded sandbox timeout. Already-submitted provider work may still
consume tokens. Browser disconnect alone does not stop a run.

Public events support SSE IDs and replay on reload. Unexpected worker exits and
backend restarts preserve completed evidence and mark the run failed/interrupted.
No automatic retry submits another paid run. Token cost is labeled as an estimate when calculated from published Tinker prices; unavailable pricing remains unknown.

The server binds to loopback and rejects non-local Host/Origin requests. It has no
multi-user authentication and must not be exposed publicly.

## Checks

```sh
.venv-product/bin/python -m unittest tests.test_product -v
.venv-product/bin/python -m unittest discover -s tests -p 'test_*harness*.py'
cd product_web && npm run build
```

## Research features

- Answers attach model-attributed citations to the paragraph containing their
  exact `claim` sentence. Clicking a citation opens and highlights its pinned
  source range. “Source read” verifies availability to the model, not factual
  support; older answers without claim mappings retain an additional-sources list.
- The composer continues the current chat by default, keeps the finished run's repository and checkpoint, and creates
  a new addressable run linked to its parent. It carries the prior question,
  bounded answer text, and up to three cited source excerpts (6,000 characters
  total). Source excerpts are loaded again from the pinned commit. Other source
  must be investigated again. “New chat” clears the visible conversation and starts without this context. Previous turns remain visible above the current investigation.
- The investigation map shows encountered files in discovery order and distinguishes
  search matches, successful reads, and read attempts. Each file opens the inspector.
- “Add GitHub repo” imports a public HTTPS GitHub URL and branch, tag, or commit
  into a persistent source-only catalog entry. Git objects are read without a
  checkout or executing repository code. Private-repo authentication and automatic
  execution environments are not supported. Files over 2 MB, symlinks, and
  submodules are omitted; imports are limited to 10,000 regular files, 100 MB of
  source, and two minutes. Reimporting the same name and commit reuses the entry.
  This control is available in both Single and Compare. The commit/ref is
  optional: leave it blank to resolve the default branch's HEAD. Compare selects
  the imported repository and switches to free-form questions automatically,
  since imported repositories have no saved benchmark rubric. Existing runs
  remain in history when importing a new repository.

## Model comparison and quick demo

On macOS, double-click `action-trace.command` in the repository folder. It opens
the comparison page, reuses an existing backend if one is already running, or
builds and starts the backend otherwise. Keep its Terminal window open while
using a newly started server. From Terminal, `./scripts/demo-product.sh --open`
does the same thing. Saved comparisons are available under Recent comparisons.

On this Mac, `com.action-trace.demo` is installed as a user LaunchAgent at
`~/Library/LaunchAgents/com.action-trace.demo.plist`. It starts at login and keeps
the backend running after browsers, Codex, or Terminal close. Open
<http://localhost:8000/?mode=compare> or <http://127.0.0.1:8000/?mode=compare>
in any browser on this Mac. The service uses this repository's `.venv-eval`,
saved provider login, and run directory. Logs are under `artifacts/product-server`.
It binds only to this machine; access from other devices is not enabled.

After backend edits, let active runs finish, then restart the service with
`launchctl kickstart -k gui/$(id -u)/com.action-trace.demo`.
For frontend edits, run `npm --prefix product_web run build` and reload the browser.
To disable the background service, run
`launchctl bootout gui/$(id -u) ~/Library/LaunchAgents/com.action-trace.demo.plist`
and remove that plist if it should not start at the next login.

Verified September 29, 2026: [base versus the three-update bash checkpoint](http://127.0.0.1:8000/?comparison=64377a2ffdc04bab9fbd7f8cbbd6c032).
Both answered the pinned Hello-World README question correctly with a source
citation: base took 12.5 seconds / 3 tool calls; checkpoint took 9.6 seconds /
1 tool call. This is an integration smoke test, not a quality benchmark.
The checkpoint is `bash-correctness-grpo-v1`, optimizer step 3, sampler
`ckpt-b59e635f596c4cde9f600f3c1c7fe6e1`.

The [harder Pydantic comparison](http://127.0.0.1:8000/?comparison=b9451c97baab47e9a9e205d838c4150a)
is also saved: base exhausted action-format repairs without an accepted answer;
the checkpoint completed but scored 25/100 against the draft assertion rubric.
These live runs use the product research tools, not the original bash-only
training/evaluation harness. For that experiment's matched-case results, see
[the pilot report](../reports/bash-correctness-grpo-v1/README.md) and
[saved trajectories](../reports/bash-correctness-grpo-v1/trajectories.html).

For future inference, retain the checkpoint in Tinker: this pilot was saved with
a 48-hour TTL. Saved comparison answers remain viewable locally after expiry,
but new inference needs available sampling weights. Tinker supports changing or
removing checkpoint expiry; see its [checkpoint CLI documentation](https://tinker-docs.thinkingmachines.ai/tinker/cli/checkpoint/).
For a shared deployment, add authentication and persistent storage for runs,
source snapshots, and project checkpoint manifests. Keep Tinker credentials on
the backend. Sync approved checkpoint manifests, then verify remote availability;
do not silently switch a saved demo to an arbitrary newest checkpoint.

Run `./scripts/demo-product.sh` and open <http://127.0.0.1:8000/?mode=compare>.
The script builds the frontend and starts the existing local backend, using
`.venv-product` or the existing `.venv-eval` environment. It does not install
packages, open public ports, or replace saved investigations.

Choose Model A and Model B, then submit one shared question. Hosted Qwen models,
including `Qwen/Qwen3.5-397B-A17B`, are discovered through Tinker's server
capabilities. Compatible instruction models use their own tokenizers in
non-thinking mode; unsupported renderers and raw base/pretraining variants are
disabled. Project sampling checkpoints remain separately grouped and verified.
The server caches discovery for 60 seconds; Refresh models forces rediscovery.

Each side has its own worker, tool timeline, answer, citations, stop control,
context, artifacts, and limits. Follow-ups preserve only that side's prior context.
A failed or stopped side does not interrupt the other. New comparison unlocks
model/repository choices. Recent comparisons and `?comparison=<id>` restore both
sides, including after a browser reload. Backend restart marks unfinished runs
interrupted and never automatically resubmits paid work.

“Try a scripted comparison” provides a quick, clearly labeled interface demo
without inference credentials, paid model calls, or actual execution. The two
scripted policies intentionally use different tool counts. This is not a model
quality benchmark.

Live metrics show model turns, tool calls, input/output tokens, elapsed time,
model wait (including first-call adapter initialization), and tool time. Token
cost estimates use standard uncached input and output rates from Tinker's public
`models.json`, captured with the model configuration when the run starts. They
exclude sandbox costs and are not invoices. Interrupted or otherwise incomplete
usage is labeled partial. Missing rates show “Cost unavailable.”

Optional `PRODUCT_MODEL_PRICING_FILE` overrides are a JSON object keyed by picker
model ID or exact Tinker model name, for example:

```json
{
  "Qwen/Qwen3.5-397B-A17B": {
    "input_per_million": 3.0,
    "output_per_million": 7.5,
    "source": "manually verified rate",
    "checked_at": "2026-09-29"
  }
}
```

Overrides require finite, nonnegative numbers. Historical runs keep their saved
rates. Each run has a 900-second total limit; individual provider requests
have up to 180 seconds within that budget. Both models run concurrently, but
independent sequential tool investigations can still take tens of seconds.

Comparison API: `POST /api/comparisons` accepts `repo_id`, `left_model_id`,
`right_model_id`, `question`, optional `parent_comparison_id`, and optional
`preview`. `GET /api/comparisons` lists comparisons; `GET /api/comparisons/{id}`
returns both runs; `POST /api/comparisons/{id}/cancel` stops both. Existing per-run
endpoints continue to work. `preview=true` always uses synthetic policies,
regardless of the requested model IDs, and is labeled in all persisted results.

Additional checks: `.venv-eval/bin/python -m unittest tests.test_product_comparison.ComparisonTests -v`.

Demo budgets are centralized in `product_api/limits.py` and snapshotted into each new run. Hosted models use up to 32,768 context tokens, capped by provider capabilities. Checkpoints retain their trained context limit; generation reserves at most one quarter of that window, up to 8,192 tokens. Both action-format and citation-repair allowances are five attempts within the same overall budgets. Previously finished runs retain their original results.


## Known-answer comparison scores

New comparisons default to a repository-specific known-answer benchmark. Select
an exact question with a saved reference answer and weighted assertion rubric.
The default Pydantic question compares copying, frozen models, strict validation,
and instance revalidation across six assertions. Free-form questions and scripted
previews remain explicitly unscored; historical free-form answers do not acquire
an invented rubric retroactively.

Before either run starts, the API freezes the rubric, verifies evidence file
hashes against the pinned checkout, and saves the source excerpts and rubric hash.
The answering workers receive only the question and permitted tools, never the
reference answer or rubric. Both answers are graded by the same fixed
`Qwen/Qwen3.5-397B-A17B` judge at temperature zero, without candidate model names.
The grader must return a verdict for every assertion, using valid evidence IDs
and actual answer passages. The application computes the weighted score:
`100 * sum(weight * credit) / sum(weight)`, where supported earns 1, partial earns
0.5, and missing or contradicted earns 0. Unresolved assertions, invalid grading,
and absent answers do not receive an invented numeric score.

The comparison shows each total out of 100 and an assertion table with weights,
verdicts, reasons, quoted answer passages, and clickable pinned source evidence.
This measures reference-assertion coverage, not every additional factual claim.
Reference review status is visible: current draft references are not human-reviewed.
Expand reference provenance to inspect the saved answer and rubric hash.

Grading starts automatically after both runs finish, has a separate 600-second
supervisor limit, and persists incremental results in `comparisons/<id>.grade.json`.
Grader token cost is shown separately from answering costs. Stop both also stops
grading; restart preserves results and never automatically resubmits paid work.
Scored questions do not inherit conversational context: choose New comparison for
another scored question. Free-form comparisons retain independent follow-ups.

`GET /api/benchmarks?repo_id=<id>` lists available known-answer questions.
`POST /api/comparisons` accepts `benchmark_id` with its exact question and repository;
it rejects altered questions, preview mode, and parent context for scored runs.
Comparison responses include the frozen benchmark and grading state.

Checks: `.venv-eval/bin/python -m unittest tests.test_product_grading tests.test_claim_grading -v`.
