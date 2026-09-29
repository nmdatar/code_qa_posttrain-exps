# Modal execution environments

Implements the environment factory and disposable execution boundary in
[Dataset Generation — End-to-End Roadmap](https://app.excalidraw.com/s/8Ufs2ZMhWhu/AJrov8yM1oi):
build → setup/readiness checks → environment manifest → clean instance per attempt.
The model/controller and private verifier remain outside the sandbox.

Validation on 2026-09-28: all 115 repository tests passed with Modal 1.5.5,
including a test using the actual SDK process/stream wrappers and a fake router.
End-to-end remote smoke verification passed: image preparation, setup/readiness,
two fresh rollout sandboxes, filesystem isolation, output collection, and cleanup.
The two tiny-fixture attempts reached readiness in 1.67 and 1.39 seconds and closed
in 2.23 and 1.79 seconds total. This is a two-sample smoke test, not a throughput
benchmark. The [saved report](../artifacts/modal-smoke-remote/e3ce3c08b64143b084d063f789d31b72/report.json)
has sibling manifests and journals; [Modal run](https://modal.com/apps/nmdatar/main/ap-HAJ8KlP9lQKTdbm0E7kAb7).
Local readiness previously timed out before `Sandbox.exec` returned a process;
those sandboxes were also terminated. Run the controller remotely for this host's
network environment.

## Install and smoke test

Python 3.11+ and Git are required. Install the optional SDK and authenticate the
trusted controller using Modal's normal credentials configuration:

```bash
python3 -m pip install -e '.[modal]'
modal token new
python3 scripts/smoke_modal_sandbox.py
```

The live smoke test uses paid CPU resources: it prepares a Modal Python base
and pins its image ID, builds a tiny public fixture, checks readiness, and creates
two sequential isolated attempts. Each attempt verifies a previous attempt's
temporary file is absent. All sandboxes are terminated on normal/error exits;
finite server-side lifetimes cover controller crashes. Results, the pinned recipe,
manifest, and host-side journals live in `artifacts/modal-smoke/<run-id>/`.
No training, inference, grading calls, or GPUs are allocated.

An optional remote-controller variant is available:

```bash
python3 scripts/smoke_modal_remote.py
```

This uploads the `agent_harness` package and synthetic smoke script to a temporary
Modal Function and runs the same test there. Use it when that source upload is
approved, for example to investigate local access to regional command routers.
It returns artifacts to `artifacts/modal-smoke-remote/`. The function has a
10-minute lifetime; its sandboxes retain their own bounded lifetimes/cleanup.

## Sandbox cost estimates

At the published 2026-09-28 base rates, sandbox CPU costs $0.00003942 per physical
core-second and memory $0.00000667 per GiB-second. Modal bills the higher of
resource request and actual usage. The defaults request 0.5 core/1 GiB and cap
usage at 2 cores/2 GiB. Estimated sandbox compute for five minutes alive:

| Rollouts | At requested resources | At hard limits throughout |
|---|---:|---:|
| 1 | $0.0079 | $0.0277 |
| 1,000 | $7.91 | $27.65 |
| 10,000 | $79.14 | $276.54 |
| 100,000 | $791.40 | $2,765.40 |

These are estimates, not observed billing. Duration includes time waiting for
model responses while a sandbox remains alive. Multiply tasks by attempts per
task and repeated training passes to count rollouts; include retries. For example,
1,000 tasks × 8 attempts × 5 minutes is about $63–$221 in sandbox compute per pass.
Ten-minute rollouts double these figures. Eight continuously occupied slots cost
about $0.76–$2.65/hour at these resource bounds. Concurrency accelerates spending
per wall-clock hour; it does not multiply the cost of a fixed number of equal-length
rollouts.

Excluded: model sampling, private grading, Tinker training, one-time image builds,
controller Functions, storage, egress, plan fees and regional surcharges; account
credits/discounts are not applied. Prepared image reuse amortizes dependency
installation/builds across questions and attempts. There is no billable warm pool
kept running by this implementation.

Sources: [Modal pricing](https://modal.com/pricing),
[sandbox billing rules](https://modal.com/docs/guide/sandbox-resources).

Offline contract tests allocate no remote resources:

```bash
python3 -m unittest tests.test_environment_images tests.test_modal_backend tests.test_sandbox_cli -v
```

## Prepare a repository once

Select a repository containing solver-visible source only. Preparation uses a
full pinned Git commit, never local modifications or untracked files. `.git` is
not uploaded. Submodules and escaping/cyclic symlinks are rejected. Private gold,
credentials, and grading fixtures must be kept outside the selected repository;
committed files are not automatically classified as secret or public.

Write a recipe such as the following (replace the base digest with a real digest,
and adapt filenames to the repository). The smoke test saves a working example.

```json
{
  "mode": "executable",
  "base_image": "python:3.11-slim-bookworm@sha256:<64-character-registry-digest>",
  "apt_packages": ["git", "ripgrep"],
  "dependency_files": ["requirements.lock"],
  "dependency_commands": ["python3 -m pip install --require-hashes -r requirements.lock"],
  "build_commands": ["python3 -m compileall -q /repo"],
  "setup_commands": [],
  "readiness_commands": [["python3", "-c", "import your_package"]]
}
```

For `source_reading`, omit unnecessary dependencies/builds and check that source
and intended search tools are present. The mode records the intended capability;
it does not itself provide a read-only filesystem or enforce a tool allowlist.
`executable` tasks whose readiness fails must be quarantined, never silently
downgraded to source-reading tasks.

Alternatively, use `"base_image_id": "im-..."` with a real, already-built Modal
image ID instead of `base_image`. Exactly one base field is required. To prepare
a native base once on the trusted controller:

```python
import modal
app = modal.App.lookup("action-interview-environments", create_if_missing=True)
base = modal.Image.debian_slim(python_version="3.11").build(app)
print(base.object_id)  # Record this immutable ID in base_image_id.
```

```bash
python3 -m agent_harness prepare \
  --repo /absolute/path/to/solver-repository \
  --commit FULL_COMMIT_SHA \
  --recipe recipe.json \
  --out artifacts/environments/environment.json

python3 -m agent_harness validate \
  --manifest artifacts/environments/environment.json

python3 -m agent_harness run \
  --manifest artifacts/environments/environment.json \
  --episode-id example-attempt-1 -- python3 -c 'print("ready")'

python3 -m agent_harness smoke \
  --manifest artifacts/environments/environment.json
```

`qa-sandbox` is the equivalent installed entry point. Each command returns JSON.
`run` returns a nonzero CLI status when the executed command fails. `prepare`
refuses to overwrite an existing manifest. `validate` is local and does not prove
that the remote image still exists; fresh instance creation/readiness checks do.

The dependency files form an earlier image layer than the source snapshot, so
question changes cause no builds and source changes need not reinstall the same
dependencies. Builds run remotely and may access package registries. Use locked
versions/hashes for packages and external downloads; pinning the base image alone
does not freeze downloads made by build commands. Build timeouts are currently
provider-managed (`Image.run_commands` has no timeout argument in Modal 1.5.5).

The manifest records source hash, exact commit, normalized recipe, concrete Modal
image ID, readiness results, and SDK version. Its environment hash includes image
identity: rebuilding mutable package inputs cannot silently retain an old
environment ID. It is an integrity check, not a signature; only accept manifests
from trusted preparation storage. Repository imports and builds never run on the
controller host.

## Use in a rollout controller

```python
import json
from agent_harness.modal_backend import ModalSandboxBackend, SandboxLimits

manifest = json.load(open("artifacts/environments/environment.json"))
backend = ModalSandboxBackend(
    manifest,
    artifact_dir="artifacts/sandbox-events",
    limits=SandboxLimits(lifetime_seconds=1800, max_tool_calls=30),
    max_concurrency=8,
)

with backend.episode("run1-task2-attempt3") as episode:
    observation = episode.execute(["python3", "-m", "pytest", "-q"], timeout_seconds=60)
    # Feed observation to the policy; preserve sampled tokens/logprobs externally.
    # Save the answer and full trajectory to controller-owned artifact storage.
# Remote compute has been terminated before grading starts.
```

The backend reuses the built image, never a previously used sandbox. Each instance
initializes its own finite setup commands and repeats readiness checks, with
network blocked. Setup can create fixtures; services requiring supervised
background processes or sidecars need a future adapter. Retain one sandbox for
all tools in an attempt. `episode()` guarantees cleanup on exceptions. Low-level
`create()` users must call `close()` in `finally`.

Resource requests and hard limits, command deadlines, output retention caps, and
tool-call/lifetime budgets are explicit `SandboxLimits` fields. The CLI accepts
these through `--limits limits.json`. Per-task/run budget intersection belongs to
the trusted dispatcher; it must provide the stricter effective values.

The backend is a low-level API for an authorized dispatcher. It does not implement
the tool registry, policy sampling, grading, or Tinker updates. Passing arbitrary
model-selected SDK arguments is unsupported. No host mounts, model/Modal/grader
credentials, private task fields, or shared mutable volumes are injected. The
source copy is writable within each isolated attempt; the verifier must use its
own pristine snapshot.

Calls within an episode are serialized. Multiple threads can run independent
episodes on a shared backend. Capacity is bounded per backend instance and excess
creation fails immediately; a remote coordinator must also bound global sampling,
execution, and grading concurrency. There is no warm pool or cross-host scheduler
in this change.

## Failure handling and records

Journals are append-only, flushed/fsynced JSONL on the trusted controller, created
exclusively by episode ID. Use a new attempt ID for retries. They record image and
environment identity, limits, sandbox ID, setup/readiness, commands, results,
provisioning duration, errors, and teardown. Output is untrusted text; byte counts
and elapsed time are host measurements. Streams drain concurrently and retain only
the configured bytes per stream, with explicit truncation flags. These journals
are execution diagnostics, not complete signed `EpisodeMetrics`, model trajectories,
or a durable multi-host artifact store; the orchestrator must persist those before
training.

Nonzero command exit statuses are observations. Infrastructure failures raise
`SandboxInfrastructureError` or `TimeoutError` and must remain unresolved for
grading. Modal's negative exit sentinel is an unknown outcome, not a scored
command failure. Host deadlines can also expire while waiting for provisioning
or transport, so they are recorded as infrastructure errors. No command is
automatically replayed after a lost response. A command deadline ends the whole
episode to discard potentially surviving child processes. `BudgetExceeded` denotes
host-enforced tool/lifetime exhaustion; higher-level task policy determines rewards.
Provisioning and readiness are separately recorded and do not consume tool calls.

Successful `close()` is idempotent. Failed termination is recorded with sandbox ID
and retains capacity until cleanup succeeds. Creation failures with incomplete
cleanup are available through `backend.pending_cleanup_ids`; retry them with
`backend.retry_cleanup(episode_id)`. Other owners can retry `episode.close()`.
If the controller is lost, use the recorded sandbox ID to terminate through Modal;
the configured maximum lifetime is a final backstop. Ambiguous creation failures
are never retried automatically; inspect the app's episode tags before retrying.

Modal API references: [images](https://modal.com/docs/sdk/py/latest/Image),
[sandboxes](https://modal.com/docs/sdk/py/latest/Sandbox),
[command execution](https://modal.com/docs/guide/sandbox-spawn).
