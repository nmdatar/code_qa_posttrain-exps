# Research harness

The `agent_harness` package runs bounded, single-agent research episodes. Its core does not know about repositories, grading rubrics, model providers, or RL optimizers. Code understanding is the first tool bundle.

Run complete research episodes and private grading remotely using [the Modal deployment guide](REMOTE_HARNESS.md). The remote coordinator supports bounded concurrent workers, durable results, and HTTP or Tinker sampling. The local CLI below remains useful for development.

## Try it

Python 3.11+ and Git are sufficient for an offline smoke run:

```bash
python3 -m agent_harness demo --output /tmp/harness-smoke
python3 -m unittest discover -s tests -v
```

The demo uses scripted list/read/final actions. It verifies mechanics; it is not a model-quality result. You can supply `--repo /path/to/repository --commit FULL_SHA` to use a real pinned snapshot.

To investigate a repository with a configured model:

```bash
# Set AGENT_API_KEY in your environment when the endpoint requires it.
python3 -m agent_harness run \
  --repo /path/to/repository --commit FULL_SHA \
  --question 'How does request validation work?' \
  --model YOUR_MODEL_VERSION --base-url https://YOUR_PROVIDER/v1 \
  --output /tmp/research-run
```

Use `--task /path/to/task.json` instead of `--question` for an existing private QA TaskSpec. Only its allowlisted public question, task ID, tool permissions and budgets reach the policy. Keep the task file outside the investigated repository. The CLI validates AnswerSubmission structure, not semantic correctness.

HTTP requests have no automatic retries. Missing token usage and cost remain unknown. Set both `--input-price-per-million` and `--output-price-per-million` to record a configured token-cost estimate. This is not a provider invoice or sandbox-cost measurement.

## Add a domain or tool

Implement a tool object with `spec: ToolSpec` and `execute(arguments, context) -> ToolObservation`. Register it with `ToolRegistry.register(tool)`, or expose a factory through the `agent_harness.tools` Python entry-point group and explicitly select its name with `registry.load_plugins([...])`.

```python
from agent_harness import ToolSpec, ToolObservation, ToolRegistry

class Lookup:
    spec = ToolSpec(
        name='lookup', version='1', type='catalog', capabilities=('read',),
        input_schema={'type': 'object', 'properties': {'key': {'type': 'string'}},
                      'required': ['key'], 'additionalProperties': False},
        required_resources=('catalog',),
    )

    def execute(self, arguments, context):
        return ToolObservation('ok', context.resources['catalog'].get(arguments['key']))

registry = ToolRegistry([Lookup()])
```

The host supplies resource handles in ResearchRequest. A tool receives only the handles declared in `required_resources`; policy arguments cannot grant a new resource. Filter with `allowed_types`, `allowed_capabilities`, and `permitted_tools`. `None` means unrestricted within the registered/configured set; an empty set denies access. Every declared capability must be allowed. The same checks run at dispatch.

Tool types are extensible strings. The registry validates an explicit JSON Schema subset (objects, arrays, primitive types, enum/const, bounds, patterns); unsupported keywords fail registration. Installed Python plugins are trusted host code, not sandboxed third-party code.

## Code understanding and execution

The initial bundle provides `list_files`, `search_code`, `read_file`, `find_symbols`, and `read_artifact`. Search is literal and case-sensitive; use its returned file/line pagination cursor. Symbol lookup currently supports Python. Other languages still support file search and reading.

Source reads use immutable Git blobs from the full pinned commit, ignoring checkout modifications. Symlinks and submodules are rejected. Results carry source locations and hashes. Larger results are stored as episode-scoped artifacts with bounded retrieval.

Add `--sandbox-manifest /path/to/environment.json` to enable `run_tests` and `python_probe`. The manifest must come from the existing dataset environment builder and bind a ready execution image to the same source hashes and commit. Docker and Modal adapters reuse that builder's runtime isolation. Execution is never performed by a local shell in the harness process. Each test/probe first runs a trusted source-hash verifier in isolation, then runs the requested command in another fresh sandbox from the same immutable image ID. A mismatch prevents execution. These two operations add startup overhead but keep post-install source identity explicit. Mutable interpreter/test state does not persist across calls.

The default test runner is pytest; the Python API can configure another argv runner through ExecutionEnvironment. Dependencies must already exist in the built image. Modal requires its optional SDK and configured credentials on the trusted host. No SDK installation, image build, cloud deployment, or paid API run occurs during the offline demo.

## Results and evaluator integration

Each unique episode writes an append-only `trajectory.jsonl`, `metrics.json`, and content-addressed `artifacts/`. The CLI also writes `result.json`. Reusing an episode ID is rejected rather than overwriting prior records. Trajectories preserve observations, actions, context transformations, tool versions and available token metadata.

The loop enforces step, tool-call, generated-token, wall-time and context limits. Invalid tool attempts consume the call budget. Context compaction records removed payloads rather than silently erasing evidence. Completion, budget exhaustion, policy error, and infrastructure error remain distinct outcomes.

`agent_harness.qa_adapter.run_qa_episode` runs the trusted loop and binds its measured metrics to the existing QA submission/task hashes. It keeps the signing key and private task out of tool context. Unknown usage/cost blocks metrics signing. Failed policy episodes with no answer get an empty grading submission so malformed actions cannot evade grading as infrastructure exclusions. Private runtime fixtures remain the evaluator's responsibility; agent probes are not fixture attestations.

The existing metrics schema uses zero for unavailable first-token timing. This non-streaming adapter does not measure TTFT; that field must not be interpreted as an observed zero latency. Elapsed episode and tool times are measured independently.

## Current boundaries

- The synchronous runner uses POSIX main-thread deadlines. Modal workers run it in isolated subprocesses; the remote coordinator manages concurrency and persistent journals.
- The local artifact store is backed by separate durable Modal Volumes in the remote deployment. Ambiguous remote submissions become unresolved rather than being retried automatically.
- The HTTP adapter does not provide exact conditioning/generated token IDs and behavior log probabilities. The optional Tinker adapter records these when available, using explicit model and renderer configuration.
- No optimizer, checkpoint promotion, automatic policy selection, or live evaluator calibration is implemented by this package.
- Synthetic smoke runs validate mechanics. Real provider sampling, evaluator quality, and production throughput require separate measurements.

See [remote deployment and batch commands](REMOTE_HARNESS.md) and [the harness design](../requirements/AGENT_HARNESS.md) for the remaining RL roadmap.
