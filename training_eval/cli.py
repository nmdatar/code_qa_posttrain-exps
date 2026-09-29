"""Offline demo and explicit adapter-based training/evaluation entry points."""
import argparse
import importlib
import json
from pathlib import Path

from .backends import FakeBackendFactory, TinkerBackendFactory
from .checkpoints import LocalCheckpointStore
from .contracts import ModelRef, RunSpec, StageSpec
from .evaluation import Evaluator
from .mocks import demo_inputs
from .model import ModelFactory
from .pipeline import Pipeline
from .tracking import JsonTracker, WandbSink


def _plugin(name, config):
    module, separator, attribute = name.partition(":")
    if not separator or not module or not attribute:
        raise ValueError("Plugin must be an importable module:function")
    result = getattr(importlib.import_module(module), attribute)(config)
    if not isinstance(result, dict):
        raise ValueError("Input plugin must return a dictionary of pipeline adapters")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo", help="Synthetic offline SFT → GRPO → checkpoint → eval")
    demo.add_argument("--output", required=True)
    demo.add_argument("--wandb-mode", choices=["disabled", "offline", "online"], default="disabled")
    for command in ("run", "evaluate"):
        p = commands.add_parser(command)
        p.add_argument("--config", required=True, help="JSON config; see TRAINING.md")
        p.add_argument("--output", required=True)
        p.add_argument("--plugin", required=True, help="module:function returning external adapters")
        if command == "run":
            p.add_argument("--resume", help="Committed checkpoint manifest; keep original run spec")
            p.add_argument("--stop-after-batches", type=int)
        else:
            p.add_argument("--final-test", action="store_true")
    args = parser.parse_args(argv)
    tracker = None
    try:
        if args.command == "demo":
            config = {"backend": {"kind": "fake"}, "model": {"base_model": "synthetic-bernoulli"},
                      "run": {"run_id": "synthetic-demo", "seed": 7, "max_tokens": 8, "max_turns": 2,
                              "stages": [{"strategy": "sft", "steps": 2, "learning_rate": 0.1},
                                         {"strategy": "grpo", "steps": 3, "learning_rate": 0.1,
                                          "batch_size": 2, "group_size": 4}]},
                      "tracking": {"mode": args.wandb_mode, "project": "training-eval-demo"}}
            inputs = demo_inputs()
        else:
            config = json.loads(Path(args.config).read_text())
            inputs = _plugin(args.plugin, config)
        output = Path(args.output).resolve()
        run = dict(config["run"])
        run["stages"] = tuple(StageSpec(**s) for s in run.get("stages", []))
        spec = RunSpec(**run)
        store = LocalCheckpointStore(output / "checkpoints")
        tracking = config.get("tracking", {})
        mode = tracking.get("mode", "disabled")
        if mode not in {"disabled", "offline", "online"}:
            raise ValueError("Invalid tracking mode")
        organization = {key: tracking[key] for key in (
            "experiment_id", "run_name", "tags", "notes", "source_run_id", "metadata") if key in tracking}
        organization["job_type"] = tracking.get("job_type", "evaluation" if args.command == "evaluate" else "training")
        for key in ("experiment_id", "run_name", "notes", "source_run_id", "job_type"):
            if key in organization and (not isinstance(organization[key], str) or not organization[key].strip()):
                raise ValueError(f"tracking.{key} must be a nonempty string")
        if "tags" in organization and (not isinstance(organization["tags"], list)
                or any(not isinstance(tag, str) or not tag.strip() for tag in organization["tags"])):
            raise ValueError("tracking.tags must be a list of nonempty strings")
        if "metadata" in organization and not isinstance(organization["metadata"], dict):
            raise ValueError("tracking.metadata must be an object")
        sink = (WandbSink(tracking["project"], spec.run_id, mode=mode,
                          config={**spec.to_dict(), "organization": organization},
                          directory=output / "wandb", entity=tracking.get("entity"),
                          group=organization.get("experiment_id"), name=organization.get("run_name"),
                          job_type=organization["job_type"], tags=organization.get("tags"),
                          notes=organization.get("notes"))
                if mode != "disabled" else None)
        tracker = JsonTracker(output / "logs", spec.run_id, sink=sink)
        tracker.log("run_metadata", organization)
        backend_config = dict(config["backend"])
        kind = backend_config.pop("kind")
        if kind == "fake":
            backends = FakeBackendFactory(output / "backend", **backend_config)
        elif kind == "tinker":
            backends = TinkerBackendFactory(**backend_config)
        else:
            raise ValueError("backend.kind must be fake or tinker (custom backends use the Python API)")
        factory = ModelFactory(backends, store)
        reference = ModelRef(**config["model"])
        if args.command == "evaluate":
            bundle = factory.resolve(reference, "evaluate")
            evaluator = Evaluator(inputs["evaluation_tasks"], inputs["environment"], inputs["verifier"], tracker,
                                  run_id=spec.run_id, max_tokens=spec.max_tokens, max_turns=spec.max_turns,
                                  temperature=spec.temperature, seed=spec.seed, context_limit=bundle.backend.context_limit)
            result = evaluator.evaluate(bundle.policy, checkpoint=reference.checkpoint, final_test=args.final_test)
        else:
            purpose = "fork"
            if getattr(args, "resume", None):
                reference, purpose = ModelRef(checkpoint=args.resume), "resume"
            pipeline = Pipeline(factory, store, tracker, **inputs)
            result = pipeline.run(spec, reference, purpose=purpose,
                                  stop_after_batches=getattr(args, "stop_after_batches", None))
        # Keep the full RNG state in the checkpoint, not the console summary.
        summary = {k: v for k, v in result.items() if k not in {"state", "rows"}}
        if "state" in result:
            summary.update({k: result["state"][k] for k in ("optimizer_step", "attempted_batches")})
        tracker.close()
        if tracker.sink_errors:
            summary["tracking_warnings"] = sorted(set(tracker.sink_errors))
            summary["tracking_note"] = "External logging had failures; durable local events can be replayed."
        tracker = None
        (output / "result.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2))
        return 0
    except (ValueError, RuntimeError, KeyError, ImportError, OSError) as exc:
        parser.error(str(exc))
    finally:
        if tracker:
            tracker.close()


if __name__ == "__main__":
    raise SystemExit(main())
