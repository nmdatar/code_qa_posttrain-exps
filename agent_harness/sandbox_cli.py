"""Prepare images and exercise disposable Modal environments."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import uuid

from .images import prepare_environment, validate_manifest
from .modal_backend import ModalSandboxBackend, SandboxLimits


def load(path):
    from qa_eval.security import load_json
    return load_json(path)


def save_new(path, value):
    """Publish once; never replace a manifest referenced by existing tasks."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, indent=2, allow_nan=False) + "\n"
    # Link a completed temporary file atomically, refusing any existing target.
    import os
    import tempfile
    fd, temporary = tempfile.mkstemp(prefix=".manifest-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        os.unlink(temporary)


def smoke(backend, prefix=None):
    """Two sequential instances prove that one attempt's file is not reused."""
    prefix = prefix or f"smoke-{uuid.uuid4().hex}"
    marker = f"/tmp/{prefix}.marker"
    results = []
    for index in range(2):
        with backend.episode(f"{prefix}-{index}") as episode:
            script = ("from pathlib import Path; import sys; "
                      "p=Path(sys.argv[1]); "
                      "assert not p.exists(), 'state leaked between attempts'; "
                      "p.write_text('attempt'); print('isolated')")
            result = episode.execute(["python3", "-c", script, marker])
            if result.exit_code != 0:
                raise RuntimeError(f"Sandbox isolation smoke failed: {result.stderr}")
            results.append({"sandbox_id": episode.sandbox_id, "journal": str(episode.journal_path),
                            "result": asdict(result)})
    if results[0]["sandbox_id"] == results[1]["sandbox_id"]:
        raise RuntimeError("Backend reused a sandbox across attempts")
    return {"status": "passed", "environment_id": backend.manifest["environment_id"], "attempts": results}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare", help="Build pinned repository image and validate readiness on Modal")
    prepare.add_argument("--repo", required=True, type=Path)
    prepare.add_argument("--commit", required=True, help="Full commit SHA; no moving refs")
    prepare.add_argument("--recipe", required=True, type=Path)
    prepare.add_argument("--app", default="action-interview-environments")
    prepare.add_argument("--out", required=True, type=Path)
    check = sub.add_parser("validate", help="Validate a manifest locally, without allocating compute")
    check.add_argument("--manifest", required=True, type=Path)
    for command in ("run", "smoke"):
        run = sub.add_parser(command, help="Run a bounded command" if command == "run" else "Verify fresh-instance isolation")
        run.add_argument("--manifest", required=True, type=Path)
        run.add_argument("--events", type=Path, default=Path("artifacts/sandbox-events"))
        run.add_argument("--limits", type=Path, help="JSON fields for SandboxLimits")
        if command == "run":
            run.add_argument("--episode-id", default=None)
            run.add_argument("argv", nargs=argparse.REMAINDER, help="-- executable arg ...")
    args = parser.parse_args(argv)
    if args.command == "prepare":
        if args.out.exists():
            parser.error("Output already exists; use a new manifest path")
        manifest = prepare_environment(args.repo, args.commit, load(args.recipe), args.app)
        save_new(args.out, manifest)
        print(json.dumps({"manifest": str(args.out), "environment_id": manifest["environment_id"],
                          "image_id": manifest["image_id"]}))
        return 0
    manifest = load(args.manifest)
    validate_manifest(manifest)
    if args.command == "validate":
        print(json.dumps({"status": "valid", "environment_id": manifest["environment_id"]}))
        return 0
    limits = SandboxLimits(**load(args.limits)) if args.limits else SandboxLimits()
    backend = ModalSandboxBackend(manifest, args.events, limits=limits)
    if args.command == "smoke":
        print(json.dumps(smoke(backend), indent=2))
        return 0
    command = args.argv[1:] if args.argv[:1] == ["--"] else args.argv
    if not command:
        parser.error("run requires an executable after --")
    with backend.episode(args.episode_id or f"run-{uuid.uuid4().hex}") as episode:
        result = episode.execute(command)
    print(json.dumps({"sandbox_id": episode.sandbox_id, "journal": str(episode.journal_path),
                      "result": asdict(result)}, indent=2))
    return 0 if result.exit_code == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
