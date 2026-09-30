"""Docker environments: tracked source only, isolated fresh containers per operation.

Build-time network is allowed to install dependencies. Runtime networking, host
mounts, privileges, and credentials are never passed into containers.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import os
import posixpath
from pathlib import Path
import selectors
import shlex
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from functools import wraps


class EnvironmentError(RuntimeError):
    """Infrastructure failure, never an incorrect-answer result."""


def _infrastructure_errors(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except (EnvironmentError, ValueError):
            raise
        except Exception as exc:
            raise EnvironmentError(f"{function.__name__}: {type(exc).__name__}: {exc}") from exc
    return wrapped


@dataclass(frozen=True)
class RunLimits:
    timeout_seconds: float = 30
    memory_mb: int = 512
    cpus: float = 1
    pids: int = 64
    output_bytes: int = 65536

    def __post_init__(self):
        if any(x <= 0 for x in (self.timeout_seconds, self.memory_mb, self.cpus,
                                self.pids, self.output_bytes)):
            raise ValueError("All resource limits must be positive")


@dataclass(frozen=True)
class EnvironmentRecipe:
    base_image: str = "python:3.12-slim"
    install_commands: list[list[str]] = field(default_factory=list)
    readiness_command: list[str] = field(default_factory=lambda: ["python", "--version"])
    capability: str = "source_reading"

    def __post_init__(self):
        if self.capability not in ("source_reading", "execution"):
            raise ValueError("Unsupported environment capability")
        if not self.base_image or self.base_image.startswith("-") or "\n" in self.base_image:
            raise ValueError("Invalid base image")
        for command in [*self.install_commands, self.readiness_command]:
            _validate_command(command)


def _validate_command(command):
    if not command or not all(isinstance(x, str) and "\0" not in x for x in command):
        raise ValueError("Commands must be nonempty argv lists")


def _capture(command, *, timeout_seconds, output_bytes, stdin=None):
    """Drain both streams while retaining only a bounded combined output."""
    started = time.monotonic()
    retained = {"stdout": bytearray(), "stderr": bytearray()}
    truncated = False
    timed_out = False
    # A file avoids blocking on input before output is drained.
    with tempfile.TemporaryFile() as input_file:
        if stdin is not None:
            input_file.write(stdin.encode())
        input_file.seek(0)
        try:
            process = subprocess.Popen(command, stdin=input_file, stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE)
        except OSError as exc:
            raise EnvironmentError(str(exc)) from exc
        with selectors.DefaultSelector() as selector:
            for name, stream in (("stdout", process.stdout), ("stderr", process.stderr)):
                selector.register(stream, selectors.EVENT_READ, name)
            try:
                while selector.get_map():
                    if time.monotonic() - started > timeout_seconds:
                        timed_out = True
                        process.kill()
                        break
                    for key, _ in selector.select(timeout=0.05):
                        chunk = os.read(key.fileobj.fileno(), 16384)
                        if not chunk:
                            selector.unregister(key.fileobj)
                            continue
                        remaining = output_bytes - sum(map(len, retained.values()))
                        retained[key.data].extend(chunk[:remaining])
                        truncated |= len(chunk) > remaining
                process.wait(timeout=max(0.1, timeout_seconds - (time.monotonic() - started)))
            except subprocess.TimeoutExpired:
                timed_out = True
                process.kill()
                process.wait()
            finally:
                for stream in (process.stdout, process.stderr):
                    stream.close()
    return {**{key: value.decode("utf-8", errors="replace") for key, value in retained.items()},
            "exit_code": process.returncode, "timed_out": timed_out,
            "truncated": truncated, "duration_seconds": time.monotonic() - started}


class DockerBackend:
    def __init__(self, executable="docker"):
        self.executable = executable

    def _command(self, args, timeout=30):
        return _capture([self.executable, *args], timeout_seconds=timeout,
                        output_bytes=65536)

    def available(self):
        try:
            result = self._command(["info", "--format", "{{.ServerVersion}}"], timeout=10)
            return (result["exit_code"] == 0 and not result["timed_out"]
                    and bool(result["stdout"].strip().strip('"'))
                    and "Cannot connect" not in result["stderr"])
        except EnvironmentError:
            return False

    def run(self, image_id: str, command: list[str], limits: RunLimits | None = None,
            stdin: str | None = None):
        _validate_command(command)
        if not image_id or image_id.startswith("-"):
            raise ValueError("Invalid image reference")
        limits = limits or RunLimits()
        name = "dataset-attempt-" + uuid.uuid4().hex
        args = [self.executable, "run", "--rm", "--name", name, "--network", "none",
                "--read-only", "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
                "--memory", f"{limits.memory_mb}m", "--memory-swap", f"{limits.memory_mb}m",
                "--cpus", str(limits.cpus), "--pids-limit", str(limits.pids),
                "--user", "65534:65534", "--workdir", "/workspace",
                "--tmpfs", "/tmp:rw,nosuid,nodev,size=64m", "--env", "HOME=/tmp",
                "--env", "PYTHONDONTWRITEBYTECODE=1", "--interactive",
                "--entrypoint", command[0], image_id, *command[1:]]
        try:
            return _capture(args, timeout_seconds=limits.timeout_seconds,
                            output_bytes=limits.output_bytes, stdin=stdin)
        finally:
            # Killing docker's client alone does not terminate the container.
            self._command(["rm", "--force", name], timeout=10)


def _git(snapshot, *args):
    result = subprocess.run(["git", "-C", str(snapshot), *args], capture_output=True,
                            check=True, timeout=30)
    return result.stdout


def _git_blobs(snapshot, object_ids):
    """Read distinct blobs in one Git process with binary-safe batch framing."""
    ids = list(dict.fromkeys(object_ids))
    if not ids:
        return {}
    result = subprocess.run(
        ["git", "-C", str(snapshot), "cat-file", "--batch"],
        input=("\n".join(ids) + "\n").encode("ascii"), capture_output=True,
        check=True, timeout=30)
    data, offset, blobs = result.stdout, 0, {}
    for expected in ids:
        end = data.find(b"\n", offset)
        if end < 0:
            raise EnvironmentError("Truncated Git batch header")
        header = data[offset:end].split()
        if len(header) != 3 or header[0].decode("ascii") != expected or header[1] != b"blob":
            raise EnvironmentError("Unexpected Git batch object")
        try:
            size = int(header[2])
        except ValueError as exc:
            raise EnvironmentError("Invalid Git batch size") from exc
        offset = end + 1
        if size < 0 or offset + size >= len(data) or data[offset + size:offset + size + 1] != b"\n":
            raise EnvironmentError("Truncated Git batch blob")
        blobs[expected] = data[offset:offset + size]
        offset += size + 1
    if offset != len(data):
        raise EnvironmentError("Unexpected trailing Git batch data")
    return blobs


def _export_snapshot(snapshot, commit, source=None):
    """Export tracked Git blobs; materialize internal tracked file/tree links.

    The image builder need not follow arbitrary host symlinks. Link metadata is
    preserved in the environment manifest; hashes describe the resolved bytes
    tools observe. Internal tracked directories are expanded; external links, cycles, and submodules fail.
    """
    entries = {}
    for entry in _git(snapshot, "ls-tree", "-rz", "--full-tree", commit).split(b"\0"):
        if not entry:
            continue
        metadata, raw_path = entry.split(b"\t", 1)
        mode, kind, object_id = metadata.decode().split()
        path = raw_path.decode("utf-8")
        if kind != "blob" or mode not in ("100644", "100755", "120000"):
            raise EnvironmentError(f"Unsupported repository object: {path}")
        if path.startswith("/") or ".." in path.split("/") or "\\" in path:
            raise EnvironmentError("Unsafe repository path")
        entries[path] = (mode, object_id)
    blobs = _git_blobs(snapshot, (entry[1] for entry in entries.values()))
    symlinks, hashes = {}, {}

    def content(path):
        return blobs[entries[path][1]]

    exported = {}

    def expand(path, output_path, seen=()):
        if path not in entries or path in seen:
            raise EnvironmentError("Symlink must resolve to a tracked internal file/tree")
        mode, _ = entries[path]
        if mode != "120000":
            exported[output_path] = (content(path), mode)
            return path
        target = content(path).decode("utf-8")
        if target.startswith("/") or "\\" in target or "\0" in target:
            raise EnvironmentError("Unsafe symlink target")
        resolved = posixpath.normpath(posixpath.join(posixpath.dirname(path), target))
        if resolved.startswith("../") or resolved in ("..", "."):
            raise EnvironmentError("Unsafe symlink target")
        next_seen = (*seen, path)
        directory = resolved not in entries
        if directory:
            children = [p for p in entries if p.startswith(resolved + "/")]
            if not children:
                raise EnvironmentError("Symlink target is not a tracked tree")
            for child in children:
                suffix = child[len(resolved) + 1:]
                expand(child, output_path + "/" + suffix, next_seen)
            final_path = resolved
        else:
            final_path = expand(resolved, output_path, next_seen)
        symlinks[path] = {"target": target, "resolved_path": final_path,
                          "materialized": True, "directory": directory}
        return final_path

    for path in entries:
        expand(path, path)
    for path, (data, mode) in exported.items():
        if source is not None:
            destination = source / path
            if not destination.resolve().is_relative_to(source.resolve()):
                raise EnvironmentError("Unsafe repository path")
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)
            destination.chmod(0o755 if mode == "100755" else 0o644)
        hashes[path] = hashlib.sha256(data).hexdigest()
    return hashes, symlinks


def build_environment(snapshot: Path, commit: str, recipe: EnvironmentRecipe,
                      output_dir: Path, backend: DockerBackend | None = None):
    """Build from git objects, not working-tree files; save manifest and logs.

    No private probes or grading data may be supplied as readiness/install commands.
    Private assertions are dispatched separately using run(..., stdin=...).
    """
    snapshot, output_dir = Path(snapshot), Path(output_dir)
    backend = backend or DockerBackend()
    if not backend.available():
        raise EnvironmentError("Docker daemon unavailable; executable tasks must remain quarantined")
    actual_commit = _git(snapshot, "rev-parse", "HEAD").decode().strip()
    if actual_commit != commit or len(commit) != 40:
        raise EnvironmentError("Snapshot HEAD does not match full pinned commit")
    pull = backend._command(["pull", recipe.base_image], timeout=600)
    if pull["exit_code"] or pull["timed_out"]:
        raise EnvironmentError("Base-image pull failed: " + pull["stderr"])
    inspect = backend._command(["image", "inspect", recipe.base_image,
                                "--format", "{{json .RepoDigests}}"])
    try:
        base_digest = json.loads(inspect["stdout"])[0]
    except (ValueError, IndexError, TypeError) as exc:
        raise EnvironmentError("Could not resolve immutable base-image digest") from exc
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="dataset-build-") as temporary:
        context = Path(temporary)
        source = context / "repo"
        source.mkdir()
        file_hashes, materialized_symlinks = _export_snapshot(snapshot, commit, source)
        dockerfile = f"FROM {base_digest}\nWORKDIR /workspace\nCOPY repo/ /workspace/\n"
        for command in recipe.install_commands:
            dockerfile += "RUN " + json.dumps(command) + "\n"
        dockerfile += "ENV PYTHONDONTWRITEBYTECODE=1\nUSER 65534:65534\n"
        (context / "Dockerfile").write_text(dockerfile)
        (output_dir / "Dockerfile").write_text(dockerfile)
        result = backend._command(["build", "--iidfile", str(context / "image-id"),
                                   str(context)], timeout=900)
        (output_dir / "build.json").write_text(json.dumps(result, indent=2) + "\n")
        if result["exit_code"] or result["timed_out"]:
            raise EnvironmentError("Image build failed; see build.json")
        image_id = (context / "image-id").read_text().strip()
    readiness = backend.run(image_id, recipe.readiness_command, RunLimits(timeout_seconds=60))
    manifest = {
        "schema_version": "1", "environment_id": "env-" + image_id.removeprefix("sha256:")[:16],
        "commit": commit, "capability": recipe.capability, "image_digest": image_id,
        "base_image_digest": base_digest, "snapshot_files": file_hashes,
        "materialized_symlinks": materialized_symlinks,
        "snapshot_sha256": hashlib.sha256(json.dumps(file_hashes, sort_keys=True).encode()).hexdigest(),
        "recipe_sha256": hashlib.sha256(dockerfile.encode()).hexdigest(),
        "install_commands": recipe.install_commands, "readiness_command": recipe.readiness_command,
        "readiness": readiness,
        "status": "ready" if readiness["exit_code"] == 0 and not readiness["timed_out"] and not readiness.get("truncated") else "quarantined",
        "backend": "docker", "tool_version": "dataset-docker-v1",
        "runtime_policy": {"network": "none", "host_mounts": False, "read_only": True,
                           "fresh_container_per_operation": True},
    }
    (output_dir / "environment.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def _modal_module():
    try:
        import modal
        from modal.config import config
    except ImportError as exc:
        raise EnvironmentError("Install the optional modal dependency first") from exc
    if not (config.get("token_id") and config.get("token_secret")):
        raise EnvironmentError("Modal authentication unavailable; run python -m modal token new")
    return modal


class ModalBackend:
    """Fresh gVisor-backed Modal Sandbox per operation, no volumes or secrets.

    Image IDs are reusable across client processes. Apps are ephemeral contexts,
    not deployed endpoints. The remote wrapper bounds output before transport.
    """
    def available(self):
        try:
            _modal_module()
            return True
        except EnvironmentError:
            return False

    @_infrastructure_errors
    def run(self, image_id: str, command: list[str], limits: RunLimits | None = None,
            stdin: str | None = None):
        _validate_command(command)
        limits = limits or RunLimits()
        modal = _modal_module()
        started = time.monotonic()
        # This source is our trusted runtime, not repository code executed locally.
        wrapper = (
            "import json, os, selectors, subprocess, tempfile, time, resource\n"
            "class EnvironmentError(RuntimeError): pass\n"
            + inspect.getsource(_capture)
            + "\npayload=json.loads(input())\n"
            + "import ctypes\n"
            + "libc=ctypes.CDLL(None, use_errno=True)\n"
            + "assert libc.prctl(38, 1, 0, 0, 0) == 0, 'no_new_privs failed'\n"
            + "if os.geteuid() == 0:\n    os.setgroups([])\n    os.setgid(65534)\n    os.setuid(65534)\n"
            + "assert os.geteuid() != 0, 'privilege drop failed'\n"
            + "resource.setrlimit(resource.RLIMIT_NPROC, (payload['pids'], payload['pids']))\n"
            + "resource.setrlimit(resource.RLIMIT_FSIZE, (67108864, 67108864))\n"
            + "result=_capture(payload['command'], timeout_seconds=payload['timeout'], "
              "output_bytes=payload['output_bytes'], stdin=payload['stdin'])\n"
            + "print(json.dumps(result))\n"
        )
        sandbox = None
        app = modal.App("dataset-environment-attempt")
        with app.run():
            try:
                sandbox = modal.Sandbox.create(
                    "python", "-I", "-c", wrapper, app=app,
                    image=modal.Image.from_id(image_id), block_network=True,
                    timeout=max(10, int(limits.timeout_seconds) + 15),
                    cpu=(limits.cpus, limits.cpus), memory=(limits.memory_mb, limits.memory_mb),
                    workdir="/workspace", env={"HOME": "/tmp", "PYTHONDONTWRITEBYTECODE": "1"},
                    secrets=[], volumes={}, include_oidc_identity_token=False,
                )
                payload = json.dumps({"command": command, "stdin": stdin,
                    "timeout": limits.timeout_seconds, "output_bytes": limits.output_bytes,
                    "pids": limits.pids}) + "\n"
                # Large source inventories can exceed Modal's local stdin buffer.
                # Drain bounded chunks before EOF; the remote input() still sees
                # exactly one complete JSON line and never a partial request.
                for offset in range(0, len(payload), 65536):
                    sandbox.stdin.write(payload[offset:offset + 65536])
                    sandbox.stdin.drain()
                sandbox.stdin.write_eof()
                sandbox.stdin.drain()
                stdout = sandbox.stdout.read()
                sandbox.wait()
                if sandbox.returncode != 0:
                    raise EnvironmentError("Modal runtime wrapper failed: " + sandbox.stderr.read()[:2000])
                result = json.loads(stdout)
                result["sandbox_id"] = sandbox.object_id
                result["provisioning_and_run_seconds"] = time.monotonic() - started
                return result
            finally:
                if sandbox is not None:
                    sandbox.terminate()


@_infrastructure_errors
def build_modal_environment(snapshot: Path, commit: str, recipe: EnvironmentRecipe,
                            output_dir: Path, backend: ModalBackend | None = None):
    """Build an immutable Modal image from only pinned Git blobs."""
    modal = _modal_module()
    backend = backend or ModalBackend()
    snapshot, output_dir = Path(snapshot), Path(output_dir)
    if len(commit) != 40 or _git(snapshot, "rev-parse", "HEAD").decode().strip() != commit:
        raise EnvironmentError("Snapshot HEAD does not match full pinned commit")
    output_dir.mkdir(parents=True, exist_ok=True)
    file_hashes = {}
    with tempfile.TemporaryDirectory(prefix="dataset-modal-build-") as temporary:
        source = Path(temporary) / "repo"
        source.mkdir()
        file_hashes, materialized_symlinks = _export_snapshot(snapshot, commit, source)
        image = modal.Image.from_registry(recipe.base_image).add_local_dir(
            source, "/workspace", copy=True).workdir("/workspace")
        if recipe.install_commands:
            image = image.run_commands(*(shlex.join(command) for command in recipe.install_commands))
        # Protect citation source from mutation; /tmp remains writable per sandbox.
        image = image.run_commands("chmod -R a-w /workspace").dockerfile_commands(
            "USER 65534:65534", "ENV PYTHONDONTWRITEBYTECODE=1")
        app = modal.App("dataset-environment-build")
        with app.run():
            image.build(app)
            image_id = image.object_id
    readiness = backend.run(image_id, recipe.readiness_command, RunLimits(timeout_seconds=60))
    recipe_record = {"base_image": recipe.base_image, "install_commands": recipe.install_commands,
                     "readiness_command": recipe.readiness_command}
    manifest = {
        "schema_version": "1", "environment_id": "env-" + image_id,
        "commit": commit, "capability": recipe.capability, "image_digest": image_id,
        "base_image_reference": recipe.base_image, "snapshot_files": file_hashes,
        "materialized_symlinks": materialized_symlinks,
        "snapshot_sha256": hashlib.sha256(json.dumps(file_hashes, sort_keys=True).encode()).hexdigest(),
        "recipe_sha256": hashlib.sha256(json.dumps(recipe_record, sort_keys=True).encode()).hexdigest(),
        **recipe_record, "readiness": readiness,
        "status": "ready" if readiness["exit_code"] == 0 and not readiness["timed_out"] and not readiness.get("truncated") else "quarantined",
        "backend": "modal", "tool_version": "dataset-modal-v2", "modal_version": modal.__version__,
        "runtime_policy": {"network": "blocked", "host_mounts": False, "secrets": False,
                           "source_read_only": True, "fresh_sandbox_per_operation": True},
    }
    (output_dir / "environment.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (output_dir / "recipe.json").write_text(json.dumps(recipe_record, indent=2) + "\n")
    return manifest
