"""Generic sandbox contract and adapters for isolated repository execution."""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import inspect
import re
from pathlib import Path, PurePosixPath
import stat
from typing import Any, Mapping, Protocol

from dataset_builder.environment import DockerBackend, ModalBackend, RunLimits, EnvironmentError


def _source_digest(expected, root="/workspace"):
    """Hash only regular expected files without importing any repository code."""
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Invalid source root")
    observed = {}
    for name in sorted(expected):
        parts = name.split("/")
        if (not name or PurePosixPath(name).is_absolute() or "\\" in name
                or "\0" in name or any(part in ("", ".", "..") for part in parts)):
            raise ValueError("Unsafe source path")
        current = root
        for part in parts:
            current = current / part
            if current.is_symlink():
                raise ValueError("Source path is a symlink")
        if not stat.S_ISREG(current.stat().st_mode):
            raise ValueError("Source is not a regular file")
        digest = hashlib.sha256()
        with current.open("rb") as stream:
            for chunk in iter(lambda: stream.read(65536), b""):
                digest.update(chunk)
        observed[name] = digest.hexdigest()
    return hashlib.sha256(json.dumps(observed, sort_keys=True).encode()).hexdigest()


_SOURCE_VERIFIER = (
    "import hashlib, json, stat, sys\nfrom pathlib import Path, PurePosixPath\n"
    + inspect.getsource(_source_digest)
    + "\nprint(_source_digest(json.load(sys.stdin)))\n"
)


class SandboxBackend(Protocol):
    """Host-provided backend. Implementations must never run payloads on the host."""

    name: str

    def run(self, image_id: str, command: list[str], limits: RunLimits,
            stdin: str | None = None) -> dict[str, Any]: ...


class DockerSandbox:
    name = "docker"

    def __init__(self, backend: DockerBackend | None = None):
        self.backend = backend or DockerBackend()

    def run(self, image_id, command, limits, stdin=None):
        return self.backend.run(image_id, command, limits, stdin=stdin)


class ModalSandbox:
    name = "modal"

    def __init__(self, backend: ModalBackend | None = None):
        self.backend = backend or ModalBackend()

    def run(self, image_id, command, limits, stdin=None):
        return self.backend.run(image_id, command, limits, stdin=stdin)


@dataclass(frozen=True)
class ExecutionEnvironment:
    """Trusted configuration selected by the host, never by model arguments."""

    backend: SandboxBackend
    manifest: Mapping[str, Any]
    test_runner: tuple[str, ...] = ("python", "-m", "pytest", "-p", "no:cacheprovider")
    python_executable: str = "python"
    limits: RunLimits = field(default_factory=RunLimits)

    def verify(self, repository) -> None:
        manifest = self.manifest
        if manifest.get("status") != "ready" or manifest.get("capability") != "execution":
            raise ValueError("Environment must be ready and execution-capable")
        if manifest.get("backend") != self.backend.name:
            raise ValueError("Environment backend mismatch")
        if manifest.get("commit") != repository.commit:
            raise ValueError("Environment commit mismatch")
        image_id = manifest.get("image_digest")
        if not isinstance(image_id, str) or not image_id:
            raise ValueError("Environment image is missing")
        if self.backend.name == "docker" and not re.fullmatch(r"sha256:[0-9a-f]{64}", image_id):
            raise ValueError("Docker requires an immutable full image ID")
        if self.backend.name == "modal" and not re.fullmatch(r"im-[A-Za-z0-9_-]+", image_id):
            raise ValueError("Modal requires an immutable image ID")
        files = repository.snapshot_hashes()
        digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
        if manifest.get("snapshot_files") != files or manifest.get("snapshot_sha256") != digest:
            raise ValueError("Environment snapshot does not match pinned source")
        readiness = manifest.get("readiness", {})
        if readiness.get("exit_code") != 0 or readiness.get("timed_out") or readiness.get("truncated"):
            raise ValueError("Environment readiness failed")
        policy = manifest.get("runtime_policy", {})
        if policy.get("host_mounts") is not False or policy.get("network") not in ("none", "blocked"):
            raise ValueError("Environment does not declare isolated execution")
        if not (policy.get("read_only") is True or policy.get("source_read_only") is True):
            raise ValueError("Environment source must be read-only")
        for command in (self.test_runner, (self.python_executable,)):
            if not command or any(not isinstance(x, str) or not x or "\0" in x for x in command):
                raise ValueError("Invalid host-configured runner")

    def run(self, repository, command: list[str], *, stdin: str | None = None):
        try:
            self.verify(repository)
        except ValueError as exc:
            raise EnvironmentError(str(exc)) from exc
        # Verify the post-install immutable image, not just the build manifest.
        # -I excludes cwd/PYTHONPATH and -S disables installed startup hooks.
        verification = self.backend.run(self.manifest["image_digest"],
            [self.python_executable, "-I", "-S", "-c", _SOURCE_VERIFIER], self.limits,
            stdin=json.dumps(self.manifest["snapshot_files"], sort_keys=True))
        if (verification.get("exit_code") != 0 or verification.get("timed_out")
                or verification.get("truncated")
                or verification.get("stdout", "").strip() != self.manifest["snapshot_sha256"]):
            raise EnvironmentError("Runtime image source does not match the pinned snapshot")
        result = self.backend.run(self.manifest["image_digest"], command, self.limits, stdin=stdin)
        result = dict(result)
        for stream in ("stdout", "stderr"):
            result[stream + "_sha256"] = hashlib.sha256(result.get(stream, "").encode()).hexdigest()
        return result
