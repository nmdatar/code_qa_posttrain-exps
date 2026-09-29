"""Modal environments with one disposable sandbox per attempt.

This is a low-level execution backend for a trusted dispatcher, not an agent
tool registry. Never expose its constructor, manifests, or arbitrary SDK options
to model output. No grader data, credentials, or host directories are mounted.
"""

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import importlib
import json
import math
import os
from pathlib import Path
import re
import threading
import time

from .process import run_process
from .stream_process import StreamProcess, SERVER


class BudgetExceeded(RuntimeError):
    """The episode's declared execution budget has been exhausted."""


class SandboxInfrastructureError(RuntimeError):
    """Unresolved infrastructure outcome; must not become a zero training reward."""


@dataclass(frozen=True)
class SandboxLimits:
    lifetime_seconds: int = 1800
    command_timeout_seconds: int = 60
    readiness_timeout_seconds: int = 120
    max_tool_calls: int = 30
    max_output_bytes: int = 65536
    cpu: float = 0.5
    cpu_limit: float = 2.0
    memory_mib: int = 1024
    memory_limit_mib: int = 2048

    def __post_init__(self):
        for name in ("lifetime_seconds", "command_timeout_seconds", "readiness_timeout_seconds",
                     "max_tool_calls", "max_output_bytes", "memory_mib", "memory_limit_mib"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.lifetime_seconds > 86400:
            raise ValueError("Modal sandbox lifetime cannot exceed 24 hours")
        for value in (self.cpu, self.cpu_limit):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError("CPU requests and limits must be finite and positive")
        if self.cpu > self.cpu_limit or self.memory_mib > self.memory_limit_mib:
            raise ValueError("Resource request exceeds hard limit")


class _Journal:
    def __init__(self, directory, episode_id):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / f"{episode_id}.jsonl"
        # An existing ID is a previous attempt, never silently replay it.
        self.stream = self.path.open("x", encoding="utf-8")

    def write(self, event, **fields):
        self.stream.write(json.dumps({"event": event, "time_unix": time.time(), **fields},
                                     allow_nan=False, sort_keys=True) + "\n")
        self.stream.flush()
        os.fsync(self.stream.fileno())


class ModalSandboxBackend:
    def __init__(self, manifest, artifact_dir, *, limits=None, max_concurrency=8, modal_module=None):
        from .images import validate_manifest
        validate_manifest(manifest)
        if type(max_concurrency) is not int or max_concurrency < 1:
            raise ValueError("max_concurrency must be a positive integer")
        # Copy trusted config so later caller mutations cannot change identity.
        self.manifest = json.loads(json.dumps(manifest))
        self.artifact_dir = Path(artifact_dir)
        self.limits = limits or SandboxLimits()
        self._slots = threading.BoundedSemaphore(max_concurrency)
        self._modal = modal_module
        self._pending_cleanup = {}
        self._cleanup_lock = threading.Lock()

    @property
    def pending_cleanup_ids(self):
        """Episode IDs whose failed creation still owns remote capacity."""
        with self._cleanup_lock:
            return tuple(sorted(self._pending_cleanup))

    def retry_cleanup(self, episode_id):
        """Retry termination of a retained failed creation, never its commands.

        Raises KeyError for an ID that is not pending. A failed termination
        remains pending and continues to occupy capacity. This registry is local
        to this backend; after a host restart, recover IDs from the journals.
        """
        with self._cleanup_lock:
            episode = self._pending_cleanup[episode_id]
        try:
            episode.close("infrastructure_error")
        finally:
            # Detach/journal failures after termination must not retain a closed
            # handle; close has already released its capacity in that case.
            if episode._closed:
                with self._cleanup_lock:
                    self._pending_cleanup.pop(episode_id, None)

    def create(self, episode_id):
        """Create and check a fresh instance; caller must close returned episode.

        Capacity is per backend instance. The coordinator must apply a global
        limit across hosts. No retry is made on an ambiguous creation failure.
        """
        if not isinstance(episode_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", episode_id):
            raise ValueError("episode_id must contain 1–100 letters, digits, underscores or hyphens")
        if not self._slots.acquire(blocking=False):
            raise RuntimeError("Sandbox concurrency limit reached")
        journal = None
        episode = None
        try:
            journal = _Journal(self.artifact_dir, episode_id)
            journal.write("creating", episode_id=episode_id, environment_id=self.manifest["environment_id"],
                          image_id=self.manifest["image_id"], limits=asdict(self.limits))
            modal = self._modal or importlib.import_module("modal")
            app = modal.App.lookup(self.manifest["app_name"], create_if_missing=True)
            started = time.monotonic()
            sb = modal.Sandbox.create(
                *(["python3", "-u", "-I", "-c", SERVER] if self.manifest["workspace_path"] == "/workspace" else []),
                app=app, image=modal.Image.from_id(self.manifest["image_id"]),
                workdir=self.manifest["workspace_path"], block_network=True, include_oidc_identity_token=False,
                timeout=self.limits.lifetime_seconds,
                cpu=(self.limits.cpu, self.limits.cpu_limit),
                memory=(self.limits.memory_mib, self.limits.memory_limit_mib),
                tags={"episode_id": episode_id, "environment_id": self.manifest["environment_id"]},
            )
            episode = ModalEpisode(sb, journal, self.limits, self._slots.release, started, self.manifest["workspace_path"])
            journal.write("created", sandbox_id=sb.object_id)
            recipe = self.manifest["recipe"]
            for phase, commands in (("setup", recipe.get("setup_commands", [])),
                                    ("readiness", recipe["readiness_commands"])):
                for argv in commands:
                    result = episode._run(argv, self.limits.readiness_timeout_seconds)
                    journal.write(phase, argv=argv, result=asdict(result))
                    if result.exit_code != 0:
                        raise SandboxInfrastructureError(f"Environment {phase} check failed")
            journal.write("ready", provisioning_seconds=time.monotonic() - started)
            return episode
        except BaseException as exc:
            try:
                if journal:
                    journal.write("infrastructure_error", error_type=type(exc).__name__)
            finally:
                if episode:
                    try:
                        episode.close("infrastructure_error")
                    except BaseException as cleanup:
                        if not episode._closed:
                            with self._cleanup_lock:
                                self._pending_cleanup[episode_id] = episode
                            exc.add_note(f"Cleanup failed; call retry_cleanup({episode_id!r}): {cleanup}")
                        else:
                            exc.add_note(f"Sandbox terminated but client cleanup failed: {cleanup}")
                else:
                    if journal:
                        journal.stream.close()
                    self._slots.release()
            raise

    @contextmanager
    def episode(self, episode_id):
        episode = self.create(episode_id)
        try:
            yield episode
        except BaseException as exc:
            try:
                episode.close("interrupted" if isinstance(exc, (KeyboardInterrupt, SystemExit)) else "error")
            except BaseException as cleanup:
                exc.add_note(f"Cleanup failed; see {episode.journal_path}: {cleanup}")
            raise
        else:
            episode.close()


class ModalEpisode:
    def __init__(self, sandbox, journal, limits, release, started, workspace_path="/repo"):
        self._workspace_path = workspace_path
        self._stream_process = StreamProcess(sandbox) if workspace_path == "/workspace" else None
        self.sandbox_id = sandbox.object_id
        self.journal_path = journal.path
        self._sandbox, self._journal, self._limits = sandbox, journal, limits
        self._release, self._started = release, started
        self._closed = False
        self._calls = 0
        self._lock = threading.RLock()

    def _run(self, argv, timeout):
        remaining = self._limits.lifetime_seconds - (time.monotonic() - self._started)
        if remaining < 1:
            raise BudgetExceeded("Sandbox lifetime exhausted")
        if self._stream_process:
            return self._stream_process.run(argv, min(timeout, int(remaining)), self._limits.max_output_bytes)
        return run_process(self._sandbox, argv, min(timeout, int(remaining)), self._limits.max_output_bytes, workdir=self._workspace_path)

    def execute(self, argv, *, timeout_seconds=None):
        """Execute authorized argv; nonzero exits are observations, not outages.

        The dispatcher must enforce allowed tools/path policy before this call.
        A timeout or lost SDK response destroys the episode (no tool replay).
        """
        if (not isinstance(argv, (list, tuple)) or not argv or not argv[0]
                or any(not isinstance(a, str) or "\0" in a for a in argv)):
            raise ValueError("Command must be a nonempty argv list")
        timeout = self._limits.command_timeout_seconds if timeout_seconds is None else timeout_seconds
        if type(timeout) is not int or not 0 < timeout <= self._limits.command_timeout_seconds:
            raise ValueError("Requested timeout exceeds command budget")
        with self._lock:
            if self._closed:
                raise RuntimeError("Episode is closed")
            if self._calls >= self._limits.max_tool_calls:
                self.close("budget_exhausted")
                raise BudgetExceeded("Tool-call budget exhausted")
            self._calls += 1
            try:
                self._journal.write("command_started", sequence=self._calls, argv=list(argv))
                result = self._run(argv, timeout)
                self._journal.write("command_finished", sequence=self._calls, result=asdict(result))
                return result
            except BaseException as exc:
                reason = "budget_exhausted" if isinstance(exc, BudgetExceeded) else "infrastructure_error"
                try:
                    self._journal.write(reason, sequence=self._calls, error_type=type(exc).__name__)
                finally:
                    try:
                        self.close(reason)
                    except BaseException as cleanup:
                        exc.add_note(f"Cleanup failed for {self.sandbox_id}: {cleanup}")
                if isinstance(exc, (BudgetExceeded, TimeoutError, KeyboardInterrupt, SystemExit)):
                    raise
                raise SandboxInfrastructureError("Command outcome unknown; do not replay or grade as failure") from exc

    def close(self, reason="completed"):
        """Terminate remote compute before detaching; repeated success is a no-op.

        A failed termination is recorded and raises; retain the handle/capacity
        so close can be retried. The remote lifetime remains a final backstop.
        """
        with self._lock:
            if self._closed:
                return
            try:
                self._sandbox.terminate(wait=True)
            except BaseException as exc:
                self._journal.write("cleanup_failed", sandbox_id=self.sandbox_id,
                                    error_type=type(exc).__name__)
                raise SandboxInfrastructureError(f"Could not terminate sandbox {self.sandbox_id}") from exc
            try:
                self._sandbox.detach()
            finally:
                self._closed = True
                try:
                    self._journal.write("closed", reason=reason, sandbox_id=self.sandbox_id,
                                        elapsed_seconds=time.monotonic() - self._started)
                finally:
                    self._journal.stream.close()
                    self._release()
