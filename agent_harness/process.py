"""Bounded, concurrent consumption of Modal command output.

The caller owns the sandbox and MUST terminate it after a timeout or transport
error: a lost response does not establish that the remote process stopped.
"""

from dataclasses import dataclass
import queue
import threading
import time


@dataclass(frozen=True)
class CommandResult:
    stdout: str
    stderr: str
    exit_code: int
    elapsed_seconds: float
    stdout_bytes: int
    stderr_bytes: int
    stdout_truncated: bool
    stderr_truncated: bool


def run_process(sandbox, argv, timeout_seconds, max_output_bytes, workdir="/repo"):
    """Run argv without host shell interpolation; retain at most cap per stream.

    Drain both streams even after the retention cap so verbose subprocesses do
    not deadlock. Daemon readers plus a host deadline also bound SDK/transport
    stalls. Output is untrusted; elapsed time and byte counts are host measured.
    """
    if (not isinstance(argv, (list, tuple)) or not argv
            or any(not isinstance(a, str) or "\0" in a for a in argv)
            or not argv[0]):
        raise ValueError("Command must be a nonempty argv list")
    if type(timeout_seconds) is not int or timeout_seconds <= 0:
        raise ValueError("Command timeout must be a positive integer")
    if type(max_output_bytes) is not int or max_output_bytes <= 0:
        raise ValueError("Output cap must be a positive integer")
    started = time.monotonic()
    completed = queue.Queue()

    def dispatch():
        try:
            process = sandbox.exec(*argv, timeout=timeout_seconds, workdir=workdir, text=False)
            completed.put(("process", process, None))
        except BaseException as exc:
            completed.put(("process", None, exc))

    def capture(name, stream):
        try:
            kept = bytearray()
            total = 0
            for chunk in stream:
                if isinstance(chunk, str):
                    chunk = chunk.encode("utf-8")
                total += len(chunk)
                kept.extend(chunk[:max(0, max_output_bytes - len(kept))])
            completed.put((name, (bytes(kept).decode("utf-8", errors="replace"), total), None))
        except BaseException as exc:
            completed.put((name, None, exc))

    def wait(process):
        try:
            completed.put(("exit_code", process.wait(), None))
        except BaseException as exc:
            completed.put(("exit_code", None, exc))

    threading.Thread(target=dispatch, daemon=True).start()
    values = {}
    while len(values) < 3:
        remaining = timeout_seconds - (time.monotonic() - started)
        if remaining <= 0:
            raise TimeoutError("Sandbox command exceeded host deadline")
        try:
            name, value, error = completed.get(timeout=remaining)
        except queue.Empty as exc:
            raise TimeoutError("Sandbox command exceeded host deadline") from exc
        if error is not None:
            raise error
        if name == "process":
            threading.Thread(target=capture, args=("stdout", value.stdout), daemon=True).start()
            threading.Thread(target=capture, args=("stderr", value.stderr), daemon=True).start()
            threading.Thread(target=wait, args=(value,), daemon=True).start()
        else:
            values[name] = value
    if type(values["exit_code"]) is not int:
        raise RuntimeError("Sandbox returned no trustworthy exit status")
    if values["exit_code"] < 0:
        # Modal converts ExecTimeoutError into -1 instead of raising it. Actual
        # signal exits use 128 + signal, so a negative value is never a known
        # command failure that can safely be graded or replayed.
        raise TimeoutError("Sandbox returned an unknown command outcome after timeout; do not replay")
    stdout, stdout_bytes = values["stdout"]
    stderr, stderr_bytes = values["stderr"]
    return CommandResult(stdout, stderr, values["exit_code"], time.monotonic() - started,
                         stdout_bytes, stderr_bytes, stdout_bytes > max_output_bytes,
                         stderr_bytes > max_output_bytes)
