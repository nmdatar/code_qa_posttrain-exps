"""Offline execution/lifecycle contracts; no Modal credentials or calls required."""

import hashlib
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from agent_harness.images import normalize_recipe
from agent_harness.modal_backend import (
    BudgetExceeded, ModalSandboxBackend, SandboxInfrastructureError, SandboxLimits,
)
from agent_harness.process import run_process


def manifest():
    recipe = normalize_recipe({
        "base_image": "python@sha256:" + "a" * 64,
        "mode": "source_reading", "readiness_commands": [["test", "-d", "/repo"]],
    })
    value = {
        "schema_version": 1, "status": "ready", "commit": "a" * 40,
        "source_sha256": "b" * 64, "recipe": recipe, "workspace_path": "/repo",
        "app_name": "offline-tests", "image_id": "im-prepared",
        "readiness": [{"command": recipe["readiness_commands"][0], "exit_code": 0}],
        "private_metadata": {"rubric": "DO NOT UPLOAD THIS RUBRIC"},
    }
    identity = {key: value[key] for key in (
        "schema_version", "image_id", "commit", "source_sha256", "recipe", "workspace_path",
    )}
    value["environment_id"] = hashlib.sha256(json.dumps(
        identity, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode()).hexdigest()
    return value


def process(stdout=(b"ok",), stderr=(), code=0):
    return SimpleNamespace(stdout=iter(stdout), stderr=iter(stderr), wait=Mock(return_value=code))


class FakeSandbox:
    def __init__(self, identity):
        self.object_id = identity
        self.exec = Mock(side_effect=lambda *args, **kwargs: process())
        self.terminate = Mock()
        self.detach = Mock()


class FakeModal:
    def __init__(self):
        self.instances = []
        self.App = SimpleNamespace(lookup=Mock(return_value="app"))
        self.Image = SimpleNamespace(from_id=Mock(side_effect=lambda image_id: image_id))
        self.Sandbox = SimpleNamespace(create=Mock(side_effect=self.create))

    def create(self, **kwargs):
        sandbox = FakeSandbox(f"sb-{len(self.instances)}")
        self.instances.append(sandbox)
        return sandbox


class ModalBackendTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.modal = FakeModal()
        self.backend = self.make_backend()

    def make_backend(self, **kwargs):
        return ModalSandboxBackend(manifest(), self.temp.name, modal_module=self.modal, **kwargs)

    def events(self, episode_id):
        return [json.loads(line) for line in
                (Path(self.temp.name) / f"{episode_id}.jsonl").read_text().splitlines()]

    def test_fresh_instances_reuse_only_prepared_image(self):
        for name in ("first", "second"):
            with self.backend.episode(name) as episode:
                episode.execute(["cat", "source.py"])
        self.assertEqual(len(self.modal.instances), 2)
        self.assertIsNot(self.modal.instances[0], self.modal.instances[1])
        self.assertEqual(self.modal.Image.from_id.call_count, 2)
        for call in self.modal.Sandbox.create.call_args_list:
            self.assertEqual(call.kwargs["image"], "im-prepared")
            self.assertTrue(call.kwargs["block_network"])
            self.assertFalse(call.kwargs["include_oidc_identity_token"])
            self.assertEqual(call.kwargs["workdir"], "/repo")
            for field in ("secrets", "volumes", "network_file_systems", "env"):
                self.assertNotIn(field, call.kwargs)
            self.assertNotIn("DO NOT UPLOAD", repr(call))
        for sandbox in self.modal.instances:
            sandbox.terminate.assert_called_once_with(wait=True)
            sandbox.detach.assert_called_once_with()
        self.assertNotIn("DO NOT UPLOAD", repr(self.events("first")))

    def test_caller_manifest_mutation_does_not_change_image(self):
        original = manifest()
        backend = ModalSandboxBackend(original, self.temp.name, modal_module=self.modal)
        original["image_id"] = "im-mutated"
        original["recipe"]["readiness_commands"][0].append("mutated")
        with backend.episode("immutable"):
            pass
        self.modal.Image.from_id.assert_called_once_with("im-prepared")
        self.assertEqual(self.modal.instances[0].exec.call_args.args, ("test", "-d", "/repo"))

    def test_readiness_failure_terminates_and_releases_capacity(self):
        backend = self.make_backend(max_concurrency=1)
        def fail(**kwargs):
            sandbox = self.modal.create(**kwargs)
            sandbox.exec.side_effect = lambda *a, **kw: process(code=2)
            return sandbox
        self.modal.Sandbox.create.side_effect = fail
        with self.assertRaises(SandboxInfrastructureError):
            backend.create("failed")
        self.modal.instances[0].terminate.assert_called_once_with(wait=True)
        self.modal.Sandbox.create.side_effect = self.modal.create
        with backend.episode("next"):
            pass

    def setup_backend(self):
        value = manifest()
        value["recipe"]["setup_commands"] = [["mkdir", "-p", "/tmp/work"]]
        identity = {key: value[key] for key in (
            "schema_version", "image_id", "commit", "source_sha256", "recipe", "workspace_path",
        )}
        value["environment_id"] = hashlib.sha256(json.dumps(
            identity, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        ).encode()).hexdigest()
        backend = ModalSandboxBackend(value, self.temp.name, modal_module=self.modal,
                                      limits=SandboxLimits(max_tool_calls=1))
        return backend

    def test_setup_failure_prevents_readiness_and_terminates(self):
        backend = self.setup_backend()
        def fail(**kwargs):
            sandbox = self.modal.create(**kwargs)
            sandbox.exec.side_effect = lambda *a, **kw: process(code=2)
            return sandbox
        self.modal.Sandbox.create.side_effect = fail
        with self.assertRaisesRegex(SandboxInfrastructureError, "setup"):
            backend.create("setup_failure")
        sandbox = self.modal.instances[0]
        sandbox.exec.assert_called_once()
        self.assertEqual(sandbox.exec.call_args.args[0], "mkdir")
        sandbox.terminate.assert_called_once_with(wait=True)
        sandbox.detach.assert_called_once_with()

    def test_setup_runs_before_readiness_and_is_not_charged_as_tool(self):
        backend = self.setup_backend()
        with backend.episode("setup") as episode:
            episode.execute(["echo", "one"])
        calls = self.modal.instances[0].exec.call_args_list
        self.assertEqual([call.args[0] for call in calls], ["mkdir", "test", "echo"])

    def test_context_exception_preserved_and_sandbox_closed(self):
        with self.assertRaisesRegex(ValueError, "caller failure"):
            with self.backend.episode("exception"):
                raise ValueError("caller failure")
        self.modal.instances[0].terminate.assert_called_once_with(wait=True)
        self.assertEqual(self.events("exception")[-1]["reason"], "error")

    def test_transport_failure_is_unresolved_never_replayed(self):
        episode = self.backend.create("transport")
        sandbox = self.modal.instances[0]
        sandbox.exec.side_effect = ConnectionError("connection lost after dispatch")
        with self.assertRaises(SandboxInfrastructureError):
            episode.execute(["mutate"])
        self.assertEqual(sandbox.exec.call_count, 2)  # readiness plus one dispatch
        sandbox.terminate.assert_called_once_with(wait=True)
        with self.assertRaisesRegex(RuntimeError, "closed"):
            episode.execute(["mutate"])

    def test_timeout_closes_without_replay(self):
        episode = self.backend.create("timeout")
        sandbox = self.modal.instances[0]
        sandbox.exec.side_effect = TimeoutError("deadline")
        with self.assertRaises(TimeoutError):
            episode.execute(["slow"])
        self.assertEqual(sandbox.exec.call_count, 2)
        sandbox.terminate.assert_called_once_with(wait=True)
        self.assertEqual(self.events("timeout")[-1]["reason"], "infrastructure_error")

    def test_sdk_timeout_sentinel_closes_without_grading_as_failure(self):
        episode = self.backend.create("sdk_timeout")
        sandbox = self.modal.instances[0]
        sandbox.exec.side_effect = lambda *a, **kw: process(code=-1)
        with self.assertRaisesRegex(TimeoutError, "unknown command outcome"):
            episode.execute(["slow"])
        sandbox.terminate.assert_called_once_with(wait=True)
        self.assertEqual(self.events("sdk_timeout")[-1]["reason"], "infrastructure_error")

    def test_nonzero_exit_remains_observation(self):
        with self.backend.episode("nonzero") as episode:
            sandbox = self.modal.instances[0]
            sandbox.exec.side_effect = lambda *a, **kw: process(stderr=(b"failed",), code=3)
            result = episode.execute(["false"])
            self.assertEqual(result.exit_code, 3)
            self.assertEqual(result.stderr, "failed")
            sandbox.terminate.assert_not_called()

    def test_tool_budget_terminates_without_extra_dispatch(self):
        backend = self.make_backend(limits=SandboxLimits(max_tool_calls=1))
        episode = backend.create("budget")
        episode.execute(["true"])
        with self.assertRaises(BudgetExceeded):
            episode.execute(["true"])
        self.assertEqual(self.modal.instances[0].exec.call_count, 2)
        self.assertEqual(self.events("budget")[-1]["reason"], "budget_exhausted")

    def test_lifetime_budget_checked_before_dispatch(self):
        episode = self.backend.create("lifetime")
        episode._started -= 90000
        with self.assertRaises(BudgetExceeded):
            episode.execute(["true"])
        self.assertEqual(self.modal.instances[0].exec.call_count, 1)

    def test_capacity_and_duplicate_ids_do_not_leak_slots(self):
        backend = self.make_backend(max_concurrency=1)
        first = backend.create("unique")
        with self.assertRaisesRegex(RuntimeError, "concurrency"):
            backend.create("blocked")
        first.close()
        with self.assertRaises(FileExistsError):
            backend.create("unique")
        with backend.episode("next"):
            pass
        self.assertEqual(len(self.modal.instances), 2)

    def test_creation_error_is_not_retried_and_releases_slot(self):
        backend = self.make_backend(max_concurrency=1)
        self.modal.Sandbox.create.side_effect = ConnectionError("ambiguous creation")
        with self.assertRaises(ConnectionError):
            backend.create("ambiguous")
        self.assertEqual(self.modal.Sandbox.create.call_count, 1)
        self.modal.Sandbox.create.side_effect = self.modal.create
        with backend.episode("after"):
            pass

    def test_failed_close_retains_capacity_and_can_be_retried(self):
        backend = self.make_backend(max_concurrency=1)
        episode = backend.create("close")
        sandbox = self.modal.instances[0]
        sandbox.terminate.side_effect = [ConnectionError("offline"), None]
        with self.assertRaises(SandboxInfrastructureError):
            episode.close()
        sandbox.detach.assert_not_called()
        with self.assertRaisesRegex(RuntimeError, "concurrency"):
            backend.create("blocked")
        episode.close()
        episode.close()
        self.assertEqual(sandbox.terminate.call_count, 2)
        sandbox.detach.assert_called_once_with()
        with backend.episode("after"):
            pass

    def test_failed_creation_cleanup_is_recoverable_without_replay(self):
        backend = self.make_backend(max_concurrency=1)
        def fail(**kwargs):
            sandbox = self.modal.create(**kwargs)
            sandbox.exec.side_effect = lambda *a, **kw: process(code=2)
            sandbox.terminate.side_effect = [ConnectionError("offline"),
                                             ConnectionError("still offline"), None]
            return sandbox
        self.modal.Sandbox.create.side_effect = fail
        with self.assertRaises(SandboxInfrastructureError):
            backend.create("recover")
        self.assertEqual(backend.pending_cleanup_ids, ("recover",))
        with self.assertRaisesRegex(RuntimeError, "concurrency"):
            backend.create("blocked")
        with self.assertRaises(SandboxInfrastructureError):
            backend.retry_cleanup("recover")
        self.assertEqual(backend.pending_cleanup_ids, ("recover",))
        sandbox = self.modal.instances[0]
        sandbox.detach.assert_not_called()
        backend.retry_cleanup("recover")
        self.assertEqual(backend.pending_cleanup_ids, ())
        sandbox.detach.assert_called_once_with()
        self.assertEqual(sandbox.exec.call_count, 1)
        self.assertEqual(sandbox.terminate.call_count, 3)
        with self.assertRaises(KeyError):
            backend.retry_cleanup("recover")
        self.modal.Sandbox.create.side_effect = self.modal.create
        with backend.episode("after"):
            pass

    def test_creation_detach_failure_does_not_retain_terminated_handle(self):
        backend = self.make_backend(max_concurrency=1)
        def fail(**kwargs):
            sandbox = self.modal.create(**kwargs)
            sandbox.exec.side_effect = lambda *a, **kw: process(code=2)
            sandbox.detach.side_effect = ConnectionError("detach failed")
            return sandbox
        self.modal.Sandbox.create.side_effect = fail
        with self.assertRaises(SandboxInfrastructureError):
            backend.create("detached")
        self.assertEqual(backend.pending_cleanup_ids, ())
        self.modal.Sandbox.create.side_effect = self.modal.create
        with backend.episode("after"):
            pass

    def test_detach_error_still_releases_capacity_after_termination(self):
        backend = self.make_backend(max_concurrency=1)
        episode = backend.create("detach")
        self.modal.instances[0].detach.side_effect = ConnectionError("detach lost")
        with self.assertRaises(ConnectionError):
            episode.close()
        episode.close()
        with backend.episode("after"):
            pass

    def test_invalid_arguments_do_not_dispatch(self):
        for value in ("../x", "", "has space", None, "a" * 101):
            with self.subTest(episode_id=value), self.assertRaises(ValueError):
                self.backend.create(value)
        with self.backend.episode("valid") as episode:
            for value in ([], "echo hello", [""], ["a\0b"], [42]):
                with self.subTest(argv=value), self.assertRaises(ValueError):
                    episode.execute(value)
            for value in (0, -1, True, 61, 1.5):
                with self.subTest(timeout=value), self.assertRaises(ValueError):
                    episode.execute(["true"], timeout_seconds=value)
        self.assertEqual(self.modal.instances[0].exec.call_count, 1)

    def test_invalid_limits(self):
        for kwargs in ({"lifetime_seconds": 86401}, {"max_tool_calls": True},
                       {"max_output_bytes": 0}, {"cpu": float("nan")},
                       {"cpu": True}, {"cpu": 3}, {"memory_mib": 4096}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                SandboxLimits(**kwargs)
        for value in (0, -1, True, 1.5):
            with self.subTest(concurrency=value), self.assertRaises(ValueError):
                self.make_backend(max_concurrency=value)


class ProcessTests(unittest.TestCase):
    def test_actual_modal_stream_wrappers_and_timeout_sentinel(self):
        """Exercise SDK synchronization/iterators offline when Modal is installed."""
        try:
            from modal.container_process import _ContainerProcess
            from modal._utils.async_utils import synchronize_api
            from modal.exception import ExecTimeoutError
        except ImportError:
            self.skipTest("Optional Modal SDK is not installed")
        import asyncio

        class Router:
            def __init__(self, timeout):
                self.timeout = timeout

            async def exec_stdio_read(self, *args):
                await asyncio.sleep(.01)
                yield SimpleNamespace(data=b"hello")
                if self.timeout:
                    raise ExecTimeoutError("simulated worker timeout")

            async def exec_wait(self, *args):
                await asyncio.sleep(.02)
                if self.timeout:
                    raise ExecTimeoutError("simulated worker timeout")
                return SimpleNamespace(WhichOneof=lambda _: "code", code=0)

        async def make_process(timeout):
            return _ContainerProcess("offline-process", "offline-task", None,
                                     Router(timeout), text=False)

        make_sync = synchronize_api(make_process)
        for timeout in (False, True):
            sandbox = SimpleNamespace(exec=lambda *a, **kw: make_sync(timeout))
            with self.subTest(timeout=timeout):
                if timeout:
                    with self.assertRaisesRegex(TimeoutError, "unknown command outcome"):
                        run_process(sandbox, ["true"], 2, 4)
                else:
                    result = run_process(sandbox, ["true"], 2, 4)
                    self.assertEqual((result.stdout, result.stderr, result.exit_code),
                                     ("hell", "hell", 0))
                    self.assertEqual((result.stdout_bytes, result.stderr_bytes), (5, 5))

    def test_both_streams_drain_concurrently_past_retention_limit(self):
        barrier = threading.Barrier(3)
        drained = []
        def stream(name, chunks):
            barrier.wait(timeout=2)
            yield from chunks
            drained.append(name)
        def wait():
            barrier.wait(timeout=2)
            return 7
        proc = SimpleNamespace(stdout=stream("out", [b"abcdef", b"ghi"]),
                               stderr=stream("err", [b"123456", b"7890"]), wait=wait)
        sandbox = SimpleNamespace(exec=Mock(return_value=proc))
        result = run_process(sandbox, ["program", "$(literal)"], 3, 4)
        self.assertEqual((result.stdout, result.stderr), ("abcd", "1234"))
        self.assertEqual((result.stdout_bytes, result.stderr_bytes), (9, 10))
        self.assertTrue(result.stdout_truncated and result.stderr_truncated)
        self.assertCountEqual(drained, ["out", "err"])
        self.assertEqual(result.exit_code, 7)
        sandbox.exec.assert_called_once_with("program", "$(literal)", timeout=3,
                                             workdir="/repo", text=False)

    def test_invalid_exit_status_is_not_success(self):
        for code in (None, True, "0"):
            with self.subTest(code=code), self.assertRaises(RuntimeError):
                run_process(SimpleNamespace(exec=Mock(return_value=process(code=code))), ["x"], 2, 4)

    def test_reader_error_propagates(self):
        def broken():
            yield b"partial"
            raise ConnectionError("lost stream")
        with self.assertRaises(ConnectionError):
            run_process(SimpleNamespace(exec=Mock(return_value=process(stdout=broken()))), ["x"], 2, 4)

    def test_host_deadline_bounds_stalled_dispatch(self):
        release = threading.Event()
        def stalled(*args, **kwargs):
            release.wait(3)
            return process()
        try:
            with self.assertRaises(TimeoutError):
                run_process(SimpleNamespace(exec=stalled), ["x"], 1, 4)
        finally:
            release.set()

    def test_process_input_validation(self):
        sandbox = SimpleNamespace(exec=Mock())
        for argv, timeout, cap in (([], 1, 4), ([""], 1, 4), (["x"], True, 4),
                                   (["x"], 1, 0), (["x"], 1, True)):
            with self.subTest(argv=argv, timeout=timeout, cap=cap), self.assertRaises(ValueError):
                run_process(sandbox, argv, timeout, cap)
        sandbox.exec.assert_not_called()
