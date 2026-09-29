import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch, MagicMock

from dataset_builder.environment import (
    DockerBackend, ModalBackend, EnvironmentError, EnvironmentRecipe, RunLimits,
    _capture, build_environment,
)


class CaptureTests(unittest.TestCase):
    def test_combined_output_is_bounded_without_pipe_deadlock(self):
        result = _capture([sys.executable, "-c",
            "import sys; sys.stdout.write('x'*200000); sys.stderr.write('y'*200000)"],
            timeout_seconds=5, output_bytes=1024)
        self.assertEqual(result["exit_code"], 0)
        self.assertTrue(result["truncated"])
        self.assertEqual(len(result["stdout"]) + len(result["stderr"]), 1024)

    def test_timeout_and_stdin(self):
        result = _capture([sys.executable, "-c", "import time; time.sleep(20)"],
                          timeout_seconds=.1, output_bytes=1024)
        self.assertTrue(result["timed_out"])
        result = _capture([sys.executable, "-c", "print(input())"],
                          timeout_seconds=5, output_bytes=1024, stdin="private probe\n")
        self.assertEqual(result["stdout"], "private probe\n")


class BackendTests(unittest.TestCase):
    def test_docker_policy_and_freshness_and_cleanup(self):
        backend = DockerBackend()
        with patch("dataset_builder.environment._capture", return_value={"exit_code": 0}) as capture:
            backend.run("sha256:example", ["python", "-"], stdin="assert True")
            backend.run("sha256:example", ["python", "-"], stdin="assert True")
        calls = capture.call_args_list
        first = calls[0].args[0]
        second = calls[2].args[0]
        for flag, value in [("--network", "none"), ("--cap-drop", "ALL"),
                            ("--security-opt", "no-new-privileges"), ("--user", "65534:65534")]:
            self.assertEqual(first[first.index(flag) + 1], value)
        self.assertIn("--read-only", first)
        self.assertNotIn("--volume", first)
        self.assertNotEqual(first[first.index("--name") + 1], second[second.index("--name") + 1])
        self.assertEqual(calls[1].args[0][1:3], ["rm", "--force"])
        self.assertEqual(calls[0].kwargs["stdin"], "assert True")

    def test_cleanup_on_client_failure(self):
        backend = DockerBackend()
        with patch("dataset_builder.environment._capture", side_effect=[EnvironmentError("bad"), {}]) as call:
            with self.assertRaises(EnvironmentError):
                backend.run("image", ["python", "-"])
        self.assertEqual(call.call_args_list[-1].args[0][1:3], ["rm", "--force"])

    def test_validation_and_broken_daemon(self):
        with self.assertRaises(ValueError):
            RunLimits(timeout_seconds=0)
        with self.assertRaises(ValueError):
            EnvironmentRecipe(readiness_command=[])
        with patch.object(DockerBackend, "_command", return_value={
            "exit_code": 0, "timed_out": False, "stdout": '""',
            "stderr": "Cannot connect to Docker daemon"}):
            self.assertFalse(DockerBackend().available())

    def test_modal_policy_stdin_and_termination(self):
        modal = MagicMock()
        sandbox = modal.Sandbox.create.return_value
        sandbox.returncode = 0
        sandbox.stdout.read.return_value = json.dumps({"exit_code": 0, "stdout": "ok", "stderr": ""})
        sandbox.object_id = "sb-example"
        with patch("dataset_builder.environment._modal_module", return_value=modal):
            result = ModalBackend().run("im-example", ["python", "-"], stdin="private")
        policy = modal.Sandbox.create.call_args.kwargs
        self.assertTrue(policy["block_network"])
        self.assertEqual(policy["volumes"], {})
        self.assertEqual(policy["secrets"], [])
        self.assertFalse(policy["include_oidc_identity_token"])
        self.assertEqual(policy["memory"], (512, 512))
        self.assertEqual(result["sandbox_id"], "sb-example")
        self.assertEqual(json.loads(sandbox.stdin.write.call_args.args[0])["stdin"], "private")
        sandbox.terminate.assert_called_once()


class ImageBuildTests(unittest.TestCase):
    def test_build_uses_committed_blobs_and_excludes_untracked_gold(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repo = root / "repo"
            repo.mkdir()
            def git(*args):
                return subprocess.check_output(["git", "-C", str(repo), *args]).decode().strip()
            git("init", "-q")
            (repo / "code.py").write_text("print('pinned')\n")
            git("add", "code.py")
            git("-c", "user.name=Fixture", "-c", "user.email=fixture@example.test", "commit", "-qm", "fixture")
            commit = git("rev-parse", "HEAD")
            (repo / "code.py").write_text("print('dirty')\n")
            (repo / "private-gold.json").write_text('{"answer": "secret"}')
            backend = MagicMock(spec=DockerBackend)
            backend.available.return_value = True
            def command(args, timeout=30):
                result = {"exit_code": 0, "timed_out": False, "stdout": "", "stderr": ""}
                if args[0] == "image":
                    result["stdout"] = '["python@sha256:pinned"]'
                if args[0] == "build":
                    context = Path(args[-1])
                    self.assertEqual((context / "repo/code.py").read_text(), "print('pinned')\n")
                    self.assertFalse((context / "repo/private-gold.json").exists())
                    Path(args[2]).write_text("sha256:built")
                return result
            backend._command.side_effect = command
            backend.run.return_value = {"exit_code": 0, "timed_out": False}
            result = build_environment(repo, commit, EnvironmentRecipe(), root / "output", backend)
            self.assertEqual(result["status"], "ready")
            self.assertEqual(list(result["snapshot_files"]), ["code.py"])
            self.assertEqual(result["base_image_digest"], "python@sha256:pinned")


if __name__ == "__main__":
    unittest.main()
