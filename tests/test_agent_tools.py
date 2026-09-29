import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from agent_harness.artifacts import ArtifactStore
from agent_harness.code_tools import GitRepository, code_understanding_tools
from agent_harness.contracts import ToolContext
from agent_harness.sandbox import ExecutionEnvironment, _source_digest
from dataset_builder.environment import EnvironmentError


class FakeSandbox:
    name = "docker"

    def __init__(self):
        self.calls = []
        self.verification_digest = None

    def run(self, image_id, command, limits, stdin=None):
        self.calls.append((image_id, command, stdin))
        stdout = "ok\n"
        if command[1:4] == ["-I", "-S", "-c"]:
            stdout = self.verification_digest or hashlib.sha256(
                json.dumps(json.loads(stdin), sort_keys=True).encode()).hexdigest()
        return {"exit_code": 0, "stdout": stdout, "stderr": "", "timed_out": False,
                "truncated": False, "duration_seconds": 0.01}


class AgentToolTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.repo_path = self.root / "repository"
        self.repo_path.mkdir()
        self.git("init", "-q")
        self.git("config", "user.email", "test@example.org")
        self.git("config", "user.name", "Test")
        (self.repo_path / "a.py").write_text("class Alpha:\n    def method(self):\n        return 'match'\n# match\n")
        (self.repo_path / "empty.py").write_text("")
        self.git("add", ".")
        self.git("commit", "-qm", "fixture")
        self.commit = self.git("rev-parse", "HEAD").strip()
        self.repo = GitRepository(self.repo_path, self.commit)
        self.artifacts = ArtifactStore(self.root / "artifacts")
        self.artifacts.create_episode("one")
        self.context = ToolContext(resources={"repository": self.repo},
                                   artifacts=self.artifacts, episode_id="one")
        self.tools = {tool.spec.name: tool for tool in code_understanding_tools(include_execution=True)}

    def git(self, *args):
        return subprocess.check_output(["git", "-C", str(self.repo_path), *args], text=True)

    def call(self, name, **args):
        return self.tools[name].execute(args, self.context)

    def environment(self):
        files = self.repo.snapshot_hashes()
        manifest = {"status": "ready", "capability": "execution", "backend": "docker",
                    "commit": self.commit, "image_digest": "sha256:" + "a" * 64, "snapshot_files": files,
                    "snapshot_sha256": hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest(),
                    "readiness": {"exit_code": 0},
                    "runtime_policy": {"host_mounts": False, "network": "none", "read_only": True}}
        backend = FakeSandbox()
        env = ExecutionEnvironment(backend=backend, manifest=manifest)
        self.context.resources["sandbox"] = env
        return env, backend

    def test_reads_pinned_blobs_and_hashes_not_checkout(self):
        original = (self.repo_path / "a.py").read_bytes()
        (self.repo_path / "a.py").write_text("CHANGED")
        (self.repo_path / "secret.txt").write_text("private")
        result = self.call("read_file", path="a.py", start_line=1, line_count=2)
        self.assertIn("Alpha", result.content["text"])
        self.assertEqual(result.evidence[0]["file_sha256"], hashlib.sha256(original).hexdigest())
        self.assertEqual(self.call("read_file", path="secret.txt").status, "error")
        self.assertEqual(self.call("read_file", path="../secret.txt").status, "error")
        self.assertEqual(self.call("read_file", path="empty.py").evidence, [])
        self.assertEqual(self.call("read_file", path="a.py", start_line=99).evidence, [])

    def test_full_commit_and_symlinks_required(self):
        with self.assertRaises(ValueError):
            GitRepository(self.repo_path, "HEAD")
        (self.repo_path / "link.py").symlink_to("a.py")
        self.git("add", "link.py")
        self.git("commit", "-qm", "symlink")
        with self.assertRaisesRegex(ValueError, "symlink"):
            GitRepository(self.repo_path, self.git("rev-parse", "HEAD").strip())

    def test_search_cursor_preserves_matches_inside_same_file(self):
        first = self.call("search_code", query="match", limit=1)
        second = self.call("search_code", query="match", limit=1,
                           offset=first.content["next_offset"], start_line=first.content["next_line"])
        self.assertEqual(first.content["matches"][0]["line"], 3)
        self.assertEqual(second.content["matches"][0]["line"], 4)
        self.assertEqual(self.call("find_symbols", path="a.py").content["symbols"][1]["name"], "method")

    def test_artifacts_are_episode_scoped_and_bounded(self):
        artifact = self.artifacts.put("one", {"text": "x" * 100})
        self.assertEqual(len(self.call("read_artifact", artifact_id=artifact, limit=10).content["text"]), 10)
        self.artifacts.create_episode("two")
        other = ToolContext(resources={}, artifacts=self.artifacts, episode_id="two")
        self.assertEqual(self.tools["read_artifact"].execute({"artifact_id": artifact}, other).status, "error")
        self.assertEqual(self.call("read_artifact", artifact_id="../../secret").status, "error")

    def test_execution_uses_verified_image_and_host_runner(self):
        env, backend = self.environment()
        result = self.call("run_tests", paths=["a.py"])
        self.assertEqual(result.status, "ok")
        self.assertEqual(backend.calls[1][1], [*env.test_runner, "a.py"])
        self.assertEqual(result.content["command"], [*env.test_runner, "a.py"])
        self.call("python_probe", code="print(1)")
        self.assertEqual(backend.calls[3][2], "print(1)")
        self.assertEqual(self.call("run_tests", paths=["--unsafe"]).status, "error")
        self.assertEqual(len(backend.calls), 4)

    def test_changed_runtime_source_stops_before_actual_command(self):
        env, backend = self.environment()
        backend.verification_digest = "0" * 64
        with self.assertRaisesRegex(EnvironmentError, "Runtime image source"):
            self.call("python_probe", code="print(1)")
        self.assertEqual(len(backend.calls), 1)
        self.assertEqual(backend.calls[0][1][1:4], ["-I", "-S", "-c"])

    def test_verifier_hashes_bytes_and_rejects_symlinks(self):
        files = self.repo.snapshot_hashes()
        expected = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
        self.assertEqual(_source_digest(files, self.repo_path), expected)
        (self.repo_path / "a.py").write_text("modified by installer")
        self.assertNotEqual(_source_digest(files, self.repo_path), expected)
        (self.repo_path / "a.py").unlink()
        (self.repo_path / "a.py").symlink_to("empty.py")
        with self.assertRaisesRegex(ValueError, "symlink"):
            _source_digest(files, self.repo_path)
        with self.assertRaisesRegex(ValueError, "Unsafe"):
            _source_digest({"../outside": "hash"}, self.repo_path)

    def test_bad_manifest_never_dispatches_execution(self):
        for field, value in (("commit", "0" * 40), ("backend", "modal"),
                             ("status", "quarantined"), ("snapshot_sha256", "bad"), ("image_digest", "python:latest")):
            env, backend = self.environment()
            env.manifest[field] = value
            with self.assertRaises(EnvironmentError):
                self.call("python_probe", code="print(1)")
            self.assertEqual(backend.calls, [])


if __name__ == "__main__":
    unittest.main()
