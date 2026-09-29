import copy
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from agent_harness.images import prepare_environment, validate_manifest
from agent_harness.process import CommandResult


class FakeImage:
    object_id = "im-test"

    def __init__(self):
        self.layers = []

    def from_registry(self, base):
        self.layers.append(("base", base))
        return self

    def from_id(self, image_id):
        self.layers.append(("base_id", image_id))
        return self

    def apt_install(self, *args):
        self.layers.append(("apt", args))
        return self

    def workdir(self, path):
        self.layers.append(("workdir", path))
        return self

    def add_local_dir(self, source, destination, *, copy):
        assert copy
        files = {str(path.relative_to(source)): path.read_bytes() for path in source.rglob("*") if path.is_file()}
        self.layers.append(("copy", files))
        return self

    def run_commands(self, *commands):
        self.layers.append(("run", commands))
        return self

    def build(self, app):
        self.layers.append(("build", app))
        return self


class EnvironmentImageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.repo = Path(self.temp.name)
        self.git("init", "-q")
        self.git("config", "user.email", "test@example.test")
        self.git("config", "user.name", "Test")
        (self.repo / "main.py").write_text("print('pinned')\n")
        (self.repo / "requirements.txt").write_text("example==1.0\n")
        self.commit = self.save()
        self.recipe = {"mode": "executable", "base_image": "python@sha256:" + "a" * 64,
                       "dependency_files": ["requirements.txt"],
                       "dependency_commands": ["python -m pip install -r requirements.txt"],
                       "build_commands": ["python -m compileall main.py"],
                       "readiness_commands": [["python", "main.py"]]}
        self.image = FakeImage()
        self.terminated = False
        self.detached = False

        def create(**kwargs):
            self.sandbox_kwargs = kwargs
            return SimpleNamespace(object_id="sb-test", terminate=lambda **kw: setattr(self, "terminated", kw.get("wait") is True),
                                   detach=lambda: setattr(self, "detached", True))

        self.modal = SimpleNamespace(Image=self.image, App=SimpleNamespace(lookup=lambda *a, **kw: "app"),
                                     Sandbox=SimpleNamespace(create=create), __version__="fake")
        self.runner = patch("agent_harness.process.run_process", return_value=CommandResult("ok", "", 0, .01, 2, 0, False, False)).start()
        self.addCleanup(patch.stopall)

    def git(self, *args):
        return subprocess.run(["git", "-C", str(self.repo), *args], check=True, capture_output=True).stdout.decode().strip()

    def save(self):
        self.git("add", ".")
        self.git("commit", "-qm", "fixture")
        return self.git("rev-parse", "HEAD")

    def prepare(self):
        return prepare_environment(self.repo, self.commit, self.recipe, "tests", modal_module=self.modal)

    def test_uses_pinned_snapshot_and_dependency_layer(self):
        (self.repo / "main.py").write_text("modified working tree")
        (self.repo / "private_gold.json").write_text("untracked")
        manifest = self.prepare()
        copies = [value for kind, value in self.image.layers if kind == "copy"]
        self.assertEqual(list(copies[0]), ["requirements.txt"])
        self.assertEqual(copies[1]["main.py"], b"print('pinned')\n")
        self.assertNotIn("private_gold.json", copies[1])
        self.assertFalse(any(".git" in key for key in copies[1]))
        self.assertTrue(self.sandbox_kwargs["block_network"])
        self.assertFalse(self.sandbox_kwargs["include_oidc_identity_token"])
        self.assertEqual(self.sandbox_kwargs["cpu"], (0.5, 2))
        self.assertEqual(self.sandbox_kwargs["memory"], (1024, 2048))
        self.assertTrue(self.terminated and self.detached)
        validate_manifest(manifest)
        self.assertEqual(manifest["environment_id"], self.prepare()["environment_id"])
        changed = copy.deepcopy(manifest)
        changed["recipe"]["build_commands"] = []
        with self.assertRaisesRegex(ValueError, "identity"):
            validate_manifest(changed)
        changed = copy.deepcopy(manifest)
        changed["image_id"] = "im-other"
        with self.assertRaisesRegex(ValueError, "identity"):
            validate_manifest(changed)
        self.image.object_id = "im-other"
        self.assertNotEqual(manifest["environment_id"], self.prepare()["environment_id"])

    def test_cleanup_failure_reports_sandbox_identity(self):
        def fail(**kwargs):
            raise RuntimeError("transport lost")

        self.modal.Sandbox.create = lambda **kw: SimpleNamespace(
            object_id="sb-orphan", terminate=fail,
            detach=lambda: setattr(self, "detached", True))
        with self.assertRaisesRegex(RuntimeError, "sb-orphan"):
            self.prepare()
        self.assertTrue(self.detached)

    def test_concrete_modal_base_image(self):
        del self.recipe["base_image"]
        self.recipe["base_image_id"] = "im-ConcreteBase123"
        manifest = self.prepare()
        self.assertEqual(self.image.layers[0], ("base_id", "im-ConcreteBase123"))
        self.assertEqual(manifest["recipe"]["base_image_id"], "im-ConcreteBase123")
        validate_manifest(manifest)

    def test_requires_exactly_one_pinned_base(self):
        self.recipe["base_image_id"] = "im-ConcreteBase123"
        with self.assertRaisesRegex(ValueError, "exactly one"):
            self.prepare()
        del self.recipe["base_image"]
        self.recipe["base_image_id"] = "named-latest"
        with self.assertRaisesRegex(ValueError, "concrete Modal"):
            self.prepare()
        del self.recipe["base_image_id"]
        with self.assertRaisesRegex(ValueError, "exactly one"):
            self.prepare()
        self.assertEqual(self.image.layers, [])

    def test_failed_readiness_and_transport_errors_cleanup(self):
        self.runner.return_value = CommandResult("", "failed", 1, .01, 0, 6, False, False)
        with self.assertRaisesRegex(RuntimeError, "readiness failed"):
            self.prepare()
        self.assertTrue(self.terminated and self.detached)
        self.terminated = self.detached = False
        self.runner.side_effect = TimeoutError("remote timeout")
        with self.assertRaises(TimeoutError):
            self.prepare()
        self.assertTrue(self.terminated and self.detached)

    def test_setup_precedes_readiness(self):
        self.recipe["setup_commands"] = [["python", "-c", "print('setup')"]]
        manifest = self.prepare()
        self.assertEqual([call.args[1] for call in self.runner.call_args_list], self.recipe["setup_commands"] + self.recipe["readiness_commands"])
        self.assertEqual(len(manifest["readiness"]), 1)

    def test_rejects_moving_refs_base_tags_and_unsafe_dependencies(self):
        self.recipe["base_image"] = "python:latest"
        with self.assertRaises(ValueError):
            self.prepare()
        self.recipe["base_image"] = "python@sha256:" + "a" * 64
        self.recipe["dependency_files"] = ["../private.json"]
        with self.assertRaises(ValueError):
            self.prepare()
        self.recipe["dependency_files"] = []
        self.commit = "HEAD"
        with self.assertRaises(ValueError):
            self.prepare()
        self.assertEqual(self.image.layers, [])

    def test_rejects_escaping_symlinks_and_submodules(self):
        (self.repo / "escape").symlink_to("../../private")
        self.commit = self.save()
        with self.assertRaisesRegex(ValueError, "symlink"):
            self.prepare()
        self.git("rm", "escape")
        self.git("update-index", "--add", "--cacheinfo", f"160000,{self.commit},vendor")
        self.git("commit", "-qm", "submodule")
        self.commit = self.git("rev-parse", "HEAD")
        with self.assertRaisesRegex(ValueError, "Submodules"):
            self.prepare()
        self.assertEqual(self.image.layers, [])


if __name__ == "__main__":
    unittest.main()
