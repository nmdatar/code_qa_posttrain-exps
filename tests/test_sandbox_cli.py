import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from agent_harness.cli import main, save_new, smoke
from agent_harness.modal_backend import ModalSandboxBackend
from tests.test_modal_backend import FakeModal, manifest


class SandboxCLITests(unittest.TestCase):
    def test_atomic_manifest_publication_never_overwrites(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "environment.json"
            save_new(target, {"value": 1})
            with self.assertRaises(FileExistsError):
                save_new(target, {"value": 2})
            self.assertEqual(json.loads(target.read_text()), {"value": 1})
            self.assertEqual(list(Path(tmp).iterdir()), [target])

    def test_validate_never_imports_modal_or_allocates(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "environment.json"
            save_new(target, manifest())
            with patch("agent_harness.modal_backend.importlib.import_module") as sdk, contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(["validate", "--manifest", str(target)]), 0)
                sdk.assert_not_called()

    def test_smoke_closes_two_separate_instances(self):
        with tempfile.TemporaryDirectory() as tmp:
            modal = FakeModal()
            backend = ModalSandboxBackend(manifest(), tmp, modal_module=modal)
            report = smoke(backend, "smoke-test")
            self.assertEqual(report["status"], "passed")
            self.assertNotEqual(report["attempts"][0]["sandbox_id"], report["attempts"][1]["sandbox_id"])
            for sandbox in modal.instances:
                sandbox.terminate.assert_called_once_with(wait=True)


if __name__ == "__main__":
    unittest.main()
