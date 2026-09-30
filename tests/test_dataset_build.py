"""Exercise preparation against real local Git state, without network or Docker."""

import copy
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from dataset_builder.build import prepare, read_jsonl
from dataset_builder.contracts import canonical_hash, validate_bundle


class DatasetBuildTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.repo = self.base / "repository"
        self.repo.mkdir()
        self.git("init", "--quiet")
        self.git("config", "user.name", "Dataset test")
        self.git("config", "user.email", "dataset-test@example.invalid")
        self.source = "def timeout():\n    return 30\n"
        (self.repo / "client.py").write_text(self.source)
        self.git("add", "client.py")
        self.git("commit", "--quiet", "-m", "Local fixture")
        self.commit = self.git("rev-parse", "HEAD")
        self.spec = {
            "repository": {"url": "https://github.com/example/repository",
                           "commit": self.commit, "family_id": "example/repository"},
            "split": "development",
            "environment": {"base_image": "python:3.12-slim", "capability": "execution",
                            "install_commands": [], "readiness_command": ["python", "--version"]},
            "tasks": [{"id": "timeout", "user_prompt": "What is the default timeout?",
                       "category": "behavior", "provenance": {"source": "PRIVATE_SOURCE_MARKER"},
                       "claims": [{"id": "default", "text": "PRIVATE_GOLD_MARKER: 30 seconds.",
                                   "evidence": [{"path": "client.py", "start_line": 1,
                                                 "end_line": 2, "symbol": "timeout"}]}],
                       "assertions": [{"id": "timeout-value", "code": "print('PRIVATE_PROBE_MARKER')",
                                       "expected_stdout": "PRIVATE_EXPECTED_MARKER\n"}]}],
        }
        self.checkout = patch("dataset_builder.build.checkout", return_value={"path": str(self.repo)}).start()
        self.addCleanup(patch.stopall)

    def git(self, *args):
        return subprocess.run(["git", "-C", str(self.repo), *args], check=True,
                              capture_output=True, text=True).stdout.strip()

    def prepare_spec(self, spec=None, name="output"):
        path = self.base / (name + "-spec.json")
        path.write_text(json.dumps(self.spec if spec is None else spec))
        output = self.base / name
        return output, prepare(path, output, self.base / "repos")

    def test_full_preparation_binds_snapshot_and_keeps_gold_private(self):
        output, manifest = self.prepare_spec()
        public = read_jsonl(output / "public/tasks.jsonl")
        private = read_jsonl(output / "private/tasks.jsonl")
        environment = json.loads((output / "public/environment.json").read_text())
        validate_bundle(public, private, [environment])
        evidence = private[0]["claims"][0]["evidence"][0]
        self.assertEqual(evidence["file_sha256"], hashlib.sha256(self.source.encode()).hexdigest())
        self.assertEqual((evidence["start_line"], evidence["end_line"]), (1, 2))
        self.assertEqual(public[0]["repository"]["commit"], self.commit)
        self.assertEqual(public[0]["user_prompt"], self.spec["tasks"][0]["user_prompt"])
        self.assertTrue(public[0]["system_prompt"])
        for path in (output / "public").rglob("*"):
            if path.is_file():
                self.assertNotIn("PRIVATE_", path.read_text())
        self.assertFalse(manifest["training_eligible"])
        self.assertFalse(private[0]["human_reviewed"])
        self.assertEqual(private[0]["gold_status"], "draft")
        assertion = self.spec["tasks"][0]["assertions"][0]
        self.assertEqual(private[0]["probes"][0]["expected_stdout_sha256"],
                         hashlib.sha256(assertion["expected_stdout"].encode()).hexdigest())
        for relative, checksum in manifest["artifacts"].items():
            self.assertEqual(hashlib.sha256((output / relative).read_bytes()).hexdigest(), checksum)
        self.checkout.assert_called_once_with(
            {"repo": "example/repository", "commit_id": self.commit}, self.base / "repos")

    def test_configured_budgets_are_shared_and_invalid_values_rejected(self):
        self.spec["budgets"] = {"latency_seconds": 1200, "compute_units": 100000,
                                "max_tool_calls": 40, "max_output_tokens": 6000,
                                "max_submission_bytes": 64000}
        output, _ = self.prepare_spec()
        self.assertEqual(read_jsonl(output / "public/tasks.jsonl")[0]["budgets"], self.spec["budgets"])
        self.assertEqual(read_jsonl(output / "private/tasks.jsonl")[0]["budgets"], self.spec["budgets"])
        self.spec["budgets"]["latency_seconds"] = -1
        with self.assertRaises(ValueError):
            self.prepare_spec(name="bad-budget")

    def test_repeat_preparation_has_deterministic_content_hashes(self):
        _, first = self.prepare_spec(name="first")
        _, second = self.prepare_spec(name="second")
        self.assertEqual(first, second)
        self.assertEqual(first["source_spec_sha256"], canonical_hash(self.spec))
        changed = copy.deepcopy(self.spec)
        changed["tasks"][0]["user_prompt"] += " Explain the units."
        _, third = self.prepare_spec(changed, "third")
        self.assertNotEqual(first["artifacts"]["public/tasks.jsonl"],
                            third["artifacts"]["public/tasks.jsonl"])

    def test_nonempty_output_is_preserved_and_rejected_before_checkout(self):
        output = self.base / "output"
        output.mkdir()
        (output / "keep.txt").write_text("existing work")
        with self.assertRaisesRegex(ValueError, "not empty"):
            self.prepare_spec()
        self.assertEqual((output / "keep.txt").read_text(), "existing work")
        self.checkout.assert_not_called()

    def test_training_requires_overlap_audit_but_never_implies_acceptance(self):
        self.spec["split"] = "train"
        with self.assertRaisesRegex(ValueError, "overlap audit"):
            self.prepare_spec()
        self.checkout.assert_not_called()
        self.spec["overlap_audit"] = {"status": "passed"}
        _, manifest = self.prepare_spec()
        self.assertFalse(manifest["training_eligible"])

    def test_invalid_evidence_ranges_fail_without_public_artifacts(self):
        for start, end in ((0, 1), (2, 1), (1, 3)):
            with self.subTest(start=start, end=end):
                self.spec["tasks"][0]["claims"][0]["evidence"][0].update(start_line=start, end_line=end)
                with self.assertRaisesRegex(ValueError, "line range"):
                    self.prepare_spec()
                self.assertFalse((self.base / "output/public/tasks.jsonl").exists())

    def test_path_escape_and_symlink_evidence_are_rejected(self):
        evidence = self.spec["tasks"][0]["claims"][0]["evidence"][0]
        for path in ("../outside.py", "/tmp/outside.py"):
            evidence["path"] = path
            with self.assertRaisesRegex(ValueError, "Unsafe evidence"):
                self.prepare_spec()
        (self.repo / "alias.py").symlink_to("client.py")
        self.git("add", "alias.py")
        self.git("commit", "--quiet", "-m", "Symlink fixture")
        self.spec["repository"]["commit"] = self.git("rev-parse", "HEAD")
        evidence["path"] = "alias.py"
        with self.assertRaisesRegex(ValueError, "symlink"):
            self.prepare_spec()

    def test_dirty_checkout_cannot_become_a_pinned_environment(self):
        (self.repo / "client.py").write_text(self.source + "# dirty\n")
        with self.assertRaisesRegex(ValueError, "not pristine"):
            self.prepare_spec()

    def test_checkout_must_match_declared_commit(self):
        self.spec["repository"]["commit"] = "0" * 40
        with self.assertRaisesRegex(ValueError, "differs"):
            self.prepare_spec()


if __name__ == "__main__":
    unittest.main()
