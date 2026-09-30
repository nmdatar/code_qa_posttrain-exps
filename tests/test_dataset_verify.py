"""Assertion verification with real bundle preparation and an isolated fake backend."""

import hashlib
import json
from unittest.mock import Mock
import unittest

from dataset_builder.build import write_json, read_jsonl
from dataset_builder.contracts import canonical_hash
from dataset_builder.environment import EnvironmentError
from dataset_builder.verify import verify_bundle
from tests import test_dataset_build as fixtures


class DatasetVerificationTests(unittest.TestCase):
    git = fixtures.DatasetBuildTests.git
    prepare_spec = fixtures.DatasetBuildTests.prepare_spec

    def setUp(self):
        fixtures.DatasetBuildTests.setUp(self)
        self.output, self.manifest = self.prepare_spec()
        self.environment = json.loads((self.output / "public/environment.json").read_text())
        files = {"client.py": hashlib.sha256(self.source.encode()).hexdigest()}
        self.built = {
            "status": "ready", "commit": self.commit, "backend": "docker",
            "image_digest": "sha256:" + "e" * 64,
            "snapshot_files": files,
            "snapshot_sha256": hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest(),
            "source_environment_sha256": canonical_hash(self.environment),
            "capability": "execution", "install_commands": [],
            "readiness_command": ["python", "--version"],
        }
        self.save_built()
        self.backend = Mock()
        self.result = {"exit_code": 0, "stdout": "PRIVATE_EXPECTED_MARKER\n", "stderr": "",
                       "timed_out": False, "truncated": False}
        self.backend.run.return_value = self.result

    def save_built(self):
        write_json(self.output / "environment-build/result.json", self.built)

    def test_matching_output_passes_without_promoting_gold(self):
        before = (self.output / "private/tasks.jsonl").read_bytes()
        report = verify_bundle(self.output, self.backend)
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["assertions"][0]["status"], "passed")
        self.assertEqual(report["gold_status"], "draft")
        self.assertFalse(report["human_reviewed"])
        self.assertFalse(report["training_eligible"])
        self.assertFalse(report["answer_quality_evaluated"])
        self.assertEqual(before, (self.output / "private/tasks.jsonl").read_bytes())
        self.assertEqual(report, json.loads((self.output / "private/verification-report.json").read_text()))
        call = self.backend.run.call_args
        self.assertEqual(call.args[:2], (self.built["image_digest"], ["python", "-"]))
        self.assertEqual(call.kwargs["stdin"], self.spec["tasks"][0]["assertions"][0]["code"])

    def test_wrong_output_or_nonzero_exit_fails(self):
        for delta in ({"stdout": "wrong\n"}, {"exit_code": 1}):
            with self.subTest(delta=delta):
                self.backend.run.return_value = {**self.result, **delta}
                report = verify_bundle(self.output, self.backend)
                self.assertEqual(report["status"], "not_passed")
                self.assertEqual(report["assertions"][0]["status"], "failed")

    def test_timeout_and_truncation_remain_unresolved(self):
        for field in ("timed_out", "truncated"):
            with self.subTest(field=field):
                self.backend.run.return_value = {**self.result, field: True}
                report = verify_bundle(self.output, self.backend)
                self.assertEqual(report["status"], "not_passed")
                self.assertEqual(report["assertions"][0]["status"], "unresolved")

    def test_environment_failure_is_distinct_from_assertion_failure(self):
        self.backend.run.side_effect = EnvironmentError("sandbox launch failed")
        report = verify_bundle(self.output, self.backend)
        self.assertEqual(report["status"], "not_passed")
        assertion = report["assertions"][0]
        self.assertEqual(assertion["status"], "infrastructure_error")
        self.assertIn("sandbox launch failed", assertion["execution"]["error"])

    def test_modified_artifacts_reject_before_any_execution(self):
        for relative in ("public/tasks.jsonl", "private/tasks.jsonl", "private/assertions.jsonl"):
            with self.subTest(relative=relative):
                path = self.output / relative
                previous = path.read_bytes()
                try:
                    path.write_bytes(previous + b"\n")
                    with self.assertRaisesRegex(ValueError, "hash mismatch"):
                        verify_bundle(self.output, self.backend)
                    self.backend.run.assert_not_called()
                finally:
                    path.write_bytes(previous)

    def test_wrong_environment_commit_and_nonready_status_reject(self):
        original = dict(self.built)
        for delta in ({"commit": "0" * 40}, {"status": "quarantined"}):
            with self.subTest(delta=delta):
                self.built = {**original, **delta}
                self.save_built()
                with self.assertRaisesRegex(ValueError, "not ready or wrong commit"):
                    verify_bundle(self.output, self.backend)
                self.backend.run.assert_not_called()


    def rewrite_assertions(self, records):
        """Rehash intentional fixture edits so semantic binding is exercised."""
        path = self.output / "private/assertions.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in records))
        self.manifest["artifacts"]["private/assertions.jsonl"] = hashlib.sha256(path.read_bytes()).hexdigest()
        write_json(self.output / "manifest.json", self.manifest)

    def assert_rejected_before_execution(self, message):
        with self.assertRaisesRegex(ValueError, message):
            verify_bundle(self.output, self.backend)
        self.backend.run.assert_not_called()

    def test_manifest_must_cover_every_required_artifact(self):
        original = dict(self.manifest["artifacts"])
        for relative in original:
            with self.subTest(relative=relative):
                self.manifest["artifacts"] = {k: v for k, v in original.items() if k != relative}
                write_json(self.output / "manifest.json", self.manifest)
                self.assert_rejected_before_execution("Incomplete artifact manifest")

    def test_built_environment_requires_matching_recipe_binding(self):
        for value in (None, "0" * 64):
            with self.subTest(value=value):
                self.built["source_environment_sha256"] = value
                self.save_built()
                self.assert_rejected_before_execution("recipe binding mismatch")

    def test_built_snapshot_file_inventory_and_hashes_must_match(self):
        for value in (None, {}, {"client.py": "0" * 64},
                      {**self.built["snapshot_files"], "extra.py": "a" * 64}):
            with self.subTest(value=value):
                self.built["snapshot_files"] = value
                self.save_built()
                self.assert_rejected_before_execution("snapshot binding mismatch")

    def test_assertion_inventory_matches_taskspec_probes(self):
        original = read_jsonl(self.output / "private/assertions.jsonl")
        for mode in ("missing", "wrong_task", "wrong_assertion", "extra", "duplicate"):
            with self.subTest(mode=mode):
                rows = json.loads(json.dumps(original))
                message = "probe sets differ"
                if mode == "missing":
                    rows[0]["assertions"] = []
                elif mode == "wrong_task":
                    rows[0]["task_id"] = "unknown"
                elif mode == "wrong_assertion":
                    rows[0]["assertions"][0]["id"] = "unknown"
                elif mode == "extra":
                    rows[0]["assertions"].append({**rows[0]["assertions"][0], "id": "extra"})
                else:
                    rows[0]["assertions"].append(dict(rows[0]["assertions"][0]))
                    message = "Duplicate assertion"
                self.rewrite_assertions(rows)
                self.assert_rejected_before_execution(message)

    def test_expected_output_must_match_private_probe_hash(self):
        rows = read_jsonl(self.output / "private/assertions.jsonl")
        rows[0]["assertions"][0]["expected_stdout"] = "different expectation\n"
        self.rewrite_assertions(rows)
        self.assert_rejected_before_execution("expected output differs from TaskSpec")

if __name__ == "__main__":
    unittest.main()
