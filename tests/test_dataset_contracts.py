import copy
import unittest

from dataset_builder.contracts import canonical_hash, validate_bundle, validate_public_task


def records():
    private = {
        "schema_version": "1.0", "id": "example", "question": "How is a timeout applied?",
        "repository": {"family_id": "example", "commit": "a" * 40,
                       "url": "https://github.com/example/example"},
        "lineage_id": "timeout", "split": "development", "category": "behavior",
        "answerability": "answerable",
        "claims": [{"id": "timeout", "text": "A timeout is passed to the client.",
                    "weight": 3, "separable_subparts": [], "evidence": [
                        {"path": "src/client.py", "start_line": 1, "end_line": 10,
                         "file_sha256": "b" * 64}]}],
        "critical_errors": [], "permitted_tools": ["read_file", "search"],
        "budgets": {"latency_seconds": 120, "compute_units": 100000,
                    "max_tool_calls": 30, "max_output_tokens": 8000,
                    "max_submission_bytes": 64000},
        "diagram": {"required": False, "criteria": [], "allowed_abstractions": []},
        "probes": [], "human_reviewed": False, "gold_status": "draft",
    }
    public = {key: copy.deepcopy(private[key]) for key in
              ("schema_version", "id", "repository", "split", "permitted_tools", "budgets")}
    public.update(system_prompt="Investigate using repository evidence.",
                  user_prompt=private["question"], environment_id="example-env")
    return public, private


class GeneratedDatasetContractTests(unittest.TestCase):
    def test_valid_draft_is_not_promoted_by_validation(self):
        public, private = records()
        validate_bundle([public], [private])
        self.assertFalse(private["human_reviewed"])
        self.assertEqual(private["gold_status"], "draft")

    def test_gold_fields_never_belong_in_solver_input(self):
        for field in ("claims", "assertions", "reference_answer", "human_reviewed"):
            public, _ = records()
            public[field] = "private"
            with self.assertRaises(ValueError):
                validate_public_task(public)

    def test_only_full_commit_hashes(self):
        for length in (7, 39, 41, 63, 65):
            public, _ = records()
            public["repository"]["commit"] = "a" * length
            with self.assertRaises(ValueError):
                validate_public_task(public)
        for length in (40, 64):
            public, _ = records()
            public["repository"]["commit"] = "a" * length
            validate_public_task(public)

    def test_evidence_paths_are_confined(self):
        for path in ("../secret", "/tmp/secret", "src/../../secret", "C:/secret",
                     "src\\..\\secret", "./client.py", "src//client.py"):
            public, private = records()
            private["claims"][0]["evidence"][0]["path"] = path
            with self.assertRaisesRegex(ValueError, "relative path"):
                validate_bundle([public], [private])

    def test_family_and_lineage_must_not_cross_splits(self):
        for keep_family in (True, False):
            first_public, first_private = records()
            second_public, second_private = records()
            for row in (second_public, second_private):
                row["id"] = "second"
                row["split"] = "train"
                if not keep_family:
                    row["repository"]["family_id"] = "other"
            with self.assertRaisesRegex(ValueError, "leaks across splits"):
                validate_bundle([first_public, second_public], [first_private, second_private])

    def test_public_private_bindings(self):
        for field, value in (("user_prompt", "Different question"), ("split", "train")):
            public, private = records()
            public[field] = value
            with self.assertRaises(ValueError):
                validate_bundle([public], [private])
        public, private = records()
        with self.assertRaises(ValueError):
            validate_bundle([public], [])
        with self.assertRaises(ValueError):
            validate_bundle([public, public], [private])

    def test_machine_verification_is_not_human_acceptance(self):
        public, private = records()
        private["gold_status"] = "accepted"
        with self.assertRaisesRegex(ValueError, "human review"):
            validate_bundle([public], [private])
        private["human_reviewed"] = True
        validate_bundle([public], [private])

    def test_environment_binding(self):
        public, private = records()
        environment = {"id": public["environment_id"], "repository": public["repository"]}
        validate_bundle([public], [private], [environment])
        with self.assertRaisesRegex(ValueError, "unknown environment"):
            validate_bundle([public], [private], [])
        with self.assertRaisesRegex(ValueError, "repositories must match"):
            validate_bundle([public], [private], [{**environment, "repository": {}}])

    def test_canonical_hash_ignores_object_order_but_binds_content(self):
        self.assertEqual(canonical_hash({"a": 1, "b": 2}), canonical_hash({"b": 2, "a": 1}))
        self.assertNotEqual(canonical_hash({"a": 1}), canonical_hash({"a": 2}))


if __name__ == "__main__":
    unittest.main()
