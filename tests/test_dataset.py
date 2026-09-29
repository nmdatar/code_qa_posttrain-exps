import unittest
from scripts.prepare_dataset import load_records, split_records, task_id


class DatasetContractTests(unittest.TestCase):
    def test_corrupted_download_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            load_records(b'{"question": "corrupted"}')

    def test_agent_input_excludes_reference_and_annotation(self):
        row = dict(repo="owner/repo", commit_id="a" * 40, question="Where?",
                   answer="HIDDEN GOLD", cluster={"name": "HIDDEN CLUSTER"}, qa_type={})
        tasks, references = split_records([row])
        self.assertEqual(set(tasks[0]), {"id", "repo", "commit_id", "question"})
        self.assertEqual(references[0]["reference_answer"], row["answer"])
        self.assertEqual(tasks[0]["id"], references[0]["id"])

    def test_task_identity_tracks_snapshot_not_reference_edits(self):
        row = dict(repo="owner/repo", commit_id="a" * 40, question="Where?", answer="x")
        self.assertEqual(task_id(row), task_id(dict(row, answer="revised")))
        self.assertNotEqual(task_id(row), task_id(dict(row, commit_id="b" * 40)))


if __name__ == "__main__":
    unittest.main()
