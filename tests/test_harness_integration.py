import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from qa_eval.demo import fixture, KEY
from qa_eval.harness import EpisodeRecorder
from qa_eval.security import seal, unseal
from qa_eval.dataset import seed_calibration


class HarnessIntegrationTests(unittest.TestCase):
    def test_signed_cli_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            task, answer, config, metrics, semantic = fixture(base / "repo")
            key = base / "key"
            key.write_bytes(KEY)
            key.chmod(0o600)
            files = {"task": task, "submission": answer, "experiment": config,
                     "metrics": seal("EpisodeMetrics", metrics, KEY),
                     "semantic": seal("SemanticAssessment", semantic, KEY)}
            cmd = [sys.executable, "-m", "qa_eval", "grade", "--repo", str(base / "repo"),
                   "--key", str(key), "--out", str(base / "report.json")]
            for name, value in files.items():
                p = base / (name + ".json")
                p.write_text(json.dumps(value))
                cmd += ["--" + name, str(p)]
            result = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = unseal(json.loads((base / "report.json").read_text()), "GradeReport", KEY)
            self.assertEqual(report["tier"], "accepted")

    def test_recorder_counts_retries_and_tools(self):
        with tempfile.TemporaryDirectory() as tmp:
            task, answer, config, _, _ = fixture(tmp)
            recorder = EpisodeRecorder(task, config["id"], "episode", "trusted://trajectory")
            recorder.usage(10, 5, .01)
            recorder.retry()
            recorder.usage(20, 10, .02)
            recorder.token_received()
            self.assertEqual(recorder.tool("read_file", lambda: "data"), "data")
            payload = unseal(recorder.finish(answer, KEY), "EpisodeMetrics", KEY)
            self.assertEqual(payload["input_tokens"], 30)
            self.assertEqual(payload["retries"], 1)
            self.assertEqual(payload["tool_calls"], ["read_file"])
            with self.assertRaises(ValueError):
                recorder.finish(answer, KEY)

    def test_100_seeded_tasks_are_never_falsely_reviewed(self):
        with tempfile.TemporaryDirectory() as tmp:
            repos = []
            for i in range(10):
                root = Path(tmp) / str(i)
                fixture(root)
                (root / "more.py").write_text("\n".join(f"def f{n}(x):\n    return x + {n}\n" for n in range(10)))
                subprocess.run(["git", "-C", str(root), "add", "more.py"], check=True, capture_output=True)
                subprocess.run(["git", "-C", str(root), "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
                                "-c", "commit.gpgsign=false", "commit", "-qm", "Additional fixture symbols"], check=True, capture_output=True)
                repos.append({"path": str(root), "url": f"fixture://{i}", "family_id": str(i)})
            tasks = seed_calibration(repos)
            self.assertEqual(len(tasks), 100)
            self.assertEqual(len({t["repository"]["family_id"] for t in tasks}), 10)
            self.assertTrue(all(not t["human_reviewed"] and t["gold_status"] == "draft" for t in tasks))
            self.assertTrue(any(t["diagram"]["required"] for t in tasks))
            self.assertTrue(any(t["answerability"] == "unanswerable" for t in tasks))


if __name__ == "__main__":
    unittest.main()
