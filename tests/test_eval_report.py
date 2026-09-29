import tempfile
import unittest
from html.parser import HTMLParser
from pathlib import Path

from eval_pipeline.report import write_report


class Tags(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tags = []

    def handle_starttag(self, tag, attrs):
        self.tags.append((tag, dict(attrs)))


class EvalReportTests(unittest.TestCase):
    def render(self, config=None, rows=None, summary=None):
        with tempfile.TemporaryDirectory() as directory:
            path = write_report(Path(directory), config or {}, rows or [], summary or {})
            return path.read_text()

    def test_external_text_is_escaped_in_all_surfaces(self):
        attack = '</pre><script>alert("XSS")</script><img src=x onerror=alert(1)>'
        row = {key: attack for key in ("id", "repo", "question", "reference_answer", "answer", "error", "judge_reason", "status")}
        content = self.render({"model": attack, "judge_model": attack, "dataset_revision": attack}, [row])
        parser = Tags()
        parser.feed(content)
        self.assertEqual(sum(tag == "script" for tag, _ in parser.tags), 1)
        self.assertFalse(any(tag == "img" for tag, _ in parser.tags))
        self.assertIn('&lt;script&gt;alert(&quot;XSS&quot;)&lt;/script&gt;', content)
        self.assertNotIn(attack, content)

    def test_demo_is_unambiguously_labeled(self):
        content = self.render({"mode": "demo"})
        self.assertIn("DEMO · Synthetic pipeline check", content)
        self.assertIn("do not measure model quality", content)
        self.assertNotIn("DEMO · Synthetic pipeline check", self.render({"mode": "import"}))

    def test_failures_are_visible_and_excluded_from_dimension_means(self):
        scores = dict(correctness=8, completeness=8, relevance=8, clarity=8, reasoning=8)
        content = self.render(rows=[
            dict(status="success", scores=scores, question="Good question"),
            dict(status="judge_error", scores=dict.fromkeys(scores, 1), error="Judge unavailable"),
        ])
        self.assertIn('data-score="40"', content)
        self.assertIn('data-status="judge_error" data-score="-1"', content)
        self.assertIn("Judge unavailable", content)
        self.assertIn('value="8.0"', content)
        self.assertIn("Means include only fully scored questions", content)

    def test_missing_values_and_empty_runs_are_renderable(self):
        content = self.render()
        self.assertIn("0 questions", content)
        self.assertIn("No questions match", content)
        self.assertIn("—", content)
        self.assertNotIn("nan", content)

    def test_partial_scores_are_unscored(self):
        content = self.render(rows=[dict(status="success", scores={"correctness": 9})])
        self.assertIn('data-score="-1"', content)
        self.assertIn(">Unscored<", content)


if __name__ == "__main__":
    unittest.main()
