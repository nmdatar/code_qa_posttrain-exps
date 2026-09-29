import copy
import tempfile
import unittest
from pathlib import Path
from qa_eval.demo import fixture, KEY
from qa_eval.grading import evaluate
from qa_eval.security import seal
from qa_eval.reporting import compare, scorecard, calibration


class ReportingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        t, a, c, m, s = fixture(Path(self.tmp.name))
        self.report, _ = evaluate(t, a, seal("EpisodeMetrics", m, KEY), c, self.tmp.name, KEY,
                                  semantic_envelope=seal("SemanticAssessment", s, KEY))

    def cohort(self, accepted=True, fast=False):
        rows = []
        for i in range(20):
            r = copy.deepcopy(self.report)
            r.update(task_id=str(i), family_id=str(i), task_hash=f"{i:064x}",
                     planned_task_hashes=[f"{j:064x}" for j in range(20)])
            r["metrics"]["episode_id"] = str(i)
            if not accepted:
                r.update(tier="partial", reward=.2)
            if fast:
                r["metrics"]["latency_seconds"] *= .1
                r["compute_units"] *= .1
            rows.append(r)
        return rows

    def test_fast_but_worse_checkpoint_blocked(self):
        result = compare(self.cohort(), self.cohort(accepted=False, fast=True), draws=100)
        self.assertEqual(result["promotion"], "blocked")

    def test_quality_improvement_eligible_but_not_automatic_promotion(self):
        result = compare(self.cohort(accepted=False), self.cohort(), draws=100)
        self.assertTrue(result["quality_gate"])
        self.assertEqual(result["promotion"], "eligible_pending_human_audit")

    def test_equal_quality_faster_eligible(self):
        result = compare(self.cohort(), self.cohort(fast=True), draws=100)
        self.assertTrue(result["efficiency_gate"])

    def test_both_omit_same_task_still_blocked(self):
        rows = self.cohort()[:-1]
        result = compare(rows, rows, draws=100)
        self.assertIn("baseline omits preregistered tasks", result["blockers"])
        self.assertEqual(scorecard(rows)["accepted_answer_rate"], .95)

    def test_mismatched_accounting_blocked(self):
        a, b = self.cohort(), self.cohort(fast=True)
        for r in b:
            r["scoring_contract_hash"] = "a"*64
        result = compare(a, b, draws=100)
        self.assertEqual(result["promotion"], "blocked")

    def test_confidence_bounds_require_sufficient_review(self):
        rows = []
        for i in range(400):
            label = "accepted" if i >= 250 else "material_error"
            rows.append({"task_id": str(i), "family_id": str(i % 20),
                         "predicted_tier": "accepted" if label == "accepted" else "failed",
                         "reviews": [{"reviewer": "one", "label": label}, {"reviewer": "two", "label": label}]})
        self.assertEqual(calibration(rows)["status"], "passed")
        rows[0]["reviews"][1]["label"] = "accepted"
        self.assertNotEqual(calibration(rows)["status"], "passed")

    def test_duplicate_calibration_variants_not_independent(self):
        row = {"task_id": "same", "family_id": "same", "predicted_tier": "failed",
               "reviews": [{"reviewer": "one", "label": "material_error"}, {"reviewer": "two", "label": "material_error"}]}
        self.assertTrue(calibration([row, row])["correlated_variants_in_strata"])


if __name__ == "__main__":
    unittest.main()
