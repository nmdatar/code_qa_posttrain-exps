import copy
import json
import tempfile
import unittest
from pathlib import Path
from posttrain.data import adapt_task, MissingRubric, inventory_release, load_tasks, validate_artifacts
from posttrain.calibration import export_packet, import_reviews

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "data/releases/repo-qa-1000-v1"

class DataTests(unittest.TestCase):
    def test_release_inventory(self):
        if not RELEASE.exists(): self.skipTest("Collection artifact unavailable")
        info = inventory_release(RELEASE)
        self.assertEqual((info["tasks"], info["strict_scorable"], info["reference_available"]), (992,12,987))
        self.assertEqual(info["split_counts"], {"development":992})
        self.assertEqual(len(load_tasks(RELEASE, strict=False)),12)
        with self.assertRaises(MissingRubric): load_tasks(RELEASE)
        self.assertEqual(validate_artifacts(RELEASE)["artifacts_checked"],305)
    def test_binding(self):
        if not RELEASE.exists(): self.skipTest("Collection artifact unavailable")
        public = {r["id"]:r for r in map(json.loads,(RELEASE/"public/tasks.jsonl").read_text().splitlines())}
        private = next(r for r in map(json.loads,(RELEASE/"private/grading.jsonl").read_text().splitlines()) if "record" in r)
        item=copy.deepcopy(public[private["task_id"]]); item["repository"]["commit"]="a"*40
        with self.assertRaises(ValueError): adapt_task(item,private)

class CalibrationTests(unittest.TestCase):
    def example(self):
        return {"task_id":"t1","family_id":"f1","predicted_tier":"accepted","question":"<script>bad</script>","answer":"answer","origin":"constructed"}
    def decision(self,packet,reviewer="alice"):
        row=packet["examples"][0]
        return {"example_id":row["example_id"],"example_sha256":row["example_sha256"],"reviewer":reviewer,"reviewer_type":"human","rationale":"Inspected evidence","label":"accepted"}
    def test_independent_reviews_and_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            result=export_packet([self.example()],tmp); p=json.loads(Path(result["packet"]).read_text())
            self.assertNotIn("<script>",Path(result["review_html"]).read_text())
            d={"packet_sha256":p["packet_sha256"],"decisions":[self.decision(p),self.decision(p,"bob")]}
            r=import_reviews(p,d)
            self.assertEqual(r["report"]["reviewed_tasks"],1)
            self.assertEqual(r["report"]["status"],"needs_more_review_or_examples")
            d["decisions"][1]["reviewer"]="alice"
            self.assertEqual(import_reviews(p,d)["report"]["pending_or_disputed"],1)
    def test_incremental_and_idempotent_import(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=json.loads(Path(export_packet([self.example()],tmp)["packet"]).read_text())
            out=Path(tmp)/"imported.json"
            d={"packet_sha256":p["packet_sha256"],"decisions":[self.decision(p)]}
            first=import_reviews(p,d,out)
            self.assertEqual(first["report"]["pending_or_disputed"],1)
            d["decisions"][0]["reviewer"]=" ALICE "
            self.assertEqual(import_reviews(p,d,out),first)
            d["decisions"]=[self.decision(p,"Bob")]
            combined=import_reviews(p,d,out)
            self.assertEqual(combined["report"]["reviewed_tasks"],1)
            self.assertEqual(len(combined["rows"][0]["reviews"]),2)
            d["decisions"]=[self.decision(p,"alice")]
            d["decisions"][0]["label"]="partial"
            with self.assertRaises(ValueError):import_reviews(p,d,out)
            self.assertEqual(json.loads(out.read_text()),combined)
    def test_tamper_and_model_review_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=json.loads(Path(export_packet([self.example()],tmp)["packet"]).read_text())
            d={"packet_sha256":p["packet_sha256"],"decisions":[self.decision(p)]}
            d["decisions"][0]["reviewer_type"]="model"
            with self.assertRaises(ValueError): import_reviews(p,d)
            d["decisions"][0]["reviewer_type"]="human"; p["examples"][0]["answer"]="tampered"
            with self.assertRaises(ValueError): import_reviews(p,d)
    def test_disagreement_stays_pending(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=json.loads(Path(export_packet([self.example()],tmp)["packet"]).read_text())
            d={"packet_sha256":p["packet_sha256"],"decisions":[self.decision(p),self.decision(p,"bob")]}
            d["decisions"][1]["label"]="partial"
            self.assertEqual(import_reviews(p,d)["report"]["pending_or_disputed"],1)

if __name__ == "__main__": unittest.main()
