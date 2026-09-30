import json
import unittest
from pathlib import Path
from unittest.mock import patch
from posttrain.calibration_scoring import score_example
from posttrain.storage import digest

ROOT=Path(__file__).resolve().parents[1]
class QualityOnlyTests(unittest.TestCase):
    def inputs(self):
        release=ROOT/"data/releases/repo-qa-development-posttrain-v1"
        if not release.exists():self.skipTest("Diagnostic collection unavailable")
        g=json.loads((release/"private/grading.jsonl").read_text().splitlines()[0]);t=g["record"]
        b=json.loads((release/"private/runtime-bindings.json").read_text())[t["id"]]
        e={"example_id":"e", "rubric_sha256":digest(t),"submission":{"schema_version":"1.0","task_id":t["id"],"text":"example","citations":[],"diagram":None}}
        return e,t,b["source_root"]
    def test_no_rollout_telemetry(self):
        e,t,root=self.inputs()
        with patch("posttrain.calibration_scoring.judge",return_value={}) as j,patch("posttrain.calibration_scoring.decide",return_value=("partial",0.0,False,[])):
            result=score_example(e,t,root,{},lambda *args:None)
            self.assertEqual(j.call_args.args[8],{"execution_records":[],"probes":[]})
            self.assertIsNone(result["metrics"]);self.assertIsNone(result["reward"])
            self.assertEqual(result["predicted_tier"],"partial")
    def test_changed_rubric_rejected(self):
        e,t,root=self.inputs();e["rubric_sha256"]="0"*64
        with self.assertRaises(ValueError):score_example(e,t,root,{},lambda *args:None)
    def test_failed_judge_is_unresolved(self):
        e,t,root=self.inputs()
        with patch("posttrain.calibration_scoring.judge",side_effect=TimeoutError("timeout")):
            result=score_example(e,t,root,{},lambda *args:None)
            self.assertEqual(result["predicted_tier"],"unresolved")
            self.assertIsNone(result["reward"])
