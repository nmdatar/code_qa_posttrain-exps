"""Synthetic contract demonstration. No real judge/model/human claims are made."""

import copy
from pathlib import Path
import subprocess
import tempfile
from .security import file_hash, digest, bindings, seal
from .deterministic import snapshot, read_evidence
from .dataset import freeze
from .grading import evaluate
from .reporting import scorecard, calibration

KEY = b"synthetic-test-key-never-use-in-production-0000"


def fixture(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    (root / "executor.py").write_text('''def cancel(state):
    state["cancelled"] = True

def run_job(state, job):
    if state.get("cancelled"):
        return
    try:
        return job()
    finally:
        state["cleaned"] = True
''')
    def git(*args):
        return subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True, text=True).stdout.strip()
    git("init", "-q")
    git("add", "executor.py")
    git("-c", "user.name=QA fixture", "-c", "user.email=fixture@example.invalid", "-c", "commit.gpgsign=false", "commit", "-qm", "Synthetic evaluator fixture")
    commit = git("rev-parse", "HEAD")
    ref = {"path": "executor.py", "start_line": 1, "end_line": 2, "file_sha256": file_hash(root / "executor.py"), "symbol": "cancel"}
    task = {"schema_version": "1.0", "id": "fixture-cancel", "question": "What does cancel do?",
            "repository": {"family_id": "synthetic-executor", "commit": commit, "url": "fixture://executor"},
            "lineage_id": "synthetic-cancel", "split": "final_test", "category": "lookup",
            "answerability": "answerable", "claims": [{"id": "c1", "text": "Sets cancelled to True.", "weight": 3,
                                                         "separable_subparts": [], "evidence": [ref]}],
            "critical_errors": ["Claims cancellation kills a running job."], "permitted_tools": ["read_file", "search"],
            "budgets": {"latency_seconds": 100, "compute_units": 10000, "max_tool_calls": 10,
                        "max_output_tokens": 1000, "max_submission_bytes": 20000},
            "diagram": {"required": False, "criteria": [], "allowed_abstractions": []}, "probes": [],
            # True only to exercise the accepted branch in this explicitly
            # synthetic fixture. Never label these as human-calibration data.
            "human_reviewed": True, "gold_status": "accepted"}
    submission = {"schema_version": "1.0", "task_id": task["id"], "text": "It sets cancelled to True. [src]",
                  "citations": [{"id": "src", **ref}], "diagram": None}
    config = {"schema_version": "1.0", "id": "synthetic-demo", "reward_version": "1.0", "frozen": False,
              "latency_enabled": True, "environment_id": "synthetic-environment", "accounting": {"input_token": 1, "output_token": 2, "tool_second": 100},
              "training_judge_family": "fixture-training", "evaluation_judge_family": "fixture-evaluation",
              "judge_version": "fixture-1", "judge_command": ["false"], "judge_timeout_seconds": 10,
              "max_evidence_bytes": 100000, "candidate_preselected": True,
              "baseline_manifest_hash": "0"*64, "budgets_manifest_hash": "0"*64, "task_manifest": []}
    config = freeze(config, [task], {"synthetic": True})
    metrics = {"schema_version": "1.0", **bindings(task, submission), "experiment_id": config["id"], "episode_id": "fixture-1",
               "trajectory_ref": "fixture://trajectory", "latency_seconds": 20, "time_to_first_token_seconds": 1,
               "input_tokens": 100, "output_tokens": 50, "tool_seconds": 1, "tool_calls": ["read_file"],
               "retries": 0, "cost": .01, "termination_reason": "completed", "integrity_violation": False,
               "execution_records": [], "probes": []}
    evidence_key, _ = read_evidence(root, ref)
    finding = {"id": "c1", "coverage": "complete", "verdict": "supported", "material_error": False,
               "reason": "Fixture assertion supported by assignment.", "evidence_keys": [evidence_key]}
    semantic = {"schema_version": "1.0", **bindings(task, submission), "experiment_hash": digest(config),
                "judge_family": config["evaluation_judge_family"], "judge_version": config["judge_version"],
                "prompt_version": "1.0", "snapshot_fingerprint": snapshot(root, commit),
                "required_claims": [finding], "additional_claims": [{**finding, "id": "a1"}],
                "extracted_claims": [{"id": "a1", "text": "Sets cancelled to True.", "source": "text"}],
                "citation_links": [{"claim_id": "a1", "citation_id": "src", "supported": True, "reason": "Assignment matches."}],
                "uncited_claim_ids": [], "critical_error": False, "false_execution_claim": False,
                "answer_mode": "answer", "answerability_satisfied": True,
                "diagram": {"assessed": False, "supported": True, "complete": True, "consistent": True,
                            "readable": True, "material_error": False, "findings": []},
                "assessment_complete": True, "needs_review": False, "evidence_keys": [evidence_key], "disagreements": []}
    return task, submission, config, metrics, semantic


def run():
    with tempfile.TemporaryDirectory(prefix="qa-eval-demo-") as root:
        task, submission, config, metrics, semantic = fixture(root)
        reports = []
        for name in ("correct_slow", "correct_fast", "wrong_fast", "unsupported", "uncertain"):
            m, s = copy.deepcopy(metrics), copy.deepcopy(semantic)
            m["episode_id"] = name
            if name == "correct_slow":
                m["latency_seconds"] = 10000
            elif name in {"correct_fast", "wrong_fast"}:
                m["latency_seconds"] = 0
            if name == "wrong_fast":
                s["critical_error"] = True
            if name == "unsupported":
                s["required_claims"][0]["verdict"] = "insufficient"
            if name == "uncertain":
                s["needs_review"] = True
            report, _ = evaluate(task, submission, seal("EpisodeMetrics", m, KEY), config, root, KEY,
                                 semantic_envelope=seal("SemanticAssessment", s, KEY))
            reports.append(report)
        return {"kind": "synthetic_contract_smoke_test", "real_model_run": False, "human_calibrated": False,
                "examples": [{"case": r["metrics"]["episode_id"], "tier": r["tier"], "reward": r["reward"]} for r in reports],
                "scorecard": scorecard(reports), "calibration": calibration([])}
