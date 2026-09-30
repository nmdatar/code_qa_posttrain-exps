"""Grade constructed calibration answers without fabricated rollout telemetry."""
import argparse
import concurrent.futures
import json
import re
from pathlib import Path
import threading
from .storage import atomic, read, digest, BudgetLedger, BudgetExceeded
from .data import read_jsonl, validate_artifacts
from .calibration import export_packet
from .tinker_judge import judge_call
from qa_eval.source import GitSource
from qa_eval.deterministic import inspect, snapshot
from qa_eval.judging import judge
from qa_eval.grading import decide
from qa_eval.schema import SUBMISSION, validate

DEFAULT_COMMAND = ["tinker", "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16", "nemotron3_disable_thinking", "Qwen/Qwen3-8B", "qwen3_disable_thinking"]

def score_example(example, task, source_root, experiment, call):
    """Quality-only assessment; no latency/token/cost or policy reward invented."""
    if example["rubric_sha256"] != digest(task):
        raise ValueError("Calibration rubric changed")
    submission = example["submission"]
    validate(submission, SUBMISSION)
    source = GitSource(source_root, task["repository"]["commit"])
    source.catalog_paths = frozenset({e["path"] for c in task["claims"] for e in c["evidence"]} | {c["path"] for c in submission["citations"]})
    checks, evidence, graph, _, fingerprint = inspect(task, submission, None, source, diagnostic_machine_review=True)
    semantic = None
    if not any(c["status"] in {"failure", "unresolved"} for c in checks):
        try:
            # These are true construction facts, not synthetic episode metrics:
            # no repository commands or runtime probes were performed.
            semantic = judge(task, submission, experiment, "evaluation", source, evidence, graph,
                             fingerprint, {"execution_records": [], "probes": []}, call=call)
            if snapshot(source, task["repository"]["commit"]) != fingerprint:
                raise ValueError("Snapshot changed while grading")
        except BudgetExceeded:
            raise
        except Exception as exc:
            checks.append({"name": "semantic_verifier", "status": "unresolved", "detail": type(exc).__name__ + ": " + str(exc)})
    tier, coverage, material, reasons = decide(task, submission, semantic, checks)
    return {"example_id": example["example_id"], "example_sha256": digest(example), "rubric_sha256": digest(task),
            "predicted_tier": tier, "coverage": coverage, "material_error": material,
            "checks": checks, "semantic": semantic, "reasons": reasons, "origin": "constructed",
            "metrics": None, "reward": None, "catalog_scope": "reference_and_citation_paths_only",
            "experiment_sha256": digest(experiment), "human_reviewed": False}

def score_packet(examples_path, release, output, ledger, command=None, call_impl=judge_call):
    release, output = Path(release), Path(output)
    validate_artifacts(release)
    tasks = {g["task_id"]: g["record"] for g in read_jsonl(release / "private/grading.jsonl")}
    reviews = {r["task_id"]: r for r in read_jsonl(release / "private/rubric_reviews.jsonl")}
    bindings = read(release / "private/runtime-bindings.json")
    examples = read_jsonl(examples_path)
    if len({e["example_id"] for e in examples}) != len(examples): raise ValueError("Duplicate example")
    if any(not re.fullmatch(r"[A-Za-z0-9_-]+", e["example_id"]) for e in examples): raise ValueError("Unsafe example ID")
    experiment = {"id": "constructed-calibration-v1", "training_judge_family": "nemotron", "evaluation_judge_family": "qwen3",
                  "judge_command": command or DEFAULT_COMMAND, "judge_timeout_seconds": 120,
                  "max_evidence_bytes": 96000, "judge_version": "tinker-evaluation-v1"}
    output.mkdir(parents=True, exist_ok=True)
    halt = threading.Event()
    def worker(example):
        task = tasks[example["task_id"]]; review = reviews[task["id"]]
        if review["status"] != "supported" or review["rubric_sha256"] != digest(task):
            raise ValueError("No independent exact rubric review")
        path = output / "scores" / (example["example_id"] + ".json")
        if path.exists():
            existing = read(path)
            if existing["example_sha256"] != digest(example) or existing["experiment_sha256"] != digest(experiment):
                raise ValueError("Existing score binding mismatch")
            return existing
        def call(command, request, timeout):
            if halt.is_set(): raise BudgetExceeded("Calibration dispatch stopped")
            try:
                return ledger.execute("calibration_judge", .01, lambda: call_impl(command, request, timeout),
                                      {"example_id": example["example_id"], "stage": request["stage"], "quality_only": True})
            except BudgetExceeded:
                halt.set(); raise
        try:
            result = score_example(example, task, bindings[task["id"]]["source_root"], experiment, call)
        except BudgetExceeded as exc:
            result = {"example_id": example["example_id"], "example_sha256": digest(example), "rubric_sha256": digest(task),
                      "experiment_sha256": digest(experiment), "predicted_tier": "unresolved", "reasons": [str(exc)],
                      "origin": "constructed", "metrics": None, "reward": None, "human_reviewed": False}
        atomic(path, result)
        return result
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(worker, examples))
    packet_examples = []
    for example, result in zip(examples, results):
        item = {**example, "predicted_tier": result["predicted_tier"], "reviews": [],
                "evidence": {"rubric": tasks[example["task_id"]]},
                "quality_report_ref": str(output / "scores" / (example["example_id"] + ".json")),
                "status": "awaiting_independent_human_review"}
        packet_examples.append(item)
    packet = export_packet(packet_examples, output / "human-review")
    counts = {tier: sum(r["predicted_tier"] == tier for r in results) for tier in ("accepted", "partial", "failed", "unresolved")}
    report = {"examples": len(results), "tiers": counts, "human_reviewed": 0, "gate": packet["gate"],
              "packet": packet["packet"], "quality_only": True, "metrics": None, "reward": None,
              "judge_command": experiment["judge_command"], "catalog_scope": "reference_and_citation_paths_only"}
    atomic(output / "report.json", report)
    return report

def main():
    p = argparse.ArgumentParser(); p.add_argument("--examples", default="reports/posttrain/calibration/pending-examples.jsonl")
    p.add_argument("--release", default="data/releases/repo-qa-development-posttrain-v1")
    p.add_argument("--output", default="reports/posttrain/calibration/scored-v1")
    p.add_argument("--ledger", default="artifacts/posttrain/spending.json")
    args = p.parse_args()
    print(json.dumps(score_packet(args.examples, args.release, args.output, BudgetLedger(args.ledger)), indent=2))

if __name__ == "__main__": main()
