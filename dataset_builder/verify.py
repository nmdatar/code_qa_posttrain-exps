"""Execute private behavior assertions in fresh isolated environments.

These checks verify task ground truth, not a model's natural-language answer.
A passing report never upgrades draft gold to human-reviewed training data.
"""
import hashlib
import json
from pathlib import Path

from .build import read_jsonl, write_json
from .contracts import canonical_hash, validate_bundle
from .environment import DockerBackend, EnvironmentError, RunLimits
from qa_eval.deterministic import read_evidence, snapshot


def passed_execution(result):
    return result["exit_code"] == 0 and not result["timed_out"] and not result["truncated"]


def verify_bundle(bundle, backend=None):
    root = Path(bundle)
    manifest = json.loads((root / "manifest.json").read_text())
    for relative, expected in manifest["artifacts"].items():
        p = root / relative
        if not p.resolve().is_relative_to(root.resolve()):
            raise ValueError("Manifest path escapes bundle")
        if hashlib.sha256(p.read_bytes()).hexdigest() != expected:
            raise ValueError("Bundle artifact hash mismatch: " + relative)
    public = read_jsonl(root / "public/tasks.jsonl")
    private = read_jsonl(root / "private/tasks.jsonl")
    environment = json.loads((root / "public/environment.json").read_text())
    built = json.loads((root / "environment-build/result.json").read_text())
    if built["status"] != "ready" or built["commit"] != environment["repository"]["commit"]:
        raise ValueError("Environment not ready or wrong commit")
    validate_bundle(public, private, [environment])
    checkout = Path(environment["snapshot_path"])
    snapshot(checkout, environment["repository"]["commit"])
    for task in private:
        for claim in task["claims"]:
            for evidence in claim["evidence"]:
                read_evidence(checkout, evidence)
    if backend is None:
        if built["backend"] == "modal":
            from .environment import ModalBackend
            backend = ModalBackend()
        else:
            backend = DockerBackend()
    limits = RunLimits(timeout_seconds=30, memory_mb=512, cpus=1, output_bytes=65536)
    image_id = built.get("image_id", built.get("image_digest"))
    records = []
    for task in read_jsonl(root / "private/assertions.jsonl"):
        for assertion in task["assertions"]:
            try:
                observed = backend.run(image_id, ["python", "-"], limits, stdin=assertion["code"])
                matched = observed["stdout"] == assertion["expected_stdout"]
                status = "passed" if passed_execution(observed) and matched else "failed"
                if observed["timed_out"] or observed["truncated"]:
                    status = "unresolved"
            except EnvironmentError as exc:
                observed = {"error": str(exc)}
                status = "infrastructure_error"
            records.append({"task_id": task["task_id"], "assertion_id": assertion["id"],
                            "fixture_sha256": hashlib.sha256(assertion["code"].encode()).hexdigest(),
                            "expected_stdout_sha256": hashlib.sha256(assertion["expected_stdout"].encode()).hexdigest(),
                            "status": status, "execution": observed})
    report = {"bundle_hash": canonical_hash(manifest), "environment_hash": canonical_hash(built),
              "backend": built["backend"], "image_id": image_id,
              "status": "passed" if records and all(r["status"] == "passed" for r in records) else "not_passed",
              "assertions": records, "human_reviewed": False, "gold_status": "draft",
              "training_eligible": False, "answer_quality_evaluated": False}
    write_json(root / "private/verification-report.json", report)
    return report
