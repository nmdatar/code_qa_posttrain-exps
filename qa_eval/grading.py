"""Quality gates first, efficiency last. No environmental tokens are policy rewards."""

from .schema import validate_task, validate, SUBMISSION, METRICS, EXPERIMENT, REPORT
from .security import digest, bindings, unseal, check_bindings
from .deterministic import inspect, snapshot
from .judging import judge, validate_semantic


def compute_units(metrics, config):
    rates = config["accounting"]
    return (metrics["input_tokens"] * rates["input_token"] + metrics["output_tokens"] * rates["output_token"]
            + metrics["tool_seconds"] * rates["tool_second"])


def reward_for(tier, coverage, metrics, task, experiment):
    if tier == "unresolved":
        return None, None
    if tier == "failed":
        return 0.0, None
    if tier == "partial":
        return .2 * max(0, min(1, coverage)), None
    if tier != "accepted":
        raise ValueError("Unknown tier")
    compute = max(0, 1 - compute_units(metrics, experiment) / task["budgets"]["compute_units"])
    if experiment["latency_enabled"]:
        latency = max(0, 1 - metrics["latency_seconds"] / task["budgets"]["latency_seconds"])
        efficiency = .5 * latency + .5 * compute
    else:
        efficiency = compute
    efficiency = max(0, min(1, efficiency))
    return .9 + .1 * efficiency, efficiency


def decide(task, submission, semantic, checks):
    statuses = {c["status"] for c in checks}
    # Validity of the environment/gold is a prerequisite for grading any answer.
    if "unresolved" in statuses:
        return "unresolved", 0.0, False, [c["detail"] for c in checks if c["status"] == "unresolved"]
    if "failure" in statuses:
        return "failed", 0.0, False, [c["detail"] for c in checks if c["status"] == "failure"]
    if semantic is None:
        return "unresolved", 0.0, False, ["No authenticated semantic assessment"]
    s = semantic
    if not s["assessment_complete"] or s["needs_review"] or s["disagreements"]:
        return "unresolved", 0.0, False, ["Semantic assessment requires adjudication"]
    findings = s["required_claims"] + s["additional_claims"]
    material = (s["critical_error"] or s["false_execution_claim"] or s["diagram"]["material_error"]
                or any(c["material_error"] for c in findings))
    if material:
        return "failed", 0.0, True, ["Material falsehood, contradiction, or false execution claim"]
    specs = {c["id"]: c for c in task["claims"]}
    points = 0.0
    for c in s["required_claims"]:
        fraction = 0
        if c["verdict"] == "supported":
            if c["coverage"] == "complete":
                fraction = 1
            elif c["coverage"] == "partial" and specs[c["id"]]["separable_subparts"]:
                fraction = .5
        points += specs[c["id"]]["weight"] * fraction
    coverage = points / sum(c["weight"] for c in task["claims"])
    reasons = [c["detail"] for c in checks if c["status"] == "defect"]
    if coverage < 1:
        reasons.append("Missing or unsupported required facts")
    if any(c["verdict"] != "supported" or c["coverage"] != "complete" for c in s["additional_claims"]):
        reasons.append("Additional assertions are not fully supported")
    if s["uncited_claim_ids"]:
        reasons.append("Substantive assertions lack supporting citations")
    cited_claims = {x["claim_id"] for x in s["citation_links"] if x["supported"]}
    if {c["id"] for c in s["extracted_claims"]} - cited_claims:
        reasons.append("Extracted assertions lack assessed citation support")
    if any(not x["supported"] for x in s["citation_links"]):
        reasons.append("Citation does not entail its associated claim")
    linked = {x["citation_id"] for x in s["citation_links"] if x["supported"]}
    if {c["id"] for c in submission["citations"]} - linked:
        reasons.append("Unassessed or irrelevant submitted citation")
    expected_mode = {"answerable": "answer", "unanswerable": "abstention", "needs_clarification": "clarification"}[task["answerability"]]
    if not s["answerability_satisfied"] or s["answer_mode"] != expected_mode:
        reasons.append("Answerability requirements not met")
        if s["answer_mode"] != "answer" and task["answerability"] == "answerable":
            coverage = 0.0
    if submission["diagram"] is not None:
        d = s["diagram"]
        if not all(d[k] for k in ("assessed", "supported", "complete", "consistent", "readable")):
            reasons.append("Diagram fails semantic, consistency, or readability checks")
    return ("partial" if reasons else "accepted"), coverage, False, reasons


def evaluate(task, submission, metrics_envelope, experiment, root, key, role="evaluation",
             semantic_envelope=None, call=None):
    validate_task(task)
    validate(experiment, EXPERIMENT)
    submission_error = None
    try:
        validate(submission, SUBMISSION)
    except ValueError as exc:
        submission_error = str(exc)
    if role not in {"training", "evaluation"}:
        raise ValueError("Invalid judge role")
    if not experiment["frozen"]:
        raise ValueError("Freeze experiment before graded runs")
    if {"id": task["id"], "task_hash": digest(task)} not in experiment["task_manifest"]:
        raise ValueError("Task or its budget was changed after experiment freeze")
    if experiment["training_judge_family"] == experiment["evaluation_judge_family"]:
        raise ValueError("Independent evaluation requires a different judge family")
    if role == "training" and task["split"] != "train":
        raise ValueError("Held-out tasks cannot supply training rewards")
    checks, metrics = [], None
    try:
        metrics = unseal(metrics_envelope, "EpisodeMetrics", key)
        validate(metrics, METRICS)
        check_bindings(metrics, task, submission)
        if metrics["experiment_id"] != experiment["id"]:
            raise ValueError("Metrics are from a different experiment")
    except (ValueError, KeyError, TypeError) as exc:
        metrics = None
        checks.append({"name": "metrics_authentication", "status": "unresolved", "detail": str(exc)})
    if submission_error:
        det = [{"name": "submission_schema", "status": "failure", "detail": submission_error}]
        evidence, graph, svg, fingerprint = {}, None, None, None
    else:
        det, evidence, graph, svg, fingerprint = inspect(task, submission, metrics, root)
    checks += det
    semantic = None
    if not any(c["status"] in {"failure", "unresolved"} for c in checks):
        try:
            if semantic_envelope:
                semantic = unseal(semantic_envelope, "SemanticAssessment", key)
                validate_semantic(semantic, task, submission, experiment, role, fingerprint)
            else:
                kwargs = {"call": call} if call else {}
                semantic = judge(task, submission, experiment, role, root, evidence, graph, fingerprint, metrics, **kwargs)
            # Detect mutation during evidence collection/judging.
            if snapshot(root, task["repository"]["commit"]) != fingerprint:
                raise ValueError("Repository changed during grading")
        except Exception as exc:
            # Operational judge failures must never become negative policy labels.
            semantic = None
            checks.append({"name": "semantic_verifier", "status": "unresolved", "detail": type(exc).__name__ + ": " + str(exc)})
    tier, coverage, material, reasons = decide(task, submission, semantic, checks)
    reward, efficiency = reward_for(tier, coverage, metrics, task, experiment)
    report = {"schema_version": "1.0", "reward_version": experiment["reward_version"],
              "task_id": task["id"], **bindings(task, submission),
              "family_id": task["repository"]["family_id"], "lineage_id": task["lineage_id"],
              "split": task["split"], "category": task["category"], "answerability": task["answerability"],
              "tier": tier, "reward": reward, "coverage": coverage, "efficiency": efficiency,
              "checks": checks, "reasons": reasons, "semantic": semantic, "metrics": metrics,
              "experiment_hash": digest(experiment), "environment_id": experiment["environment_id"],
              "scoring_contract_hash": digest({k: experiment[k] for k in ("reward_version", "accounting", "latency_enabled", "budgets_manifest_hash", "environment_id")}),
              "planned_task_hashes": sorted(t["task_hash"] for t in experiment["task_manifest"]),
              "latency_enabled": experiment["latency_enabled"],
              "compute_units": compute_units(metrics, experiment) if metrics else None,
              "diagram_present": isinstance(submission, dict) and submission.get("diagram") is not None, "material_error": material,
              "judge_role": role, "candidate_preselected": experiment["candidate_preselected"]}
    validate(report, REPORT)
    return report, svg
