"""Independent scorecards, repository-clustered comparisons, and calibration."""

from collections import defaultdict
import math
import random
import statistics
from .schema import REPORT, validate


def mean(values):
    return statistics.fmean(values) if values else None


def percentile(values, q):
    if not values:
        return None
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    low, high = math.floor(pos), math.ceil(pos)
    return ordered[low] + (ordered[high] - ordered[low]) * (pos - low)


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def validate_reports(reports):
    seen = set()
    for r in reports:
        validate(r, REPORT)
        if r["metrics"]:
            identity = (r["experiment_hash"], r["metrics"]["episode_id"])
            if identity in seen:
                raise ValueError("Duplicate episode in scorecard")
            seen.add(identity)
    for field in ("reward_version", "experiment_hash", "environment_id", "latency_enabled", "judge_role", "scoring_contract_hash"):
        if len({r[field] for r in reports}) > 1:
            raise ValueError(f"Do not mix {field} values in one scorecard")


def scorecard(reports):
    validate_reports(reports)
    planned = set(reports[0]["planned_task_hashes"]) if reports else set()
    if any(set(r["planned_task_hashes"]) != planned for r in reports):
        raise ValueError("Mixed planned task cohorts")
    observed = {r["task_hash"] for r in reports}
    if observed - planned:
        raise ValueError("Unplanned task in report")
    missing = len(planned - observed)
    n = len(reports) + missing
    accepted = sum(r["tier"] == "accepted" for r in reports)
    metrics = [r["metrics"] for r in reports if r["metrics"]]
    latencies = [m["latency_seconds"] for m in metrics]
    known = [r for r in reports if r["semantic"]]
    claims = [c for r in known for c in r["semantic"]["additional_claims"]]
    # Deduplicate identical evidence links; raw citation volume is not a score.
    links = []
    for r in known:
        links += list({(x["claim_id"], x["citation_id"]): x["supported"]
                       for x in r["semantic"]["citation_links"]}.values())
    abstain = [r for r in reports if r["answerability"] != "answerable"]
    answerable = [r for r in known if r["answerability"] == "answerable"]
    diagrams = [r for r in known if r["diagram_present"]]
    known_cost = sum(m["cost"] for m in metrics if m["cost"] is not None)
    complete_metrics = len(metrics) == n
    cost = known_cost if complete_metrics and all(m["cost"] is not None for m in metrics) else None
    known_compute = sum(r["compute_units"] or 0 for r in reports)
    return {
        "episodes": len(reports), "missing_tasks": missing, "accepted_answer_rate": ratio(accepted, n),
        "material_error_rate": ratio(sum(r["material_error"] for r in reports), n),
        "required_claim_coverage": mean([r["coverage"] for r in reports] + [0] * missing),
        "unsupported_claim_rate": ratio(sum(c["verdict"] != "supported" for c in claims), len(claims)),
        "citation_support_rate": mean(links),
        "correct_abstention_or_clarification_rate": ratio(sum(r["tier"] == "accepted" for r in abstain), len(abstain)),
        "false_refusal_rate": ratio(sum(r["semantic"]["answer_mode"] != "answer" for r in answerable), len(answerable)),
        "diagram_correctness": mean([all(r["semantic"]["diagram"][k] for k in ("supported", "complete", "consistent"))
                                      and not r["semantic"]["diagram"]["material_error"] for r in diagrams]),
        "latency_p50": percentile(latencies, .5), "latency_p95": percentile(latencies, .95),
        "total_compute_units": known_compute if complete_metrics else None,
        "observed_compute_units": known_compute, "observed_cost": known_cost,
        "total_cost_all_attempts": cost, "cost_per_accepted_answer": ratio(cost, accepted) if cost is not None else None,
        "unresolved_rate": ratio(sum(r["tier"] == "unresolved" for r in reports), n),
        "infrastructure_failure_rate": ratio(sum(any(c["name"] in {"infrastructure", "metrics_authentication", "snapshot"}
                                                       and c["status"] == "unresolved" for c in r["checks"]) for r in reports), n),
        "missing_metrics": n - len(metrics), "semantically_assessed_episodes": len(known),
        "eligible_for_promotion": bool(n and not missing and len(metrics) == n and all(r["tier"] != "unresolved" for r in reports)),
        "slice_rates": {category: ratio(sum(r["tier"] == "accepted" for r in reports if r["category"] == category),
                                        sum(r["category"] == category for r in reports))
                        for category in sorted({r["category"] for r in reports})}}


def clustered_ci(values_by_family, draws=5000, seed=42):
    """Paired repository-cluster bootstrap, retaining within-family correlation."""
    if len(values_by_family) < 2:
        return None
    rng = random.Random(seed)
    groups = list(values_by_family.values())
    samples = []
    for _ in range(draws):
        values = [v for group in rng.choices(groups, k=len(groups)) for v in group]
        samples.append(statistics.fmean(values))
    return [percentile(samples, .025), percentile(samples, .975)]


def compare(baseline, candidate, draws=5000):
    a, b = scorecard(baseline), scorecard(candidate)
    blockers = []
    for label, rows in (("baseline", baseline), ("candidate", candidate)):
        if not rows or any(r["tier"] == "unresolved" or not r["metrics"] for r in rows):
            blockers.append(label + " has missing/unresolved episodes")
        if rows and {r["task_hash"] for r in rows} != set(rows[0]["planned_task_hashes"]):
            blockers.append(label + " omits preregistered tasks")
        if any(r["judge_role"] != "evaluation" or r["split"] != "final_test" for r in rows):
            blockers.append(label + " requires independent final-test grading")
        if any(not r["candidate_preselected"] for r in rows):
            blockers.append(label + " checkpoint was not preregistered")
    if baseline and candidate:
        for field in ("reward_version", "environment_id", "latency_enabled", "scoring_contract_hash"):
            if baseline[0][field] != candidate[0][field]:
                blockers.append("Mismatched " + field)
        judge_a = {(r["semantic"]["judge_family"], r["semantic"]["judge_version"]) for r in baseline if r["semantic"]}
        judge_b = {(r["semantic"]["judge_family"], r["semantic"]["judge_version"]) for r in candidate if r["semantic"]}
        if judge_a != judge_b or len(judge_a) != 1:
            blockers.append("Judge versions/families differ across checkpoints")
    def group(rows):
        result = defaultdict(list)
        for r in rows:
            result[r["task_hash"]].append(r)
        return result
    aa, bb = group(baseline), group(candidate)
    if aa.keys() != bb.keys() or any(len(aa[k]) != len(bb[k]) for k in aa.keys() & bb.keys()):
        blockers.append("Cohorts/repeat counts differ; missing attempts cannot be dropped")
    accept, errors, latency, compute = [defaultdict(list) for _ in range(4)]
    for k in aa.keys() & bb.keys():
        x, y = aa[k], bb[k]
        family = x[0]["family_id"]
        accept[family].append(mean([r["tier"] == "accepted" for r in y]) - mean([r["tier"] == "accepted" for r in x]))
        errors[family].append(mean([r["material_error"] for r in y]) - mean([r["material_error"] for r in x]))
        # Only tasks accepted on every repeat for BOTH checkpoints enter this
        # matched-correct efficiency analysis. All attempts remain in scorecard.
        if all(r["tier"] == "accepted" and r["metrics"] for r in x + y):
            latency[family].append(mean([r["metrics"]["latency_seconds"] for r in y]) - mean([r["metrics"]["latency_seconds"] for r in x]))
            compute[family].append(mean([r["compute_units"] for r in y]) - mean([r["compute_units"] for r in x]))
    cis = {name: clustered_ci(data, draws) for name, data in
           (("accepted_rate_delta", accept), ("material_error_delta", errors),
            ("matched_correct_latency_delta", latency), ("matched_correct_compute_delta", compute))}
    if len(accept) < 10:
        blockers.append("Fewer than 10 repository families; insufficient promotion evidence")
    quality = efficiency = False
    if not blockers and cis["accepted_rate_delta"] and cis["material_error_delta"]:
        ac, er = cis["accepted_rate_delta"], cis["material_error_delta"]
        quality = ac[0] > 0 and er[1] <= .005 and b["material_error_rate"] <= a["material_error_rate"]
        noninferior = ac[0] >= -.01 and er[1] <= .005
        lc, cc = cis["matched_correct_latency_delta"], cis["matched_correct_compute_delta"]
        faster = bool(baseline[0]["latency_enabled"] and lc and lc[1] < 0)
        cheaper = bool(cc and cc[1] < 0)
        efficiency = noninferior and (faster or cheaper)
    return {"baseline": a, "candidate": b, "paired_cluster_95ci": cis,
            "repository_families": len(accept), "blockers": blockers,
            "quality_gate": quality, "efficiency_gate": efficiency,
            "promotion": "eligible_pending_human_audit" if quality or efficiency else "blocked",
            "note": "Human audit/calibration approval is separately mandatory; no automatic deployment."}


def wilson(successes, count):
    if not count:
        return None
    z = 1.959963984540054
    p, denom = successes / count, 1 + z*z / count
    center = (p + z*z / (2*count)) / denom
    radius = z * math.sqrt(p*(1-p)/count + z*z/(4*count*count)) / denom
    return [max(0, center-radius), min(1, center+radius)]


def calibration(rows):
    """Rows are independently double-reviewed examples, not model self-labels."""
    valid, pending = [], 0
    for row in rows:
        if row.get("predicted_tier") not in {"accepted", "partial", "failed", "unresolved"}:
            raise ValueError("Invalid calibration predicted tier")
        if not row.get("task_id") or not row.get("family_id"):
            raise ValueError("Calibration requires task and family identity")
        reviews = row.get("reviews", [])
        if (len(reviews) != 2 or len({r.get("reviewer") for r in reviews}) != 2
                or not all(r.get("reviewer") for r in reviews)
                or reviews[0].get("label") != reviews[1].get("label")
                or reviews[0].get("label") not in {"material_error", "partial", "accepted"}):
            pending += 1
            continue
        valid.append((row, reviews[0]["label"]))
    definitions = {
        "material_error_detection": ([(r, label) for r, label in valid if label == "material_error"],
                                     lambda r: r["predicted_tier"] == "failed", .95, "minimum"),
        "false_acceptance": ([(r, label) for r, label in valid if label in {"material_error", "partial"}],
                             lambda r: r["predicted_tier"] == "accepted", .02, "maximum"),
        "false_rejection": ([(r, label) for r, label in valid if label == "accepted"],
                            lambda r: r["predicted_tier"] != "accepted", .05, "maximum")}
    results = {}
    for name, (subset, predicate, threshold, direction) in definitions.items():
        hits = sum(predicate(r) for r, _ in subset)
        ci = wilson(hits, len(subset))
        passes = bool(ci and (ci[0] >= threshold if direction == "minimum" else ci[1] <= threshold))
        results[name] = {"numerator": hits, "denominator": len(subset), "rate": ratio(hits, len(subset)),
                         "wilson_95ci": ci, "threshold": threshold, "passes_with_uncertainty": passes}
    families = len({r["family_id"] for r, _ in valid})
    task_count = len({r["task_id"] for r, _ in valid})
    # Wilson intervals assume independent items. Require independent task IDs
    # within each stratum rather than count variants as independent evidence.
    correlated = any(len({r["task_id"] for r, _ in subset}) != len(subset)
                     for subset, *_ in definitions.values())
    passed = (not pending and not correlated and families >= 10 and task_count >= 100
              and all(x["passes_with_uncertainty"] for x in results.values()))
    return {"status": "passed" if passed else "needs_more_review_or_examples", "metrics": results,
            "pending_or_disputed": pending, "reviewed_tasks": task_count, "families": families,
            "correlated_variants_in_strata": correlated,
            "note": "Human labels must be supplied by reviewers. Wilson item intervals do not remove repository correlation; use diverse families and audit by family."}
