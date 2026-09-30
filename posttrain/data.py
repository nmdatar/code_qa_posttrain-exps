"""Fail-closed bridge between collection releases and strict verifier tasks."""
from copy import deepcopy
from pathlib import Path
import hashlib
import json
from collections import Counter
from qa_eval.schema import TASK, validate

class MissingRubric(ValueError):
    pass

def read_jsonl(path):
    path = Path(path)
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []

def _index(rows, key):
    out = {}
    for row in rows:
        ident = row[key]
        if ident in out:
            raise ValueError(f"Duplicate {key}: {ident}")
        out[ident] = row
    return out

def adapt_task(public, grading, rubric=None):
    """Require authored strict rubrics; never infer missing claims or weights."""
    if grading is None or grading.get("task_id") != public["id"]:
        raise MissingRubric("Missing grading record or task ID mismatch")
    candidate = rubric if rubric is not None else grading.get("record")
    if candidate is None:
        raise MissingRubric("Reviewed reference needs an explicitly authored strict claim rubric")
    task = deepcopy(candidate)
    validate(task, TASK)
    split = {"dev": "development", "test": "final_test", "training": "train"}.get(public["split"], public["split"])
    for field, expected in (("id", public["id"]), ("repository", public["repository"]),
                            ("question", public["user_prompt"]), ("split", split),
                            ("permitted_tools", public["permitted_tools"]), ("budgets", public["budgets"])):
        if task[field] != expected:
            raise ValueError(f"Private/public {field} binding mismatch")
    return task

def inventory_release(release, review_policy="automated"):
    if review_policy not in {"automated", "human"}: raise ValueError("Unsupported review policy")
    root = Path(release)
    reviews = {}
    if (root / "private/rubric_reviews.jsonl").exists():
        validate_artifacts(root)
        artifacts = json.loads((root / "manifest.json").read_text()).get("artifacts", {})
        names = set(artifacts) if isinstance(artifacts, dict) else {x["path"] for x in artifacts}
        if {"public/tasks.jsonl", "private/grading.jsonl", "private/rubric_reviews.jsonl"}.issubset(names):
            reviews = _index(read_jsonl(root / "private/rubric_reviews.jsonl"), "task_id")
    public = _index(read_jsonl(root / "public/tasks.jsonl"), "id")
    grading = _index(read_jsonl(root / "private/grading.jsonl"), "task_id")
    environments = _index(read_jsonl(root / "public/environments.jsonl"), "environment_id")
    if grading.keys() - public.keys():
        raise ValueError("Orphan private grading records")
    families = {}
    items = []
    for ident, row in sorted(public.items()):
        family = row["repository"]["family_id"].casefold()
        split = row["split"]
        if family in families and families[family] != split:
            raise ValueError("Repository family appears in multiple splits")
        families[family] = split
        env = environments.get(row["environment_id"])
        reasons = []
        ready = bool(env and env.get("runtime_checked") and env.get("repository") == row["repository"])
        if not ready:
            reasons.append("environment_missing_unchecked_or_snapshot_mismatch")
        scorable = False
        strict = None
        if ident not in grading:
            reasons.append("grading_quarantined_or_missing")
        else:
            try:
                strict = adapt_task(row, grading[ident]); scorable = True
            except (ValueError, KeyError) as exc:
                reasons.append(str(exc))
        from qa_eval.review import automated_rubric_supported
        machine_admitted = bool(scorable and automated_rubric_supported(strict, reviews.get(ident)))
        human_admitted = bool(scorable and strict.get("human_reviewed") and strict.get("gold_status") == "accepted")
        admitted = machine_admitted if review_policy == "automated" else human_admitted
        if scorable and not admitted:
            reasons.append("independent_automated_rubric_review_missing_or_stale" if review_policy == "automated" else "human_gold_admission_pending")
        items.append({"task_id": ident, "family_id": family, "split": split,
                      "environment_id": row["environment_id"], "runtime_record_ready": ready,
                      "strict_scorable": scorable, "reference_available": ident in grading,
                      "human_reviewed": bool(grading.get(ident, {}).get("record", {}).get("human_reviewed", False)),
                      "gold_admitted": admitted, "automated_admitted": machine_admitted, "human_admitted": human_admitted, "ready": bool(ready and admitted), "blockers": reasons})
    return {"schema_version": "1.0", "release": str(root), "review_policy": review_policy, "tasks": len(items),
            "environments": len(environments), "families": len(families),
            "split_counts": dict(Counter(x["split"] for x in items)),
            "runtime_record_ready": sum(x["runtime_record_ready"] for x in items),
            "strict_scorable": sum(x["strict_scorable"] for x in items),
            "automated_admitted": sum(x["automated_admitted"] for x in items), "human_admitted": sum(x["human_admitted"] for x in items),
            "gold_admitted": sum(x["gold_admitted"] for x in items), "reference_available": len(grading), "ready": sum(x["ready"] for x in items),
            "human_reviewed": sum(x["human_reviewed"] for x in items), "items": items,
            "note": "Runtime readiness is recorded collection evidence, not a new live Modal check. Admission uses the stated review policy. Automated admission does not imply human approval or measured grader calibration."}

def load_tasks(release, split="development", strict=True):
    root = Path(release)
    grading = _index(read_jsonl(root / "private/grading.jsonl"), "task_id")
    tasks = []
    for row in read_jsonl(root / "public/tasks.jsonl"):
        if row["split"] != split:
            continue
        try:
            tasks.append(adapt_task(row, grading.get(row["id"])))
        except MissingRubric:
            if strict:
                raise
    return tasks

def validate_artifacts(release):
    root = Path(release)
    manifest = json.loads((root / "manifest.json").read_text())
    checked = 0
    artifacts = manifest.get("artifacts", {})
    entries = artifacts.items() if isinstance(artifacts, dict) else ((x["path"], x) for x in artifacts)
    for name, spec in entries:
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError("Artifact path escapes release")
        expected = spec if isinstance(spec, str) else spec["sha256"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Artifact hash mismatch: {name}")
        checked += 1
    return {"status": "passed", "artifacts_checked": checked}
