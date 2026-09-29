"""Split audits, review queues, and frozen experiment manifests."""

import ast
from pathlib import Path
from .security import digest, file_hash
from .deterministic import git, snapshot
from .schema import validate_task, validate, EXPERIMENT


def audit_splits(tasks):
    families, lineages, ids = {}, {}, set()
    for task in tasks:
        validate_task(task)
        if task["id"] in ids:
            raise ValueError("Duplicate task ID")
        ids.add(task["id"])
        for registry, key in ((families, task["repository"]["family_id"]), (lineages, task["lineage_id"])):
            if key in registry and registry[key] != task["split"]:
                raise ValueError("Repository family or task lineage leaks across splits: " + key)
            registry[key] = task["split"]
    return {"tasks": len(tasks), "families": len(families), "lineages": len(lineages), "status": "passed"}


def freeze(config, tasks, baseline_artifact):
    audit_splits(tasks)
    if len({t["split"] for t in tasks}) != 1:
        raise ValueError("Freeze one dataset split per experiment; audit splits jointly first")
    # Budget categories must be comparable and cannot quietly drift by question.
    categories = {}
    for task in tasks:
        category = task["category"]
        if category in categories and categories[category] != task["budgets"]:
            raise ValueError("Tasks in the same category have different budgets")
        categories[category] = task["budgets"]
    result = {**config, "frozen": True, "baseline_manifest_hash": digest(baseline_artifact),
              "budgets_manifest_hash": digest(categories),
              "task_manifest": sorted([{"id": t["id"], "task_hash": digest(t)} for t in tasks], key=lambda t: t["id"])}
    validate(result, EXPERIMENT)
    return result


def seed_calibration(repositories, per_family=10):
    """Generate grounded DRAFT tasks, not fictitious human-reviewed gold.

    repository manifests contain path, url, family_id. Every source is pinned
    and syntax inspected. Complex behavior and negative evidence still require
    reviewer completion before admission to training/evaluation.
    """
    if len({r["family_id"] for r in repositories}) < 10:
        raise ValueError("Calibration requires at least 10 repository families")
    tasks = []
    for repo in repositories:
        root = Path(repo["path"])
        commit = git(root, "rev-parse", "HEAD")
        snapshot(root, commit)
        choices = []
        for relative in git(root, "ls-files", "*.py").splitlines():
            p = root / relative
            if not p.is_file() or p.is_symlink():
                continue
            try:
                tree = ast.parse(p.read_text())
            except (SyntaxError, UnicodeError):
                continue
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    choices.append((relative, node, file_hash(p)))
        if len(choices) < per_family:
            raise ValueError(f"{repo['family_id']} needs {per_family} top-level Python functions for automatic seeding")
        for i, (path, node, sha) in enumerate(choices[:per_family]):
            ref = {"path": path, "start_line": node.lineno, "end_line": node.end_lineno,
                   "file_sha256": sha, "symbol": node.name}
            diagram = i % 4 == 1
            condition = i % 4 == 2
            unanswerable = i % 4 == 3
            question = (f"Where is {node.name} implemented, and what parameters does it accept?" if not (diagram or condition or unanswerable)
                        else f"Explain and diagram the control flow of {node.name}, including branch conditions." if diagram
                        else f"What boundary conditions and error paths does {node.name} handle?" if condition
                        else f"What undocumented personal motivation did the original author have for {node.name}?")
            claims = [{"id": "location", "text": f"{node.name} is defined in {path} at this snapshot.",
                       "weight": 1, "separable_subparts": [], "evidence": [ref]}]
            if not (diagram or condition or unanswerable):
                args = [a.arg for a in node.args.posonlyargs + node.args.args + node.args.kwonlyargs]
                if node.args.vararg:
                    args.append("*" + node.args.vararg.arg)
                if node.args.kwarg:
                    args.append("**" + node.args.kwarg.arg)
                claims.append({"id": "parameters", "text": "Declared parameters: " + (", ".join(args) or "none"),
                               "weight": 3, "separable_subparts": [], "evidence": [ref]})
            else:
                # Explicitly draft: an expert must replace this criterion with
                # checked atomic facts. Never used as an accepted gold label.
                claims.append({"id": "review_required", "text": "REVIEW REQUIRED: establish atomic behavior/answerability claims from evidence.",
                               "weight": 3, "separable_subparts": [], "evidence": [ref]})
            task = {"schema_version": "1.0", "id": f"{repo['family_id']}-{i:03}", "question": question,
                    "repository": {"family_id": repo["family_id"], "commit": commit, "url": repo["url"]},
                    "lineage_id": digest({"family": repo["family_id"], "path": path, "symbol": node.name}),
                    "split": "development", "category": "calibration_draft", "answerability": "unanswerable" if unanswerable else "answerable",
                    "claims": claims, "critical_errors": [], "permitted_tools": ["search", "read_file"],
                    "budgets": {"latency_seconds": 120, "compute_units": 100000, "max_tool_calls": 30,
                                "max_output_tokens": 8000, "max_submission_bytes": 64000},
                    "diagram": {"required": diagram, "criteria": ["Accurately represent branches and control flow"] if diagram else [],
                                "allowed_abstractions": []}, "probes": [], "human_reviewed": False, "gold_status": "draft"}
            validate_task(task)
            tasks.append(task)
    audit_splits(tasks)
    return tasks
