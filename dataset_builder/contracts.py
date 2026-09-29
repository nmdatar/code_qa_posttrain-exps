"""Solver-visible dataset contracts and private/public consistency checks.

The public record is deliberately an allowlist. Assertions, reference answers,
and provenance annotations belong in separate private artifacts. Structural
validation cannot establish the semantic quality of a question or its gold.
"""

from pathlib import PurePosixPath, PureWindowsPath
import re

from qa_eval.dataset import audit_splits
from qa_eval.schema import NONEMPTY, TASK, VERSION, obj, validate
from qa_eval.security import digest


PUBLIC_TASK = obj({
    "schema_version": VERSION,
    "id": NONEMPTY,
    "system_prompt": NONEMPTY,
    "user_prompt": NONEMPTY,
    "repository": TASK["properties"]["repository"],
    "environment_id": NONEMPTY,
    "split": TASK["properties"]["split"],
    "permitted_tools": TASK["properties"]["permitted_tools"],
    "budgets": TASK["properties"]["budgets"],
})


def canonical_hash(value):
    """Hash canonical JSON, consistent with the existing evaluation contracts."""
    return digest(value)


def validate_public_task(task):
    """Reject private fields and incomplete commit references; return None."""
    validate(task, PUBLIC_TASK)
    _validate_commit(task["repository"]["commit"])
    if len(task["permitted_tools"]) != len(set(task["permitted_tools"])):
        raise ValueError("Duplicate permitted tool")


def _validate_commit(commit):
    if not re.fullmatch(r"(?:[a-f0-9]{40}|[a-f0-9]{64})", commit):
        raise ValueError("Repository commit must be a full 40- or 64-character hash")


def _validate_evidence_path(path):
    posix, windows = PurePosixPath(path), PureWindowsPath(path)
    if (posix.is_absolute() or windows.drive or "\\" in path or "\x00" in path
            or any(part in {"", ".", ".."} for part in path.split("/"))):
        raise ValueError("Evidence path must be a normalized repository-relative path")


def validate_bundle(public_tasks, private_tasks, environments=None):
    """Validate paired public/private records and jointly audit all splits.

    Call over the entire candidate release, not one split at a time. Filesystem
    confinement and evidence content checks happen separately against snapshots.
    This does not promote machine-checked drafts to human-accepted gold.
    """
    public_by_id = {}
    for task in public_tasks:
        validate_public_task(task)
        if task["id"] in public_by_id:
            raise ValueError("Duplicate public task ID")
        public_by_id[task["id"]] = task
    audit_splits(private_tasks)
    if set(public_by_id) != {task["id"] for task in private_tasks}:
        raise ValueError("Public and private task IDs must match exactly")
    if environments is not None:
        environment_by_id = {}
        for environment in environments:
            if environment["id"] in environment_by_id:
                raise ValueError("Duplicate environment ID")
            environment_by_id[environment["id"]] = environment
        for public in public_tasks:
            environment = environment_by_id.get(public["environment_id"])
            if environment is None:
                raise ValueError("Public task references an unknown environment")
            if environment.get("repository") != public["repository"]:
                raise ValueError("Task and environment repositories must match")
    for private in private_tasks:
        _validate_commit(private["repository"]["commit"])
        public = public_by_id[private["id"]]
        for field in ("schema_version", "repository", "split", "permitted_tools", "budgets"):
            if public[field] != private[field]:
                raise ValueError(f"Public/private {field} mismatch for {private['id']}")
        if public["user_prompt"] != private["question"]:
            raise ValueError("Public prompt must match private question")
        if private["gold_status"] == "accepted" and not private["human_reviewed"]:
            raise ValueError("Accepted gold requires human review; machine checks leave gold draft")
        for claim in private["claims"]:
            for evidence in claim["evidence"]:
                _validate_evidence_path(evidence["path"])
                if evidence["start_line"] > evidence["end_line"]:
                    raise ValueError("Evidence line range is reversed")
