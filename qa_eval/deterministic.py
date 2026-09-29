"""Read-only source checks. Repository code is never executed in this process."""

import ast
import hashlib
import subprocess
from pathlib import Path
from .security import safe_path, file_hash, digest, canonical
from .diagrams import normalize, render_svg


def git(root, *args):
    return subprocess.run(["git", "-c", "core.fsmonitor=false", "-C", str(root), *args], capture_output=True, text=True,
                          timeout=30, check=True).stdout.strip()


def snapshot(root, commit):
    if git(root, "rev-parse", "HEAD") != commit:
        raise ValueError("Checkout commit differs from task snapshot")
    # Include untracked and ignored files: an agent might use either as evidence.
    if git(root, "status", "--porcelain", "--untracked-files=all", "--ignored"):
        raise ValueError("Repository is not pristine (including ignored/untracked files)")
    if git(root, "ls-files", "--stage").find("160000 ") >= 0:
        raise ValueError("Submodules require separately pinned snapshots; unsupported in v1")
    return digest({"commit": commit, "tree": git(root, "rev-parse", "HEAD^{tree}")})


def read_evidence(root, ref):
    p = safe_path(root, ref["path"])
    try:
        git(root, "ls-files", "--error-unmatch", "--", ref["path"])
    except subprocess.SubprocessError:
        raise ValueError("Evidence must be tracked in the pinned source tree") from None
    base = Path(root).resolve()
    original = base / ref["path"]
    if any(part.is_symlink() for part in [original, *original.parents] if part.is_relative_to(base) and part != base):
        # Disallow symlinked evidence even if it points inside the repository;
        # otherwise tracked links can expose untracked state such as .git data.
        raise ValueError("Symlink evidence is unsupported")
    if not p.is_file() or file_hash(p) != ref["file_sha256"]:
        raise ValueError("Evidence file missing or hash mismatch")
    lines = p.read_text(encoding="utf-8").splitlines()
    a, b = ref["start_line"], ref["end_line"]
    if not (1 <= a <= b <= len(lines)):
        raise ValueError("Invalid evidence line range")
    if "symbol" in ref:
        if p.suffix != ".py":
            raise ValueError("Symbol validation currently supports Python only; omit symbol for other languages")
        tree = ast.parse(p.read_text())
        found = []
        def visit(node, prefix=""):
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    name = prefix + child.name
                    found.append((name, child.lineno, child.end_lineno))
                    visit(child, name + ".")
                else:
                    visit(child, prefix)
        visit(tree)
        if not any(n == ref["symbol"] and x <= b and y >= a for n, x, y in found):
            raise ValueError("Symbol absent from cited region")
    key = digest({k: ref[k] for k in ("path", "start_line", "end_line", "file_sha256")})
    return key, {"path": ref["path"], "start_line": a, "end_line": b,
                 "text": "\n".join(f"{i}: {lines[i-1]}" for i in range(a, b+1))}


def inspect(task, submission, metrics, root):
    checks, evidence = [], {}
    def record(name, status, detail):
        checks.append({"name": name, "status": status, "detail": detail})
    fingerprint = None
    try:
        fingerprint = snapshot(root, task["repository"]["commit"])
        record("snapshot", "pass", fingerprint)
    except (ValueError, OSError, subprocess.SubprocessError) as exc:
        record("snapshot", "unresolved", str(exc))
    if task["gold_status"] != "accepted" or not task["human_reviewed"]:
        record("gold", "unresolved", "Task requires reviewed, accepted gold")
    for claim in task["claims"]:
        for ref in claim["evidence"]:
            try:
                key, content = read_evidence(root, ref)
                evidence[key] = content
            except (ValueError, OSError, SyntaxError) as exc:
                record("gold_evidence", "unresolved", str(exc))
    citation_ids = set()
    for cite in submission["citations"]:
        if cite["id"] in citation_ids:
            record("citation", "defect", "Duplicate citation ID")
        citation_ids.add(cite["id"])
        try:
            key, content = read_evidence(root, cite)
            evidence[key] = content
            record("citation:" + cite["id"], "pass", key)
        except (ValueError, OSError, SyntaxError) as exc:
            record("citation:" + cite["id"], "defect", str(exc))
    if submission["task_id"] != task["id"]:
        record("task_id", "failure", "Submission belongs to a different task")
    if len(canonical(submission)) > task["budgets"]["max_submission_bytes"]:
        record("submission_size", "failure", "Oversize submission; never truncate for grading")
    if not submission["text"].strip() and not submission["diagram"]:
        record("usable_answer", "failure", "No usable answer")
    graph, rendered = None, None
    try:
        graph = normalize(submission["diagram"])
        if graph:
            for item in graph["nodes"] + graph["edges"]:
                if any(c not in citation_ids for c in item["citations"]):
                    raise ValueError("Graph references nonexistent citation ID")
            rendered = render_svg(graph)
            record("diagram_parse_render", "pass", hashlib.sha256(rendered.encode()).hexdigest())
        elif task["diagram"]["required"]:
            record("diagram_required", "defect", "Requested diagram missing")
    except ValueError as exc:
        record("diagram_parse_render", "defect", str(exc))
    if metrics:
        if metrics["integrity_violation"]:
            record("integrity", "failure", "Harness verified an integrity violation")
        if set(metrics["tool_calls"]) - set(task["permitted_tools"]):
            record("tools", "failure", "Unauthorized tool use")
        if metrics["termination_reason"] == "infrastructure_error":
            record("infrastructure", "unresolved", "Harness infrastructure failure")
        if (len(metrics["tool_calls"]) > task["budgets"]["max_tool_calls"] or
                metrics["output_tokens"] > task["budgets"]["max_output_tokens"]):
            record("resource_limits", "failure", "Hard resource limit exceeded")
        observed = {p["id"]: p for p in metrics["probes"]}
        if len(observed) != len(metrics["probes"]):
            record("probe", "unresolved", "Duplicate probe attestation")
        for probe in task["probes"]:
            p = observed.get(probe["id"])
            if not p or not p["isolated"] or p["status"] != "completed":
                record("probe", "unresolved", "Missing isolated, successful probe attestation")
            elif p["fixture_id"] != probe["fixture_id"] or p["commit"] != task["repository"]["commit"]:
                record("probe", "unresolved", "Probe snapshot or fixture mismatch")
            elif p["stdout_sha256"] != probe["expected_stdout_sha256"]:
                record("probe", "unresolved", "Reference runtime probe contradicts gold; quarantine task")
            else:
                record("probe:" + probe["id"], "pass", "Independent fixture observation matches")
    return checks, evidence, graph, rendered, fingerprint
