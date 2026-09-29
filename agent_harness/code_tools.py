"""Bounded code-understanding tools over immutable, host-selected Git resources."""
from __future__ import annotations

import ast
from dataclasses import dataclass
import fnmatch
import hashlib
from pathlib import Path, PurePosixPath
import re
import subprocess
from typing import Any

from .contracts import ToolContext, ToolObservation, ToolSpec


def _safe_path(path: str) -> str:
    if not isinstance(path, str) or not path or "\0" in path or "\\" in path:
        raise ValueError("Expected a repository-relative path")
    parsed = PurePosixPath(path)
    if parsed.is_absolute() or any(part in ("", ".", "..") for part in path.split("/")):
        raise ValueError("Path traversal is not permitted")
    return path


class GitRepository:
    """Read Git objects rather than mutable checkout files or repository hooks."""

    max_blob_bytes = 2_000_000

    def __init__(self, path: Path | str, commit: str):
        self.path = Path(path).resolve()
        if not re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", commit):
            raise ValueError("A full lowercase pinned Git commit is required")
        if self._git("cat-file", "-t", commit).strip() != b"commit":
            raise ValueError("Pinned object is not a commit")
        self.commit = commit
        self._entries = {}
        for entry in self._git("ls-tree", "-rz", "--full-tree", commit).split(b"\0"):
            if not entry:
                continue
            metadata, raw_path = entry.split(b"\t", 1)
            mode, kind, oid = metadata.decode().split()
            name = _safe_path(raw_path.decode("utf-8"))
            if mode not in ("100644", "100755") or kind != "blob":
                raise ValueError(f"Unsupported symlink/submodule: {name}")
            self._entries[name] = oid

    def _git(self, *args):
        # communicate drains output and cleanup also covers harness-level interrupts.
        process = subprocess.Popen(["git", "--no-replace-objects", "-C", str(self.path), *args],
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            stdout, stderr = process.communicate(timeout=10)
            if process.returncode:
                raise ValueError("Git resource access failed: " + stderr.decode(errors="replace")[:500])
            return stdout
        except BaseException:
            process.kill()
            process.communicate()
            raise

    def files(self, pattern: str = "*") -> list[str]:
        return [path for path in sorted(self._entries) if fnmatch.fnmatchcase(path, pattern)]

    def blob(self, path: str, *, bounded: bool = True) -> bytes:
        path = _safe_path(path)
        if path not in self._entries:
            raise ValueError("Path is not a regular tracked file at the pinned commit")
        oid = self._entries[path]
        if bounded and int(self._git("cat-file", "-s", oid)) > self.max_blob_bytes:
            raise ValueError("File exceeds source-read byte limit")
        return self._git("cat-file", "blob", oid)

    def snapshot_hashes(self):
        return {path: hashlib.sha256(self.blob(path, bounded=False)).hexdigest()
                for path in self.files()}

    def text(self, path):
        blob = self.blob(path)
        if b"\0" in blob:
            raise ValueError("Binary files are not supported")
        return blob.decode("utf-8"), hashlib.sha256(blob).hexdigest()


def _schema(properties, required=()):
    return {"type": "object", "properties": properties, "required": list(required),
            "additionalProperties": False}


_STRING = {"type": "string", "minLength": 1, "maxLength": 1000}
_LIMIT = {"type": "integer", "minimum": 1, "maximum": 200}
_OFFSET = {"type": "integer", "minimum": 0}


@dataclass
class CodeTool:
    spec: ToolSpec
    repository_resource: str = "repository"
    sandbox_resource: str = "sandbox"

    def execute(self, arguments: dict, context: ToolContext) -> ToolObservation:
        try:
            return self._execute(arguments, context)
        except (ValueError, UnicodeError, SyntaxError, FileNotFoundError) as exc:
            return ToolObservation(status="error", content=None, error=str(exc))

    def _execute(self, args, context):
        name = self.spec.name
        if name == "read_artifact":
            value = context.artifacts.get(context.episode_id, args["artifact_id"])
            # A bounded textual view works for every JSON artifact shape.
            import json
            text = json.dumps(value, ensure_ascii=False)
            offset, limit = args.get("offset", 0), args.get("limit", 8000)
            return ToolObservation("ok", {"text": text[offset:offset + limit],
                "next_offset": offset + limit if offset + limit < len(text) else None},
                truncated=offset + limit < len(text))
        repo = context.resources[self.repository_resource]
        if not isinstance(repo, GitRepository):
            raise ValueError("Host resource is not a GitRepository")
        if name == "list_files":
            paths = repo.files(args.get("glob", "*"))
            offset, limit = args.get("offset", 0), args.get("limit", 100)
            return ToolObservation("ok", {"paths": paths[offset:offset + limit],
                "next_offset": offset + limit if offset + limit < len(paths) else None})
        if name == "read_file":
            path = args["path"]
            text, digest = repo.text(path)
            lines = text.splitlines()
            start, count = args.get("start_line", 1), args.get("line_count", 100)
            selected = lines[start - 1:start - 1 + count]
            if not selected:
                return ToolObservation("ok", {"path": path, "start_line": start, "text": "",
                                              "total_lines": len(lines)}, evidence=[])
            payload = "\n".join(selected)
            truncated = len(payload) > 16000
            payload = payload[:16000]
            return ToolObservation("ok", {"path": path, "start_line": start,
                "text": payload, "total_lines": len(lines)}, evidence=[{
                "path": path, "commit": repo.commit, "file_sha256": digest,
                "start_line": start, "end_line": start + len(selected) - 1}], truncated=truncated)
        if name == "search_code":
            paths = repo.files(args.get("glob", "*"))
            offset, limit = args.get("offset", 0), args.get("limit", 50)
            matches, consumed, examined = [], 0, 0
            next_line = 1
            truncated = False
            for path in paths[offset:offset + 200]:
                try:
                    text, digest = repo.text(path)
                except (ValueError, UnicodeError):
                    examined += 1
                    continue
                consumed += len(text)
                if consumed > 2_000_000:
                    truncated = True
                    break
                examined += 1
                for line_number, line in enumerate(text.splitlines(), 1):
                    if examined == 1 and line_number < args.get("start_line", 1):
                        continue
                    if args["query"] in line:
                        matches.append({"path": path, "line": line_number, "text": line[:1000],
                                        "commit": repo.commit, "file_sha256": digest})
                        if len(matches) >= limit:
                            examined -= 1
                            next_line = line_number + 1
                            truncated = True
                            break
                if truncated:
                    break
            next_offset = offset + examined if offset + examined < len(paths) else None
            return ToolObservation("ok", {"matches": matches, "next_offset": next_offset,
                "next_line": next_line, "files_examined": examined}, evidence=matches, truncated=truncated or next_offset is not None)
        if name == "find_symbols":
            path = args["path"]
            if not path.endswith(".py"):
                raise ValueError("Symbol lookup currently supports Python files only")
            text, digest = repo.text(path)
            symbols = [{"name": node.name, "kind": type(node).__name__, "line": node.lineno,
                        "end_line": node.end_lineno} for node in ast.walk(ast.parse(text))
                       if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
                       and (not args.get("name") or args["name"] in node.name)]
            symbols.sort(key=lambda entry: entry["line"])
            return ToolObservation("ok", {"symbols": symbols[:200]}, evidence=[{
                "path": path, "commit": repo.commit, "file_sha256": digest}], truncated=len(symbols) > 200)
        env = context.resources[self.sandbox_resource]
        if name == "run_tests":
            selections = args.get("paths", [])
            for selection in selections:
                # Only tracked file selectors; no switches, node expressions or arbitrary argv.
                if selection.startswith("-"):
                    raise ValueError("Test selection cannot be an option")
                repo.blob(selection)
            command = [*env.test_runner, *selections]
            result = env.run(repo, command)
        elif name == "python_probe":
            result = env.run(repo, [env.python_executable, "-"], stdin=args["code"])
        else:
            raise ValueError("Unknown tool")
        result["command"] = command if name == "run_tests" else [env.python_executable, "-"]
        artifact_id = context.artifacts.put(context.episode_id, result)
        return ToolObservation("ok", {**result, "artifact_id": artifact_id},
                               truncated=bool(result.get("truncated")))


def code_understanding_tools(repository_resource="repository", sandbox_resource="sandbox",
                             include_execution=False):
    """Construct tools with host-selected resource handles and no domain loop logic."""
    definitions = [
        ("list_files", "list", _schema({"glob": _STRING, "offset": _OFFSET, "limit": _LIMIT}),
         "List regular tracked paths at the pinned commit, with pagination."),
        ("search_code", "search", _schema({"query": _STRING, "glob": _STRING,
            "offset": _OFFSET, "limit": _LIMIT,
            "start_line": {"type": "integer", "minimum": 1}}, ("query",)),
         "Bounded, case-sensitive literal search; Resume with next_offset and next_line as start_line."),
        ("read_file", "read", _schema({"path": _STRING,
            "start_line": {"type": "integer", "minimum": 1}, "line_count": _LIMIT}, ("path",)),
         "Read a bounded line range from a pinned Git blob with a full-file hash."),
        ("find_symbols", "symbols", _schema({"path": _STRING, "name": _STRING}, ("path",)),
         "Find Python class and function definitions using AST parsing."),
        ("read_artifact", "read", _schema({"artifact_id": _STRING, "offset": _OFFSET,
            "limit": {"type": "integer", "minimum": 1, "maximum": 16000}}, ("artifact_id",)),
         "Read a bounded JSON text slice from an artifact owned by this episode."),
    ]
    if include_execution:
        definitions.extend([
            ("run_tests", "execute", _schema({"paths": {"type": "array", "maxItems": 50,
                "items": _STRING}}), "Run tracked test files using the host-configured isolated runner."),
            ("python_probe", "execute", _schema({"code": {"type": "string", "minLength": 1,
                "maxLength": 16000}}, ("code",)), "Run Python in a fresh isolated environment; never on the host."),
        ])
    tools = []
    for name, capability, schema, description in definitions:
        resources = () if name == "read_artifact" else (repository_resource,)
        if capability == "execute":
            resources += (sandbox_resource,)
        spec = ToolSpec(name=name, version="1", type="code_understanding", capabilities=(capability,),
                        input_schema=schema, required_resources=resources, description=description,
                        timeout_seconds=120 if capability == "execute" else 15,
                        max_output_bytes=32000)
        tools.append(CodeTool(spec, repository_resource, sandbox_resource))
    return tools
