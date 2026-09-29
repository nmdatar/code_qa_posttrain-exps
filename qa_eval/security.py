"""Bindings and authenticated exchange between the trusted harness and grader."""

import hashlib
import hmac
import json
import os
from pathlib import Path


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def load_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result
    return json.loads(Path(path).read_text(), object_pairs_hook=unique,
                      parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))


def read_key(path):
    p = Path(path)
    if os.name != "nt" and p.stat().st_mode & 0o077:
        raise ValueError("Harness key must have permissions 0600")
    key = p.read_bytes()
    if len(key) < 32:
        raise ValueError("Harness key must contain at least 32 bytes")
    return key


def seal(kind, payload, key):
    body = {"kind": kind, "payload": json.loads(canonical(payload))}
    return {**body, "signature": hmac.new(key, canonical(body), hashlib.sha256).hexdigest()}


def unseal(envelope, kind, key):
    if set(envelope) != {"kind", "payload", "signature"} or envelope["kind"] != kind:
        raise ValueError("Invalid authenticated artifact type")
    expected = seal(kind, envelope["payload"], key)["signature"]
    if not isinstance(envelope["signature"], str) or not hmac.compare_digest(expected, envelope["signature"]):
        raise ValueError("Invalid artifact signature")
    return json.loads(canonical(envelope["payload"]))


def bindings(task, submission):
    return {"task_hash": digest(task), "submission_hash": digest(submission)}


def check_bindings(payload, task, submission):
    for k, v in bindings(task, submission).items():
        if payload.get(k) != v:
            raise ValueError(f"Stale or mismatched {k}")


def safe_path(root, relative):
    """Prevent absolute paths, parent traversal, and symlink escapes."""
    p = Path(relative)
    if p.is_absolute() or not p.parts or ".." in p.parts:
        raise ValueError(f"Invalid repository path: {relative}")
    base = Path(root).resolve()
    resolved = (base / p).resolve()
    if not resolved.is_relative_to(base):
        raise ValueError("Path escapes repository")
    return resolved
