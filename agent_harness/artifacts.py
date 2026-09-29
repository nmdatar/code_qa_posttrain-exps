"""Episode-scoped, content-addressed JSON artifacts and append-only events."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any


def json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


class ArtifactStore:
    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def episode_dir(self, episode_id: str) -> Path:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", episode_id):
            raise ValueError("invalid episode identifier")
        directory = self.root / episode_id
        if directory.is_symlink():
            raise ValueError("episode directory cannot be a symlink")
        return directory

    def create_episode(self, episode_id: str) -> Path:
        directory = self.episode_dir(episode_id)
        directory.mkdir(exist_ok=False)
        (directory / "artifacts").mkdir()
        return directory

    def put(self, episode_id: str, value: Any) -> str:
        content = json_text(value).encode("utf-8")
        artifact_id = hashlib.sha256(content).hexdigest()
        directory = self.episode_dir(episode_id) / "artifacts"
        if not directory.is_dir() or directory.is_symlink():
            raise ValueError("episode has not been initialized")
        path = directory / (artifact_id + ".json")
        try:
            with path.open("xb") as stream:
                stream.write(content)
        except FileExistsError:
            if path.is_symlink() or path.read_bytes() != content:
                raise ValueError("artifact integrity violation")
        return artifact_id

    def get(self, episode_id: str, artifact_id: str) -> Any:
        if not re.fullmatch(r"[a-f0-9]{64}", artifact_id):
            raise ValueError("invalid artifact identifier")
        directory = self.episode_dir(episode_id) / "artifacts"
        path = directory / (artifact_id + ".json")
        if directory.is_symlink() or path.is_symlink():
            raise ValueError("artifact cannot be a symlink")
        content = path.read_bytes()
        if hashlib.sha256(content).hexdigest() != artifact_id:
            raise ValueError("artifact integrity violation")
        return json.loads(content)

    def append_event(self, episode_id: str, event: dict[str, Any]) -> None:
        path = self.episode_dir(episode_id) / "trajectory.jsonl"
        # O_NOFOLLOW avoids traversing a pre-existing artifact symlink.
        fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0), 0o600)
        with os.fdopen(fd, "a", encoding="utf-8") as stream:
            stream.write(json_text(event) + "\n")
            stream.flush()

    def write_metrics(self, episode_id: str, metrics: dict[str, Any]) -> str:
        path = self.episode_dir(episode_id) / "metrics.json"
        with path.open("x", encoding="utf-8") as stream:
            stream.write(json_text(metrics) + "\n")
        return str(path)
