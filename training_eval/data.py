"""Tokenized data boundaries; dataset generation belongs to external packages."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable

from .contracts import SFTExample, Task


def _identity(kind: str, payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                         allow_nan=False).encode()
    return f"{kind}:sha256:{hashlib.sha256(encoded).hexdigest()}"


def validate_sft(example: SFTExample) -> None:
    if not isinstance(example.example_id, str) or not example.example_id or not isinstance(example.split, str) or not example.split:
        raise ValueError("SFT examples require an id and split")
    if any(value is not None and (not isinstance(value, str) or not value)
           for value in (example.task_id, example.family_id)):
        raise ValueError("Optional SFT task_id/family_id must be nonempty strings")
    if len(example.tokens) < 2 or len(example.tokens) != len(example.loss_mask):
        raise ValueError("SFT tokens and mask must have equal lengths >= 2")
    if any(type(token) is not int or token < 0 for token in example.tokens):
        raise ValueError("Token ids must be nonnegative integers")
    if any(not math.isfinite(weight) or weight not in (0, 1) for weight in example.loss_mask):
        raise ValueError("SFT loss_mask must contain finite binary values")
    if example.loss_mask[0] != 0:
        raise ValueError("First token has no prediction context and must be masked")
    if not any(example.loss_mask[1:]):
        raise ValueError("SFT example must contain at least one supervised target token")


class InMemorySFTDataset:
    def __init__(self, examples: Iterable[SFTExample], *, tokenizer: str, renderer: str):
        if not tokenizer or not renderer:
            raise ValueError("Tokenizer and renderer identities are required")
        self._examples = tuple(SFTExample(e.example_id, tuple(e.tokens), tuple(e.loss_mask),
                                         e.split, e.task_id, e.family_id)
                               for e in examples)
        for example in self._examples:
            validate_sft(example)
        if len({e.example_id for e in self._examples}) != len(self._examples):
            raise ValueError("Duplicate SFT example ids")
        self.tokenizer = tokenizer
        self.renderer = renderer
        self.identity = _identity("sft", {"tokenizer": tokenizer, "renderer": renderer,
                                         "examples": [asdict(e) for e in self._examples]})

    def __len__(self) -> int:
        return len(self._examples)

    def get(self, index: int) -> SFTExample:
        return self._examples[index]


class JsonlSFTDataset(InMemorySFTDataset):
    """Rows have example_id, tokens, loss_mask; split/task_id/family_id are optional.

    Tokenizer/renderer are explicit caller-supplied identities, not inferred from
    token ids. Inputs are loaded once so subsequent file edits cannot change a run.
    """
    def __init__(self, path: str | Path, *, tokenizer: str, renderer: str):
        examples = []
        with Path(path).open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    examples.append(SFTExample(row["example_id"], tuple(row["tokens"]),
                                               tuple(row["loss_mask"]), row.get("split", "train"),
                                               row.get("task_id"), row.get("family_id")))
                    validate_sft(examples[-1])
                except (ValueError, TypeError, KeyError) as error:
                    raise ValueError(f"Invalid SFT JSONL row {line_number}: {error}") from error
        super().__init__(examples, tokenizer=tokenizer, renderer=renderer)


class InMemoryTaskSource:
    def __init__(self, tasks: Iterable[Task]):
        # JSON round-trip detaches caller-owned payloads and rejects unstable or
        # nonserializable values, ensuring the identity describes what is served.
        rows = json.loads(json.dumps([asdict(t) for t in tasks], allow_nan=False))
        self._tasks = tuple(Task(**row) for row in rows)
        if any(not t.task_id or not t.split or not t.family_id for t in self._tasks):
            raise ValueError("Tasks require task_id, split, and family_id")
        if len({t.task_id for t in self._tasks}) != len(self._tasks):
            raise ValueError("Duplicate task ids")
        self.identity = _identity("tasks", rows)

    def __len__(self) -> int:
        return len(self._tasks)

    def get(self, index: int) -> Task:
        return Task(**json.loads(json.dumps(asdict(self._tasks[index]))))
