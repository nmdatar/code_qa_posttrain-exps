#!/usr/bin/env python3
"""Prepare a pinned SWE-QA-Pro release using only the Python standard library."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
import sys
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
DATASET = "TIGER-Lab/SWE-QA-Pro-Bench"
REVISION = "596892dac60b6f500f01a7dc2becb9f66593b7b7"
SHA256 = "bba4aade95e707d012e11e622a40687bc0efd8cc509ed4a7dd3ba24c6e365737"
BASE_URL = f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}"


def load_records(raw):
    if hashlib.sha256(raw).hexdigest() != SHA256:
        raise ValueError("Dataset checksum mismatch; refusing an unverified release")
    records = [json.loads(line) for line in raw.splitlines() if line.strip()]
    seen = set()
    for row in records:
        for field in ("repo", "commit_id", "question", "answer"):
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise ValueError(f"Missing or invalid {field}")
        if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", row["repo"]):
            raise ValueError("Invalid repository name")
        if not re.fullmatch(r"[0-9a-f]{40}", row["commit_id"]):
            raise ValueError("Invalid commit hash")
        key = (row["repo"], row["commit_id"], row["question"])
        if key in seen:
            raise ValueError("Duplicate question")
        seen.add(key)
    if len(records) != 260 or len({r["repo"] for r in records}) != 26:
        raise ValueError("Unexpected dataset size")
    return records


def task_id(row):
    identity = json.dumps([row[k] for k in ("repo", "commit_id", "question")])
    return "swe-qa-pro-" + hashlib.sha256(identity.encode()).hexdigest()[:16]


def split_records(records):
    tasks, references = [], []
    for row in records:
        identifier = task_id(row)
        tasks.append({"id": identifier, "repo": row["repo"],
                      "commit_id": row["commit_id"], "question": row["question"]})
        references.append({"id": identifier, "reference_answer": row["answer"],
                           "cluster": row["cluster"], "qa_type": row["qa_type"]})
    return tasks, references


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


from dataset_builder.checkout import git, checkout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default="getsentry/responses")
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--all", action="store_true", help="Prepare all 260 tasks")
    parser.add_argument("--checkout", action="store_true", help="Fetch selected repository snapshots")
    parser.add_argument("--output", type=Path, default=ROOT / "data/swe-qa-pro")
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    source = args.output / "source.jsonl"
    if not source.exists():
        with urllib.request.urlopen(BASE_URL + "/data/test.jsonl", timeout=60) as response:
            raw = response.read()
        load_records(raw)
        source.write_bytes(raw)
    rows = load_records(source.read_bytes())
    card = args.output / "DATASET_CARD.md"
    if not card.exists():
        with urllib.request.urlopen(BASE_URL + "/README.md", timeout=60) as response:
            card.write_bytes(response.read())
    selected = rows if args.all else [r for r in rows if r["repo"] == args.repo][:args.limit]
    if not selected:
        parser.error(f"No records found for {args.repo}")
    tasks, references = split_records(selected)
    write_jsonl(args.output / "tasks.jsonl", tasks)
    write_jsonl(args.output / "references.jsonl", references)
    # This file preserves upstream fields for the official evaluation harness.
    write_jsonl(args.output / "upstream.jsonl", selected)
    snapshots = []
    if args.checkout:
        unique = {(r["repo"], r["commit_id"]): r for r in selected}
        snapshots = [checkout(row, ROOT / "artifacts/repos") for row in unique.values()]
    manifest = {"dataset": DATASET, "revision": REVISION, "source_sha256": SHA256,
                "declared_license": "MIT (dataset card)", "split": "test",
                "purpose": "pipeline smoke test" if not args.all else "full evaluation",
                "source_records": len(rows), "selected_records": len(tasks),
                "task_ids": [r["id"] for r in tasks], "snapshots": snapshots,
                "model_evaluated": False}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
