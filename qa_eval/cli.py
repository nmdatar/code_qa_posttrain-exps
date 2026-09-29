"""All signing/grading commands run on the trusted supervisor, never the agent."""

import argparse
import json
from pathlib import Path
import sys
from .security import load_json, read_key, seal, unseal
from .schema import SCHEMAS, validate


def write(path, value):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def read_reports(path, key):
    rows = load_json(path)
    return [unseal(r, "GradeReport", key) for r in rows]


def main(argv=None):
    parser = argparse.ArgumentParser(description="Correctness-first repository QA evaluation")
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("schemas", help="Export strict JSON schemas")
    p.add_argument("--out", default="schemas")
    p = commands.add_parser("validate")
    p.add_argument("record", choices=SCHEMAS)
    p.add_argument("file")
    p = commands.add_parser("attest", help="Trusted supervisor only: sign a validated record")
    p.add_argument("record", choices=("EpisodeMetrics", "SemanticAssessment", "GradeReport"))
    p.add_argument("--input", required=True)
    p.add_argument("--key", required=True)
    p.add_argument("--out", required=True)
    p = commands.add_parser("grade")
    for arg in ("task", "submission", "metrics", "experiment", "repo", "key", "out"):
        p.add_argument("--" + arg, required=True)
    p.add_argument("--role", choices=("training", "evaluation"), default="evaluation")
    p.add_argument("--semantic", help="Optional authenticated assessment for deterministic replay")
    p.add_argument("--svg")
    p = commands.add_parser("scorecard")
    p.add_argument("--reports", required=True, help="JSON array of signed GradeReport envelopes")
    p.add_argument("--key", required=True)
    p.add_argument("--out", required=True)
    p = commands.add_parser("compare")
    for arg in ("baseline", "candidate", "key", "out"):
        p.add_argument("--" + arg, required=True)
    p.add_argument("--bootstrap-draws", type=int, default=5000)
    p = commands.add_parser("calibration")
    p.add_argument("--reviews", required=True)
    p.add_argument("--out", required=True)
    p = commands.add_parser("audit-splits")
    p.add_argument("--tasks", required=True)
    p = commands.add_parser("freeze")
    for arg in ("config", "tasks", "baseline", "out"):
        p.add_argument("--" + arg, required=True)
    p = commands.add_parser("seed-calibration")
    p.add_argument("--repositories", required=True, help="JSON array: path, url, family_id")
    p.add_argument("--per-family", type=int, default=10)
    p.add_argument("--out", required=True)
    p = commands.add_parser("demo", help="Synthetic smoke test; no trained model or human calibration")
    p.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "schemas":
            for name, schema in SCHEMAS.items():
                write(Path(args.out) / (name + ".json"), {"$schema": "https://json-schema.org/draft/2020-12/schema", "title": name, **schema})
        elif args.command == "validate":
            validate(load_json(args.file), SCHEMAS[args.record])
            print("valid")
        elif args.command == "attest":
            record = load_json(args.input)
            validate(record, SCHEMAS[args.record])
            write(args.out, seal(args.record, record, read_key(args.key)))
        elif args.command == "grade":
            from .grading import evaluate
            root = Path(args.repo).resolve()
            for name in ("task", "metrics", "experiment", "key", "out", "semantic", "svg"):
                path = getattr(args, name, None)
                if path and Path(path).resolve().is_relative_to(root):
                    raise ValueError(f"{name} must be outside agent repository")
            key = read_key(args.key)
            report, svg = evaluate(load_json(args.task), load_json(args.submission), load_json(args.metrics),
                                   load_json(args.experiment), root, key, args.role,
                                   load_json(args.semantic) if args.semantic else None)
            write(args.out, seal("GradeReport", report, key))
            if args.svg and svg:
                Path(args.svg).write_text(svg)
            print(json.dumps({"tier": report["tier"], "reward": report["reward"], "reasons": report["reasons"]}))
        elif args.command == "scorecard":
            from .reporting import scorecard
            write(args.out, scorecard(read_reports(args.reports, read_key(args.key))))
        elif args.command == "compare":
            from .reporting import compare
            if args.bootstrap_draws < 1000:
                raise ValueError("Use at least 1000 bootstrap draws")
            key = read_key(args.key)
            write(args.out, compare(read_reports(args.baseline, key), read_reports(args.candidate, key), args.bootstrap_draws))
        elif args.command == "calibration":
            from .reporting import calibration
            write(args.out, calibration(load_json(args.reviews)))
        elif args.command == "audit-splits":
            from .dataset import audit_splits
            print(json.dumps(audit_splits(load_json(args.tasks))))
        elif args.command == "freeze":
            from .dataset import freeze
            write(args.out, freeze(load_json(args.config), load_json(args.tasks), load_json(args.baseline)))
        elif args.command == "seed-calibration":
            from .dataset import seed_calibration
            write(args.out, seed_calibration(load_json(args.repositories), args.per_family))
        elif args.command == "demo":
            from .demo import run
            write(args.out, run())
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(f"qa-eval: {exc}", file=sys.stderr)
        return 2
    return 0
