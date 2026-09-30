"""Strict JSON contracts, also exported as draft-2020-12 JSON Schemas.

Only the vocabulary used below is needed by the dependency-free validator.
All objects reject unknown fields to prevent silently ignored policy settings.
"""

import math


def obj(properties, optional=()):
    return {"type": "object", "properties": properties,
            "required": [k for k in properties if k not in optional],
            "additionalProperties": False}


def arr(items, minimum=0):
    return {"type": "array", "items": items, "minItems": minimum}


def enum(*values):
    return {"enum": list(values)}


TEXT = {"type": "string"}
NONEMPTY = {"type": "string", "minLength": 1}
BOOL = {"type": "boolean"}
NUMBER = {"type": "number", "minimum": 0}
POSITIVE = {"type": "number", "exclusiveMinimum": 0}
INT = {"type": "integer", "minimum": 0}
POSINT = {"type": "integer", "minimum": 1}
UNIT = {"type": "number", "minimum": 0, "maximum": 1}
HASH = {"type": "string", "pattern": "^[a-f0-9]{64}$"}
VERSION = enum("1.0")
EVIDENCE = obj({"path": NONEMPTY, "start_line": POSINT, "end_line": POSINT,
                "file_sha256": HASH, "symbol": NONEMPTY}, optional=("symbol",))
CITATION = obj({"id": NONEMPTY, **EVIDENCE["properties"]}, optional=("symbol",))
GRAPH = obj({
    "nodes": arr(obj({"id": NONEMPTY, "entity": NONEMPTY, "label": NONEMPTY,
                      "citations": arr(NONEMPTY)}), 1),
    "edges": arr(obj({"source": NONEMPTY, "target": NONEMPTY,
                      "relation": enum("calls", "contains", "inherits", "data_flow", "precedes", "signals"),
                      "condition": TEXT, "citations": arr(NONEMPTY)}))})

TASK = obj({
    "schema_version": VERSION, "id": NONEMPTY, "question": NONEMPTY,
    "repository": obj({"family_id": NONEMPTY, "commit": {"type": "string", "pattern": "^[a-f0-9]{40,64}$"},
                       "url": NONEMPTY}),
    "lineage_id": NONEMPTY, "split": enum("train", "development", "final_test"),
    "category": NONEMPTY,
    "answerability": enum("answerable", "unanswerable", "needs_clarification"),
    "claims": arr(obj({"id": NONEMPTY, "text": NONEMPTY, "weight": enum(1, 3),
                       "separable_subparts": arr(NONEMPTY), "evidence": arr(EVIDENCE)}), 1),
    "critical_errors": arr(NONEMPTY), "permitted_tools": arr(NONEMPTY),
    "budgets": obj({"latency_seconds": POSITIVE, "compute_units": POSITIVE,
                    "max_tool_calls": INT, "max_output_tokens": POSINT,
                    "max_submission_bytes": POSINT}),
    "diagram": obj({"required": BOOL, "criteria": arr(NONEMPTY), "allowed_abstractions": arr(NONEMPTY)}),
    "probes": arr(obj({"id": NONEMPTY, "fixture_id": NONEMPTY, "expected_stdout_sha256": HASH})),
    "human_reviewed": BOOL, "gold_status": enum("accepted", "draft", "ambiguous")})

SUBMISSION = obj({"schema_version": VERSION, "task_id": NONEMPTY,
                  "text": TEXT, "citations": arr(CITATION),
                  "diagram": {"oneOf": [{"type": "null"}, GRAPH,
                                        obj({"mermaid": NONEMPTY})]}})

METRICS = obj({
    "schema_version": VERSION, "task_hash": HASH, "submission_hash": HASH,
    "experiment_id": NONEMPTY, "episode_id": NONEMPTY, "trajectory_ref": NONEMPTY,
    "latency_seconds": NUMBER, "time_to_first_token_seconds": {"oneOf": [NUMBER, {"type": "null"}]},
    "input_tokens": INT, "output_tokens": INT, "tool_seconds": NUMBER,
    "tool_calls": arr(NONEMPTY), "retries": INT, "cost": {"oneOf": [NUMBER, {"type": "null"}]},
    "termination_reason": enum("completed", "budget_exhausted", "agent_error", "infrastructure_error"),
    "integrity_violation": BOOL,
    "execution_records": arr(obj({"id": NONEMPTY, "command": NONEMPTY,
                                  "exit_code": {"type": "integer"}, "stdout_sha256": HASH})),
    "probes": arr(obj({"id": NONEMPTY, "fixture_id": NONEMPTY, "commit": NONEMPTY,
                       "isolated": BOOL, "stdout_sha256": HASH,
                       "status": enum("completed", "infrastructure_error")}))})

EXTRACTION = obj({
    "claims": arr(obj({"id": NONEMPTY, "text": NONEMPTY, "source": enum("text", "diagram"),
                       "evidence_requests": arr(EVIDENCE)})),
    "answer_mode": enum("answer", "abstention", "clarification"),
    "extraction_complete": BOOL})

CLAIM_FINDING = obj({"id": NONEMPTY, "coverage": enum("absent", "partial", "complete"),
                     "verdict": enum("supported", "contradicted", "insufficient"),
                     "material_error": BOOL, "reason": NONEMPTY,
                     "evidence_keys": arr(NONEMPTY)})
SEMANTIC = obj({
    "schema_version": VERSION, "task_hash": HASH, "submission_hash": HASH,
    "experiment_hash": HASH, "judge_family": NONEMPTY, "judge_version": NONEMPTY,
    "prompt_version": {"enum": ["1.0", "1.1", "1.2"]}, "snapshot_fingerprint": HASH,
    "required_claims": arr(CLAIM_FINDING), "additional_claims": arr(CLAIM_FINDING),
    "extracted_claims": arr(obj({"id": NONEMPTY, "text": NONEMPTY, "source": enum("text", "diagram")})),
    "citation_links": arr(obj({"claim_id": NONEMPTY, "citation_id": NONEMPTY,
                               "supported": BOOL, "reason": NONEMPTY})),
    "uncited_claim_ids": arr(NONEMPTY), "critical_error": BOOL,
    "false_execution_claim": BOOL, "answer_mode": enum("answer", "abstention", "clarification"),
    "answerability_satisfied": BOOL,
    "diagram": obj({"assessed": BOOL, "supported": BOOL, "complete": BOOL,
                    "consistent": BOOL, "readable": BOOL, "material_error": BOOL,
                    "findings": arr(NONEMPTY)}),
    "assessment_complete": BOOL, "needs_review": BOOL,
    "evidence_keys": arr(NONEMPTY), "disagreements": arr(NONEMPTY)})

ASSESSMENT = obj({k: v for k, v in SEMANTIC["properties"].items() if k not in {
    "schema_version", "task_hash", "submission_hash", "experiment_hash", "judge_family",
    "judge_version", "prompt_version", "snapshot_fingerprint", "extracted_claims", "evidence_keys"}})

EXPERIMENT = obj({
    "schema_version": VERSION, "id": NONEMPTY, "reward_version": VERSION,
    "frozen": BOOL, "latency_enabled": BOOL, "environment_id": NONEMPTY,
    "accounting": obj({"input_token": NUMBER, "output_token": POSITIVE, "tool_second": NUMBER}),
    "training_judge_family": NONEMPTY, "evaluation_judge_family": NONEMPTY,
    "judge_version": NONEMPTY, "judge_command": arr(NONEMPTY, 1),
    "judge_timeout_seconds": POSITIVE, "max_evidence_bytes": POSINT,
    "candidate_preselected": BOOL, "baseline_manifest_hash": HASH,
    "budgets_manifest_hash": HASH,
    "task_manifest": arr(obj({"id": NONEMPTY, "task_hash": HASH}), 1)})

CHECK = obj({"name": NONEMPTY, "status": enum("pass", "defect", "failure", "unresolved"),
             "detail": NONEMPTY})
REPORT = obj({
    "schema_version": VERSION, "reward_version": VERSION,
    "task_id": NONEMPTY, "task_hash": HASH, "submission_hash": HASH,
    "family_id": NONEMPTY, "lineage_id": NONEMPTY, "split": NONEMPTY, "category": NONEMPTY,
    "answerability": NONEMPTY, "tier": enum("failed", "partial", "accepted", "unresolved"),
    "reward": {"oneOf": [{"type": "null"}, UNIT]}, "coverage": UNIT,
    "efficiency": {"oneOf": [{"type": "null"}, UNIT]},
    "checks": arr(CHECK), "reasons": arr(NONEMPTY),
    "semantic": {"oneOf": [{"type": "null"}, SEMANTIC]},
    "metrics": {"oneOf": [{"type": "null"}, METRICS]},
    "experiment_hash": HASH, "environment_id": NONEMPTY, "latency_enabled": BOOL,
    "scoring_contract_hash": HASH,
    "planned_task_hashes": arr(HASH, 1),
    "compute_units": {"oneOf": [{"type": "null"}, NUMBER]},
    "diagram_present": BOOL, "material_error": BOOL,
    "judge_role": enum("training", "evaluation"),
    "candidate_preselected": BOOL})

SCHEMAS = {"TaskSpec": TASK, "AnswerSubmission": SUBMISSION, "EpisodeMetrics": METRICS,
           "GradeReport": REPORT, "SemanticAssessment": SEMANTIC, "ExperimentConfig": EXPERIMENT,
           "CanonicalGraph": GRAPH, "ClaimExtraction": EXTRACTION, "JudgeOutput": ASSESSMENT}


def validate(value, schema, path="$"):
    import re
    if "oneOf" in schema:
        matches = 0
        for option in schema["oneOf"]:
            try:
                validate(value, option, path)
                matches += 1
            except ValueError:
                pass
        if matches != 1:
            raise ValueError(f"{path}: must match exactly one allowed shape")
        return
    if "enum" in schema:
        if not any(type(value) == type(v) and value == v for v in schema["enum"]):
            raise ValueError(f"{path}: invalid enum value")
    kind = schema.get("type")
    expected = {"object": dict, "array": list, "string": str, "boolean": bool,
                "integer": int, "number": (int, float), "null": type(None)}
    if kind and (not isinstance(value, expected[kind]) or
                 (kind in {"number", "integer"} and isinstance(value, bool))):
        raise ValueError(f"{path}: expected {kind}")
    if kind in {"number", "integer"}:
        if not math.isfinite(value):
            raise ValueError(f"{path}: nonfinite number")
        for constraint, failed in [("minimum", lambda x: value < x), ("maximum", lambda x: value > x),
                                   ("exclusiveMinimum", lambda x: value <= x)]:
            if constraint in schema and failed(schema[constraint]):
                raise ValueError(f"{path}: violates {constraint}")
    if kind == "string":
        if len(value) < schema.get("minLength", 0) or ("pattern" in schema and not re.search(schema["pattern"], value)):
            raise ValueError(f"{path}: invalid string")
    if kind == "array":
        if len(value) < schema.get("minItems", 0):
            raise ValueError(f"{path}: too few items")
        for i, item in enumerate(value):
            validate(item, schema["items"], f"{path}[{i}]")
    if kind == "object":
        missing = set(schema["required"]) - value.keys()
        extra = value.keys() - schema["properties"].keys()
        if missing or extra:
            raise ValueError(f"{path}: missing={sorted(missing)}, extra={sorted(extra)}")
        for k, item in value.items():
            validate(item, schema["properties"][k], f"{path}.{k}")


def unique_ids(items, name):
    ids = [x["id"] for x in items]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate {name} IDs")


def validate_task(task):
    validate(task, TASK)
    unique_ids(task["claims"], "claim")
    unique_ids(task["probes"], "probe")
    texts = [" ".join(c["text"].casefold().split()) for c in task["claims"]]
    if len(texts) != len(set(texts)):
        raise ValueError("Repeated rubric claim")
    if task["answerability"] == "answerable" and any(not c["evidence"] for c in task["claims"]):
        raise ValueError("Answerable tasks require evidence for every rubric claim")
    if task["diagram"]["required"] and not task["diagram"]["criteria"]:
        raise ValueError("Required diagrams need semantic criteria")
