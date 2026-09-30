"""Human calibration packets: immutable examples, separate reviewer decisions."""
from pathlib import Path
import hashlib
import html
import json
from qa_eval.reporting import calibration

LABELS = {"accepted", "partial", "material_error"}

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

def export_packet(examples, out):
    """Examples must already have actual judge predictions; never invent them."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    ids = set()
    for example in examples:
        row = dict(example)
        for key in ("task_id", "family_id", "predicted_tier", "question", "answer", "origin"):
            if not row.get(key):
                raise ValueError(f"Missing calibration {key}")
        if row["origin"] not in {"constructed", "model_output"}:
            raise ValueError("Example origin must identify constructed or actual model output")
        if row.get("reviews"):
            raise ValueError("Export new packets without prefilled human reviews")
        row["reviews"] = []
        row["example_id"] = row.get("example_id", digest(row)[:24])
        if row["example_id"] in ids:
            raise ValueError("Duplicate calibration example")
        ids.add(row["example_id"])
        row["example_sha256"] = digest({k: v for k, v in row.items() if k not in {"reviews", "example_sha256"}})
        rows.append(row)
    calibration(rows)  # Existing contract validation and unchanged gate.
    packet = {"schema_version": "1.0", "examples": rows, "packet_sha256": digest(rows)}
    path = out / "packet.json"
    if path.exists() and json.loads(path.read_text()) != packet:
        raise ValueError("Refuse overwriting a different calibration packet")
    path.write_text(json.dumps(packet, indent=2) + "\n")
    sections = []
    for row in rows:
        esc = lambda x: html.escape(str(x))
        sections.append(f'<article><h2>{esc(row["example_id"])}</h2><p>{esc(row["origin"])}</p><h3>Question</h3><pre>{esc(row["question"])}</pre><h3>Answer</h3><pre>{esc(row["answer"])}</pre><details><summary>Private grading evidence</summary><pre>{esc(json.dumps(row.get("evidence", {}), indent=2))}</pre></details></article>')
    # Hide predictions in HTML so reviewers assess answers independently.
    (out / "review.html").write_text('<!doctype html><meta charset="utf-8"><title>Private calibration review</title><style>body{max-width:1000px;margin:40px auto;font:16px system-ui}pre{white-space:pre-wrap}article{border-bottom:1px solid #aaa;padding:20px}</style><h1>Independent human review</h1><p>Two distinct humans review each example independently. Record reviewer identity, label, rationale, and example hash in decisions.json. Labels: accepted, partial, material_error. Do not view packet predictions before reviewing.</p>' + ''.join(sections))
    template = {"packet_sha256": packet["packet_sha256"], "decisions": [{"example_id": x["example_id"], "example_sha256": x["example_sha256"], "reviewer": "", "reviewer_type": "human", "label": "", "rationale": ""} for x in rows]}
    (out / "decisions.template.json").write_text(json.dumps(template, indent=2) + "\n")
    return {"packet": str(path), "review_html": str(out / "review.html"), "examples": len(rows), "gate": calibration(rows)}

def import_reviews(packet, decisions, out=None):
    """Merge human decisions only; disagreements stay pending under original gate."""
    packet = json.loads(Path(packet).read_text()) if isinstance(packet, (str, Path)) else packet
    decisions = json.loads(Path(decisions).read_text()) if isinstance(decisions, (str, Path)) else decisions
    rows = json.loads(json.dumps(packet["examples"]))
    if digest(rows) != packet["packet_sha256"] or decisions.get("packet_sha256") != packet["packet_sha256"]:
        raise ValueError("Calibration packet binding mismatch")
    if out is not None and Path(out).exists():
        previous = json.loads(Path(out).read_text())
        if previous.get("packet_sha256") != packet["packet_sha256"]:
            raise ValueError("Existing reviews belong to another packet")
        prior = {r["example_id"]: r for r in previous["rows"]}
        for row in rows:
            old = prior.get(row["example_id"])
            if old is None or old["example_sha256"] != row["example_sha256"]:
                raise ValueError("Existing review example binding mismatch")
            row["reviews"] = old["reviews"]
    index = {r["example_id"]: r for r in rows}
    for decision in decisions["decisions"]:
        row = index.get(decision["example_id"])
        if row is None or decision.get("example_sha256") != row["example_sha256"]:
            raise ValueError("Unknown or changed calibration example")
        if decision.get("reviewer_type") != "human" or not decision.get("reviewer", "").strip() or not decision.get("rationale", "").strip():
            raise ValueError("Real human reviewer identity and rationale required")
        if decision.get("label") not in LABELS:
            raise ValueError("Invalid human label")
        normalized = {"reviewer": decision["reviewer"].strip().casefold(), "label": decision["label"],
                      "rationale": decision["rationale"].strip(), "reviewer_type": "human"}
        previous = next((x for x in row["reviews"] if x["reviewer"].strip().casefold() == normalized["reviewer"]), None)
        if previous is not None:
            comparable = {**previous, "reviewer": previous["reviewer"].strip().casefold(),
                          "rationale": previous["rationale"].strip()}
            if comparable == normalized:
                continue
            raise ValueError("Conflicting decision from an existing reviewer")
        if len(row["reviews"]) >= 2:
            raise ValueError("Exactly two independent reviews; retain disputes for adjudication")
        row["reviews"].append(normalized)
    result = {"packet_sha256": packet["packet_sha256"], "rows": rows, "report": calibration(rows),
              "requirements": {"distinct_reviewers_per_example": 2, "minimum_tasks": 100, "minimum_families": 10,
                               "confidence_bounds": "Unchanged qa_eval.reporting.calibration Wilson gates"}}
    if out is not None:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(result, indent=2) + "\n")
    return result
