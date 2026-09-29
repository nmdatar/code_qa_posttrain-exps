"""Optional chat-completions-compatible judge adapter (stdlib only).

Usage in judge_command: ["python3", "-m", "qa_eval.http_adapter"]
Routing is host configuration and is NOT supplied to the model.
"""

import json
import os
import sys
import urllib.request
import urllib.error


def main():
    request = json.load(sys.stdin)
    role = request.pop("routing_role", "evaluation")
    if role not in {"training", "evaluation"}:
        raise ValueError("Unknown judge role")
    prefix = "QA_TRAIN_JUDGE_" if role == "training" else "QA_EVAL_JUDGE_"
    base, model = os.environ[prefix + "BASE_URL"], os.environ[prefix + "MODEL"]
    if not base.startswith("https://") and not base.startswith(("http://localhost:", "http://127.0.0.1:")):
        raise ValueError("Use HTTPS for remote judge endpoints")
    key = os.environ.get(prefix + "API_KEY")
    policy = request.pop("policy")
    schema = request.pop("output_schema")
    instructions = request.pop("instructions")
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = "Bearer " + key
    body = {"model": model, "messages": [
        {"role": "system", "content": policy + "\n" + instructions + "\nOutput JSON schema:\n" + json.dumps(schema)},
        {"role": "user", "content": json.dumps(request)}]}
    req = urllib.request.Request(base.rstrip("/") + "/chat/completions", data=json.dumps(body).encode(), headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=float(os.environ.get("QA_JUDGE_TIMEOUT", "120"))) as response:
            result = json.load(response)
        answer = result["choices"][0]["message"]["content"]
        value = json.loads(answer)
        # Parent validates the exact stage schema and evidence/claim coverage.
        json.dump(value, sys.stdout, allow_nan=False)
        return 0
    except Exception:
        # Do not forward endpoint errors, credential-bearing URLs, or bodies.
        print("Judge endpoint failed or returned invalid JSON", file=sys.stderr)
        return 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, ValueError):
        print("Invalid judge routing/configuration", file=sys.stderr)
        raise SystemExit(2)
