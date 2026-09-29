"""Trusted host-side telemetry. Import this in the runner, never in agent tools."""

import time
from .security import bindings, seal


class EpisodeRecorder:
    def __init__(self, task, experiment_id, episode_id, trajectory_ref):
        self.task = task
        self.start = time.monotonic()
        self.first_token = None
        self.closed = False
        self.record = {"schema_version": "1.0", "experiment_id": experiment_id,
                       "episode_id": episode_id, "trajectory_ref": trajectory_ref,
                       "input_tokens": 0, "output_tokens": 0, "tool_seconds": 0.0,
                       "tool_calls": [], "retries": 0, "cost": 0.0,
                       "integrity_violation": False, "execution_records": [], "probes": []}

    def usage(self, input_tokens, output_tokens, cost=0.0):
        if self.closed or min(input_tokens, output_tokens) < 0 or (cost is not None and cost < 0):
            raise ValueError("Invalid usage or closed episode")
        self.record["input_tokens"] += input_tokens
        self.record["output_tokens"] += output_tokens
        self.record["cost"] = (self.record["cost"] + cost
                               if self.record["cost"] is not None and cost is not None else None)

    def token_received(self):
        if self.first_token is None:
            self.first_token = time.monotonic() - self.start

    def tool(self, name, operation):
        if self.closed:
            raise ValueError("Closed episode")
        if name not in self.task["permitted_tools"]:
            self.record["integrity_violation"] = True
            raise ValueError("Unauthorized tool")
        if len(self.record["tool_calls"]) >= self.task["budgets"]["max_tool_calls"]:
            raise ValueError("Tool budget exhausted")
        self.record["tool_calls"].append(name)
        started = time.monotonic()
        try:
            return operation()
        finally:
            self.record["tool_seconds"] += time.monotonic() - started

    def retry(self):
        self.record["retries"] += 1

    def finish(self, submission, key, termination_reason="completed"):
        if self.closed:
            raise ValueError("Episode already finished")
        self.closed = True
        payload = {**self.record, **bindings(self.task, submission),
                   "latency_seconds": time.monotonic() - self.start,
                   "time_to_first_token_seconds": self.first_token or 0.0,
                   "termination_reason": termination_reason}
        from .schema import METRICS, validate
        validate(payload, METRICS)
        return seal("EpisodeMetrics", payload, key)
