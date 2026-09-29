"""Standalone and periodic evaluation of immutable policies via external adapters."""
from dataclasses import asdict
import statistics
import uuid
import math

from .contracts import RolloutRequest
from .rollouts import collect


class Evaluator:
    def __init__(self, tasks, environment, verifier, tracker, *, run_id="evaluation",
                 max_tokens=64, max_turns=4, temperature=1.0, seed=0, context_limit=4096):
        if any(type(v) is not int or v <= 0 for v in (max_tokens, max_turns, context_limit)):
            raise ValueError("Evaluation budgets/context limit must be positive integers")
        if not math.isfinite(temperature) or temperature < 0:
            raise ValueError("Invalid evaluation temperature")
        self.tasks, self.environment, self.verifier, self.tracker = tasks, environment, verifier, tracker
        self.run_id, self.max_tokens, self.max_turns = run_id, max_tokens, max_turns
        self.temperature, self.seed, self.context_limit = temperature, seed, context_limit

    def evaluate(self, policy, *, checkpoint=None, step=0, final_test=False):
        allowed = {"final_test", "test", "pilot_test"} if final_test else {"dev", "development"}
        if not len(self.tasks):
            raise ValueError("Evaluation requires at least one task")
        if any(self.tasks.get(i).split not in allowed for i in range(len(self.tasks))):
            raise ValueError("Evaluation split not allowed (final test must be explicitly requested)")
        evaluation_id = uuid.uuid4().hex
        rows = []
        for i in range(len(self.tasks)):
            task = self.tasks.get(i)
            request = RolloutRequest(self.run_id, -1, evaluation_id,
                                     f"{evaluation_id}-{i}", self.seed + i,
                                     self.max_tokens, self.max_turns, self.temperature)
            scored = collect(task, policy, request, self.environment, self.verifier,
                             self.tracker, role="evaluation", context_limit=self.context_limit, step=step)
            rows.append({"task_id": task.task_id, "split": task.split,
                         "verification": asdict(scored.verification) if scored else
                         {"status": "unresolved", "reward": None, "components": {},
                          "reason": "Harness infrastructure failure"}})
        rewards = [r["verification"]["reward"] for r in rows
                   if r["verification"]["status"] == "resolved"]
        report = {"evaluation_id": evaluation_id, "checkpoint": checkpoint,
                  "policy_version": policy.version, "dataset": self.tasks.identity,
                  "environment_version": self.environment.version,
                  "verifier_version": self.verifier.version,
                  "settings": {"max_tokens": self.max_tokens, "max_turns": self.max_turns,
                               "temperature": self.temperature, "seed": self.seed},
                  "expected": len(self.tasks), "attempted": len(rows),
                  "resolved": len(rewards), "unresolved": len(rows) - len(rewards),
                  "coverage": len(rewards) / len(rows),
                  "mean_reward": statistics.fmean(rewards) if rewards else None,
                  "complete": len(rewards) == len(rows), "rows": rows}
        report["artifact"] = self.tracker.artifact("evaluation", report)
        self.tracker.log("evaluation", report, step=step)
        self.tracker.log("metrics", {"eval/coverage": report["coverage"],
                         "eval/resolved": len(rewards), "eval/unresolved": report["unresolved"],
                         **({"eval/reward": report["mean_reward"]} if rewards else {})}, step=step)
        return report
