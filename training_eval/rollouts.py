"""Validate and record external harness results; do not implement an environment."""
from dataclasses import asdict
import math

from .contracts import (InfrastructureError, ScoredTrajectory, Verification)


def validate_verification(result):
    if result.status not in {"resolved", "unresolved"}:
        raise ValueError("Invalid verification status")
    if result.status == "resolved":
        if isinstance(result.reward, bool) or not isinstance(result.reward, (float, int)) or not math.isfinite(result.reward):
            raise ValueError("Resolved verification requires a finite reward")
    elif result.reward is not None:
        raise ValueError("Unresolved verification cannot carry a reward")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
           for v in result.components.values()):
        raise ValueError("Verification components must be finite")


def collect(task, policy, request, environment, verifier, tracker, *, role, context_limit, step=0):
    """Return a scored trajectory, or None for a failed harness attempt.

    Only declared InfrastructureError is recoverable. Contract violations fail
    loudly instead of silently turning a broken integration into bad rewards.
    """
    try:
        trajectory = environment.run(task, policy, request)
    except InfrastructureError as exc:
        tracker.log("rollout_failure", {"episode_id": request.episode_id,
                    "task_id": task.task_id, "policy_version": policy.version,
                    "reason": str(exc), "status": "unresolved"}, step=step)
        return None
    if (trajectory.task_id != task.task_id or trajectory.episode_id != request.episode_id
            or trajectory.policy_version != policy.version
            or trajectory.environment_version != environment.version):
        raise ValueError("External trajectory identity does not match the request")
    if len(trajectory.turns) > request.max_turns:
        raise ValueError("External harness exceeded max_turns")
    if not trajectory.turns and trajectory.termination != "infrastructure_error":
        raise ValueError("A completed external trajectory requires generated turns")
    for turn in trajectory.turns:
        if not turn.prompt_tokens or not turn.tokens or len(turn.tokens) > request.max_tokens:
            raise ValueError("Missing prompt or generation budget exceeded")
        if len(turn.prompt_tokens) + len(turn.tokens) > context_limit:
            raise ValueError("Trajectory exceeds model context limit")
        if any(type(t) is not int or t < 0 for t in (*turn.prompt_tokens, *turn.tokens)):
            raise ValueError("Invalid trajectory token IDs")
        if role == "training" or turn.logprobs:
            if len(turn.logprobs) != len(turn.tokens) or any(not math.isfinite(p) or p > 0 for p in turn.logprobs):
                raise ValueError("Missing or invalid behavior log probabilities")
    trajectory_ref = tracker.artifact("trajectory", asdict(trajectory))
    if trajectory.termination == "infrastructure_error":
        result = Verification("unresolved", None, reason="Harness infrastructure failure")
    else:
        try:
            result = verifier.verify(task, trajectory, role=role)
        except InfrastructureError as exc:
            result = Verification("unresolved", None, reason=str(exc))
    validate_verification(result)
    tracker.log("rollout", {"task_id": task.task_id, "episode_id": request.episode_id,
                "group_id": request.group_id, "policy_version": policy.version,
                "verifier_version": verifier.version, "role": role,
                "answer": trajectory.answer, "termination": trajectory.termination,
                "reward": result.reward, "status": result.status,
                "components": dict(result.components), "reason": result.reason,
                "metrics": dict(trajectory.metrics), "trajectory_ref": trajectory_ref}, step=step)
    return ScoredTrajectory(trajectory, result)
