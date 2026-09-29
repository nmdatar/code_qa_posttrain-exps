"""Minimal trainer bridge. The evaluator does not itself update model weights."""

import statistics


def prepare_group(reports):
    """Return outcome rewards for a complete same-question GRPO rollout group.

    An unresolved member excludes the WHOLE group so infrastructure/judge failures
    cannot become synthetic negative examples or bias group selection. Trainers
    must log exclusions, retry boundedly, and mask environment observations.
    """
    if not reports:
        raise ValueError("Empty rollout group")
    if any(r["judge_role"] != "training" or r["split"] != "train" for r in reports):
        raise ValueError("Only training-split, training-judge results enter RL")
    for field in ("task_hash", "experiment_hash", "reward_version"):
        if len({r[field] for r in reports}) != 1:
            raise ValueError("Rollout group mixes " + field)
    if any(r["tier"] == "unresolved" or r["reward"] is None for r in reports):
        return {"status": "retry_or_quarantine", "rewards": None, "advantages": None,
                "excluded_episodes": len(reports), "reason": "Incomplete group verification"}
    rewards = [r["reward"] for r in reports]
    center, scale = statistics.fmean(rewards), statistics.pstdev(rewards)
    return {"status": "ready", "rewards": rewards,
            "advantages": [(r - center) / scale for r in rewards] if scale else [0.0] * len(rewards),
            "excluded_episodes": 0, "zero_variance": scale == 0}
