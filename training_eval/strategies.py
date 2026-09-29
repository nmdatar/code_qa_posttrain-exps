"""Backend-neutral SFT and outcome-reward GRPO batch construction."""
from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Sequence

from .contracts import (CapabilityError, ScoredTrajectory, SFTExample,
                        TrainingRow, TrainingStrategy, UpdateBatch)
from .data import validate_sft


def _context_limit(value: int) -> None:
    if type(value) is not int or value < 1:
        raise ValueError("context_limit must be a positive integer")


class _StatelessStrategy:
    def state_dict(self) -> Mapping[str, Any]:
        return {}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state:
            raise ValueError("This strategy has no resumable client state")


class SFTStrategy(_StatelessStrategy):
    name = "sft"
    input_kind = "sft"
    requirements = frozenset({"cross_entropy"})

    def configuration(self) -> Mapping[str, Any]:
        """Immutable algorithm settings; subclasses must include any new settings."""
        return {"version": 1, "objective": "cross_entropy",
                "reduction": "mean_supervised_tokens"}

    def build_batch(self, items: Sequence[SFTExample], *, context_limit: int) -> UpdateBatch | None:
        _context_limit(context_limit)
        for example in items:
            validate_sft(example)
            if len(example.tokens) > context_limit:
                raise ValueError("SFT example exceeds model context limit; truncation is not implicit")
        selected = sum(sum(e.loss_mask[1:]) for e in items)
        if not selected:
            return None
        rows = tuple(TrainingRow(e.tokens[:-1], e.tokens[1:],
                                 tuple(w / selected for w in e.loss_mask[1:]))
                     for e in items if any(e.loss_mask[1:]))
        return UpdateBatch("cross_entropy", rows)


class GRPOStrategy(_StatelessStrategy):
    """Group-relative, population-standardized rewards with importance sampling.

    No implicit KL penalty, clipping, or critic. Each contributing trajectory has
    equal loss mass; generated tokens split its mass equally across all turns.
    The backend sums `advantages` directly and must not apply weights again.
    """
    name = "grpo"
    input_kind = "rollouts"
    requirements = frozenset({"importance_sampling"})

    def configuration(self) -> Mapping[str, Any]:
        """All objective settings are bound to checkpoint resume compatibility."""
        return {"version": 1, "objective": "importance_sampling",
                "reduction": "mean_trajectory_then_mean_generated_tokens",
                "advantage_normalization": "group_population_standard_deviation",
                "zero_variance": "skip_group", "kl_coefficient": 0.0,
                "clipping": None}

    def build_batch(self, items: Sequence[Sequence[ScoredTrajectory]], *, context_limit: int) -> UpdateBatch | None:
        _context_limit(context_limit)
        contributing: list[tuple[ScoredTrajectory, float]] = []
        seen_episodes: set[str] = set()
        policy_versions: set[str] = set()
        for group in items:
            if len(group) < 2:
                raise ValueError("GRPO requires complete groups with at least two episodes")
            identities = {(s.trajectory.task_id, s.trajectory.policy_version,
                           s.trajectory.environment_version) for s in group}
            if len(identities) != 1 or any(not part for part in next(iter(identities))):
                raise ValueError("Group must share nonempty task, policy, and environment identities")
            rewards = []
            for scored in group:
                trajectory, verification = scored.trajectory, scored.verification
                if not trajectory.episode_id or trajectory.episode_id in seen_episodes:
                    raise ValueError("Episodes must have unique nonempty ids")
                seen_episodes.add(trajectory.episode_id)
                policy_versions.add(trajectory.policy_version)
                if verification.status != "resolved" or verification.reward is None:
                    raise ValueError("Unresolved verifier outcomes cannot train")
                if not math.isfinite(verification.reward) or any(not math.isfinite(v) for v in verification.components.values()):
                    raise ValueError("Verifier rewards and components must be finite")
                rewards.append(verification.reward)
                if not trajectory.turns or not sum(len(t.tokens) for t in trajectory.turns):
                    raise ValueError("A training trajectory must contain generated tokens")
                for turn in trajectory.turns:
                    if not turn.prompt_tokens or not turn.tokens:
                        raise ValueError("Each generation requires nonempty prompt and generated tokens")
                    if len(turn.logprobs) != len(turn.tokens) or any(not math.isfinite(p) or p > 0 for p in turn.logprobs):
                        raise ValueError("Generated tokens require aligned finite nonpositive log probabilities")
                    if any(type(t) is not int or t < 0 for t in turn.prompt_tokens + turn.tokens):
                        raise ValueError("Token ids must be nonnegative integers")
                    if len(turn.prompt_tokens) + len(turn.tokens) > context_limit:
                        raise ValueError("Generation exceeds model context limit")
            # Scaling avoids overflow when users supply large finite rewards.
            scale = max(abs(r) for r in rewards)
            normalized = [r / scale for r in rewards] if scale else rewards
            mean = math.fsum(normalized) / len(normalized)
            variance = math.fsum((r - mean) ** 2 for r in normalized) / len(normalized)
            if variance == 0:
                continue
            deviation = math.sqrt(variance)
            contributing.extend((s, (r - mean) / deviation) for s, r in zip(group, normalized))
        if len(policy_versions) > 1:
            raise ValueError("An update batch must contain one rollout policy version")
        if not contributing:
            return None
        rows = []
        for scored, advantage in contributing:
            turns = scored.trajectory.turns
            weight = 1 / (len(contributing) * sum(len(t.tokens) for t in turns))
            for turn in turns:
                tokens = turn.prompt_tokens + turn.tokens
                context = len(turn.prompt_tokens) - 1
                generated = len(turn.tokens)
                rows.append(TrainingRow(tokens[:-1], tokens[1:],
                                        (0.0,) * context + (weight,) * generated,
                                        (0.0,) * context + turn.logprobs,
                                        (0.0,) * context + (advantage * weight,) * generated))
        return UpdateBatch("importance_sampling", tuple(rows))


class StrategyRegistry:
    def __init__(self, strategies: Iterable[TrainingStrategy] | None = None):
        self._strategies: dict[str, TrainingStrategy] = {}
        for strategy in (SFTStrategy(), GRPOStrategy()) if strategies is None else strategies:
            self.register(strategy)

    def register(self, strategy: TrainingStrategy) -> None:
        if not strategy.name or strategy.name in self._strategies:
            raise ValueError(f"Duplicate or empty strategy name: {strategy.name!r}")
        self._strategies[strategy.name] = strategy

    def get(self, name: str, capabilities: frozenset[str] | None = None) -> TrainingStrategy:
        if name not in self._strategies:
            raise CapabilityError(f"Unsupported strategy {name!r}; registered: {', '.join(self._strategies)}")
        strategy = self._strategies[name]
        if capabilities is not None and (missing := strategy.requirements - capabilities):
            raise CapabilityError(f"Strategy {name!r} requires backend capabilities: {sorted(missing)}")
        return strategy
