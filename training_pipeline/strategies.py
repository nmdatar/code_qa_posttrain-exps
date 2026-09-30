"""GRPO semantics stay here; Tinker computes gradients and updates weights."""
import statistics
from .rendering import shifted


def group_advantages(group):
    if len(group) < 2:
        raise ValueError('GRPO requires at least two attempts')
    for name in ('task_id', 'task_hash', 'policy_id', 'group_id', 'environment_id', 'experiment_hash', 'split'):
        if len({getattr(t, name) for t in group}) != 1:
            raise ValueError('Group mixes ' + name)
    if group[0].split != 'train':
        raise ValueError('Held-out tasks cannot train')
    if len({t.episode_id for t in group}) != len(group):
        raise ValueError('Duplicate episode in group')
    for t in group:
        if t.verification is None:
            return None
        t.verification.validate()
        for g in t.generations:
            g.validate()
            if g.policy_id != t.policy_id:
                raise ValueError('Mixed generation policy')
    if len({t.verification.version for t in group}) != 1:
        raise ValueError('Mixed reward versions')
    if any(t.verification.status == 'unresolved' for t in group):
        return None
    rewards = [t.verification.reward for t in group]
    center, scale = statistics.fmean(rewards), statistics.pstdev(rewards)
    return [(r-center)/scale for r in rewards] if scale else [0.0] * len(group)


def grpo_batch(groups):
    contributing = []
    stats = {'excluded_groups': 0, 'zero_variance_groups': 0, 'rewards': [], 'advantages': []}
    policies = {t.policy_id for group in groups for t in group}
    if len(policies) != 1:
        raise ValueError('Batch must use one behavior policy')
    for group in groups:
        advantages = group_advantages(group)
        if advantages is None:
            stats['excluded_groups'] += 1
            continue
        stats['rewards'].extend(t.verification.reward for t in group)
        stats['advantages'].extend(advantages)
        if not any(advantages):
            stats['zero_variance_groups'] += 1
            continue
        for t, a in zip(group, advantages):
            if a:
                if not t.generations:
                    raise ValueError('Resolved trajectory has no actions')
                contributing.append((t, a))
    rows = []
    for trajectory, advantage in contributing:
        tokens = sum(len(g.tokens) for g in trajectory.generations)
        scale = advantage / (tokens * len(contributing))
        for g in trajectory.generations:
            rows.append(shifted(g.prompt, g.tokens, [scale] * len(g.tokens), g.logprobs))
    # Report all resolved attempts, including those in otherwise excluded groups.
    attempts = [t for group in groups for t in group]
    resolved = [t.verification.reward for t in attempts
                if t.verification is not None and t.verification.status == 'resolved']
    stats.update(mean_reward=statistics.fmean(resolved) if resolved else None,
                 eligible_mean_reward=statistics.fmean(stats['rewards']) if stats['rewards'] else None,
                 scoring_coverage=len(resolved)/len(attempts) if attempts else 0.0,
                 attempted_trajectories=len(attempts), resolved_trajectories=len(resolved),
                 excluded_group_fraction=stats['excluded_groups']/len(groups) if groups else 0.0)
    stats['contributing_trajectories'] = len(contributing)
    stats['reward_variance'] = statistics.pvariance(stats['rewards']) if stats['rewards'] else None
    return rows, stats


def reinforce_batch(groups, baseline_state=None):
    """Prior-batch running mean baseline; never center within the current group.

    Preserve GRPO's complete-group admission and per-trajectory token averaging.
    The baseline starts at zero and is updated only after advantages are fixed.
    Its sum/count are checkpointed by the orchestrator for exact resume.
    """
    import math
    state = dict(baseline_state or {'reward_sum': 0.0, 'count': 0})
    if (set(state) != {'reward_sum', 'count'} or type(state['count']) is not int
            or state['count'] < 0 or not math.isfinite(state['reward_sum'])
            or (state['count'] == 0 and state['reward_sum'] != 0)):
        raise ValueError('Invalid running baseline state')
    baseline = state['reward_sum'] / state['count'] if state['count'] else 0.0
    _, stats = grpo_batch(groups)  # Identical validation and admission contract.
    eligible = [t for group in groups if group_advantages(group) is not None for t in group]
    advantages = [t.verification.reward - baseline for t in eligible]
    contributing = [(t, a) for t, a in zip(eligible, advantages) if a != 0]
    rows = []
    for t, advantage in contributing:
        if not t.generations:
            raise ValueError('Resolved trajectory has no actions')
        tokens = sum(len(g.tokens) for g in t.generations)
        scale = advantage / (tokens * len(contributing))
        for g in t.generations:
            rows.append(shifted(g.prompt, g.tokens, [scale] * len(g.tokens), g.logprobs))
    state['reward_sum'] += sum(t.verification.reward for t in eligible)
    state['count'] += len(eligible)
    stats.update(advantages=advantages, contributing_trajectories=len(contributing),
                 advantage_baseline=baseline, baseline_count_before=state['count']-len(eligible),
                 baseline_count_after=state['count'], algorithm='reinforce-prior-mean-v1')
    return rows, stats, state
