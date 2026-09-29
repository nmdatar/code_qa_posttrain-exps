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
    stats['contributing_trajectories'] = len(contributing)
    stats['reward_variance'] = statistics.pvariance(stats['rewards']) if stats['rewards'] else None
    return rows, stats
