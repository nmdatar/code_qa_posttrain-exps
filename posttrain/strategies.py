"""Training input validation independent of backend tensor representation."""
import math
import statistics


def prepare_group(trajectories, results):
    if len(trajectories) < 2 or len(trajectories) != len(results): raise ValueError('Incomplete rollout group')
    for key in ('task_id','policy_id'):
        if len({t.get(key) for t in trajectories}) != 1 or trajectories[0].get(key) is None:
            raise ValueError('Group mixes or lacks ' + key)
    if any(r.get('status') != 'resolved' or r.get('reward') is None for r in results):
        return {'status':'retry_or_quarantine','advantages':None,'rewards':None}
    rewards = [r['reward'] for r in results]
    if any(not isinstance(r,(int,float)) or not math.isfinite(r) for r in rewards): raise ValueError('Nonfinite reward')
    mean, std = statistics.fmean(rewards), statistics.pstdev(rewards)
    return {'status':'ready','rewards':rewards,'advantages':[(r-mean)/std if std else 0.0 for r in rewards],
            'zero_variance':std == 0}


def validate_actions(trajectory):
    actions = trajectory.get('actions', [])
    if not actions: raise ValueError('No generated actions')
    for action in actions:
        ids = action['token_ids']; probs = action['logprobs']
        if not ids or len(ids) != len(probs): raise ValueError('Token/log-probability alignment mismatch')
        if any(type(t) is not int or t < 0 for t in ids): raise ValueError('Invalid token IDs')
        if any(not math.isfinite(p) or p > 1e-6 for p in probs): raise ValueError('Invalid behavior log probabilities')
        if not action.get('prompt_tokens'): raise ValueError('Missing actual conditioning tokens')
        if action.get('policy_id',trajectory['policy_id']) != trajectory['policy_id']: raise ValueError('Mixed action policy')
    return True


def validate_sft(example):
    if example.get('split') != 'train' or example.get('status') != 'accepted':
        raise ValueError('SFT needs accepted training examples')
    if not example.get('provenance'): raise ValueError('SFT example lacks provenance')
    messages = example.get('messages', [])
    if not messages or not any(m.get('role') == 'assistant' for m in messages): raise ValueError('No assistant supervision')
    if any(m.get('role') not in {'system','user','assistant','tool'} for m in messages): raise ValueError('Invalid conversation role')
    return True
