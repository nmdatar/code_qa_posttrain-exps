"""Quality and all-attempt cost telemetry, independent of shaped reward."""
import math


def summarize(trajectories):
    trajectories = list(trajectories)
    n = len(trajectories)
    if not n:
        return {}
    correctness_rows = [t for t in trajectories if t.verification and t.verification.diagnostics.get('reward_applied') == 'correctness']
    accepted = [t for t in trajectories if t.verification is not None
                and t.verification.status == 'resolved'
                and t.verification.diagnostics.get('tier') == 'accepted']
    result = {'accepted_answers': len(accepted), 'accepted_rate': len(accepted)/n}
    if correctness_rows:
        for name in ('correctness', 'citation'):
            values = [t.verification.diagnostics.get(name + '_score') for t in correctness_rows]
            valid = [v for v in values if v is not None]
            result['mean_' + name + '_score'] = sum(valid)/len(valid) if valid else None
            result[name + '_measurement_coverage'] = len(valid)/n
        full_correct = sum(t.verification.status == 'resolved' and
                           t.verification.diagnostics.get('correctness_score') == 1 for t in correctness_rows)
        result['mean_full_correctness_rate'] = full_correct/n
    for key in ('input_tokens', 'output_tokens', 'tool_calls', 'tool_seconds', 'latency_seconds'):
        values = [t.usage.get(key) for t in trajectories]
        valid = [v for v in values if type(v) in (int, float) and math.isfinite(v) and v >= 0]
        result[key + '_measurement_coverage'] = len(valid)/n
        # Missing instrumentation must never look like free computation.
        result['mean_' + key] = sum(valid)/n if len(valid) == n else None
        result[key + '_per_accepted_answer'] = sum(valid)/len(accepted) if len(valid) == n and accepted else None
        if correctness_rows:
            result[key + '_per_fully_correct_answer'] = sum(valid)/full_correct if len(valid) == n and full_correct else None
    keys = ('input_tokens', 'output_tokens', 'tool_seconds')
    if all(result['mean_' + k] is not None for k in keys):
        result['mean_compute_units'] = (result['mean_input_tokens']
            + 2*result['mean_output_tokens'] + 100*result['mean_tool_seconds'])
        result['compute_units_per_accepted_answer'] = result['mean_compute_units']*n/len(accepted) if accepted else None
    bonuses = []
    for t in trajectories:
        feedback = t.verification.diagnostics.get('training_feedback') if t.verification else None
        if feedback and feedback.get('eligible') and feedback.get('tier') == 'accepted':
            bonuses.append(feedback['reward'] - .9)
    result['mean_accepted_efficiency_bonus'] = sum(bonuses)/len(bonuses) if bonuses else None
    penalties = [t.verification.diagnostics['training_feedback']['efficiency_penalty']
                 for t in trajectories if t.verification
                 and (t.verification.diagnostics.get('training_feedback') or {}).get('efficiency_penalty') is not None]
    result['mean_efficiency_penalty'] = sum(penalties)/len(penalties) if penalties else None
    return result
