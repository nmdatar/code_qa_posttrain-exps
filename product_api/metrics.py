"""Public cumulative accounting derived from replayable events, without provider secrets."""
import time


def summarize(events, result, config):
    metrics = dict(input_tokens=0, output_tokens=0, cost_usd=0.0, steps=0, tool_calls=0,
        model_seconds=0.0, tool_seconds=0.0)
    complete = True
    pending = None
    calls = {}
    for event in events:
        kind, elapsed = event['kind'], event['elapsed_seconds']
        if kind == 'model_request':
            pending = elapsed
            metrics['steps'] += 1
        if kind in ('model_response', 'invalid_action'):
            if pending is not None:
                metrics['model_seconds'] += max(0, elapsed - pending)
                pending = None
            usage = event.get('usage') or {}
            for key in ('input_tokens', 'output_tokens', 'cost_usd'):
                value = usage.get(key)
                if value is None:
                    if key != 'cost_usd': complete = False
                    if key == 'cost_usd': metrics[key] = None
                elif metrics[key] is not None:
                    metrics[key] += value
        if kind == 'tool_call':
            metrics['tool_calls'] += 1
            calls[event['call_id']] = elapsed
        if kind == 'tool_observation' and event['call_id'] in calls:
            metrics['tool_seconds'] += max(0, elapsed - calls.pop(event['call_id']))
    final = result.get('metrics', {})
    # Old trajectories may lack public invalid-action usage. Completed totals are authoritative.
    for key in ('input_tokens', 'output_tokens', 'cost_usd', 'steps', 'tool_calls', 'tool_seconds'):
        if final.get(key) is not None:
            metrics[key] = final[key]
    fallback_elapsed = max(0, time.time() - config['created_at']) if result['status'] == 'running' else max((e['elapsed_seconds'] for e in events), default=0)
    metrics['elapsed_seconds'] = final.get('elapsed_seconds', fallback_elapsed)
    if pending is not None:
        metrics['model_seconds'] += max(0, metrics['elapsed_seconds'] - pending)
    complete = complete and pending is None and result['status'] not in ('cancelled', 'interrupted', 'infrastructure_error')
    metrics['token_usage_complete'] = complete
    if not complete or (events and not final and result['status'] != 'running'):
        metrics['cost_usd'] = None
    pricing = config['model'].get('pricing')
    metrics['estimated_cost_usd'] = None
    if pricing and metrics['cost_usd'] is None:
        metrics['estimated_cost_usd'] = (metrics['input_tokens'] * pricing['input_per_million'] +
            metrics['output_tokens'] * pricing['output_per_million']) / 1_000_000
    # Zero before any response is not verified billing.
    if not any(e['kind'] in ('model_response', 'invalid_action') for e in events) and not final:
        metrics['cost_usd'] = None
    return metrics
