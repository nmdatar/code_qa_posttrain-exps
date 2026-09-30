"""Experimental acceptance-gated reward from the validated strict assessment.

This uses the verifier's explicit tiers, never thresholds factual coverage into
acceptance. Semantic calibration remains a property of the underlying judge.
"""
import math

VERSIONS = ('tier-quality-v1', 'tier-efficiency-v1')


def token_penalty(correctness, output_tokens, output_budget):
    """Retain 90–100% of factual correctness; never reward incorrect brevity."""
    if correctness is None:
        return None
    if (type(correctness) not in (int, float) or not math.isfinite(correctness)
            or not 0 <= correctness <= 1):
        raise ValueError('Invalid correctness')
    if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0
           for v in (output_tokens, output_budget)) or output_budget <= 0:
        raise ValueError('Missing or invalid trusted token counts')
    return correctness * (1 - .1 * min(output_tokens / output_budget, 1.))


def feedback(result, metrics, budget, version, citation_errors=()):
    if version not in VERSIONS:
        raise ValueError('Unknown tiered reward')
    tier = result.get('tier')
    if result.get('status') != 'resolved' or tier == 'unresolved':
        return {'version': version, 'reward': None, 'eligible': False, 'tier': 'unresolved'}
    if tier not in {'failed', 'partial', 'accepted'}:
        raise ValueError('Strict verifier did not return an acceptance tier')
    if citation_errors:
        tier = 'failed'
    claims = result.get('claims', [])
    # Claim weights are supplied by the frozen task rubric, not the judge.
    weights = result['reward_claim_weights']
    findings = {c['id']: c for c in claims}
    if set(findings) != set(weights) or len(findings) != len(claims):
        raise ValueError('Tiered reward requires complete required-claim findings')
    coverage = sum(w * ({'complete': 1., 'partial': .5, 'absent': 0.}[findings[k]['coverage']]
                       if findings[k]['verdict'] == 'supported' else 0.)
                   for k, w in weights.items()) / sum(weights.values())
    values = [metrics.get(k) for k in ('input_tokens', 'output_tokens', 'tool_seconds')]
    if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in values):
        raise ValueError('Missing or invalid trusted efficiency metrics')
    if type(budget) not in (int, float) or not math.isfinite(budget) or budget <= 0:
        raise ValueError('Invalid fixed task compute budget')
    compute = values[0] + 2 * values[1] + 100 * values[2]
    efficiency = max(0., min(1., 1 - compute / budget))
    reward = 0. if tier == 'failed' else .2 * coverage if tier == 'partial' else .9
    if tier == 'accepted' and version == 'tier-efficiency-v1':
        reward += .1 * efficiency
    return {'version': version, 'reward': reward, 'eligible': True, 'tier': tier,
            'components': {'required_coverage': coverage, 'compute_units': compute,
                           'task_compute_budget': budget, 'efficiency': efficiency},
            'calibrated': False}
