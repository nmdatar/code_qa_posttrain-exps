"""Versioned training-only feedback; strict evaluation never uses this scalar.

Only supported reference coverage earns positive credit. The early-training
version applies no penalties; the historical v3 formula is retained for replay.
"""
VERSION = 'positive-coverage-v4'
LEGACY_VERSION = 'supported-coverage-v3'
WEIGHTS = {'contradicted': .75, 'unsupported': .15, 'uncited': .05, 'bad_citation': .05}


def _legacy_shape(semantic, rubric, strict_score):
    if (strict_score is None or not semantic['assessment_complete'] or
            semantic['needs_review'] or semantic['disagreements']):
        return {'version': LEGACY_VERSION, 'reward': None, 'components': {}, 'eligible': False}
    specs = {c['id']: c for c in rubric['claims']}
    required = semantic['required_claims']
    if {c['id'] for c in required} != set(specs):
        raise ValueError('Training reward requires every fixed rubric claim')
    # Partial findings get training feedback without changing strict evaluation's
    # separable-subparts requirement. Unsupported findings never earn points.
    coverage = sum(specs[c['id']]['weight'] *
        ({'complete': 1., 'partial': .5, 'absent': 0.}[c['coverage']]
         if c['verdict'] == 'supported' else 0.) for c in required) / sum(c['weight'] for c in specs.values())
    assertions = semantic['additional_claims']
    contradicted = sum(c['verdict'] == 'contradicted' or c['material_error'] for c in assertions)
    unsupported = sum(c['verdict'] != 'supported' and c['verdict'] != 'contradicted'
                      and not c['material_error'] for c in assertions)
    supported_links = {x['claim_id'] for x in semantic['citation_links'] if x['supported']}
    uncited = len({c['id'] for c in assertions} - supported_links | set(semantic['uncited_claim_ids']))
    bad = len({x['citation_id'] for x in semantic['citation_links'] if not x['supported']})
    components = {'required_coverage': coverage, 'contradicted': contradicted,
                  'unsupported': unsupported, 'uncited': uncited, 'bad_citation': bad}
    score = coverage - sum(WEIGHTS[k]*components[k] for k in WEIGHTS)
    # A material defect may only appear on the required/global finding. Do not
    # let it disappear just because extraction omitted a corresponding flag.
    material = semantic['critical_error'] or semantic['diagram']['material_error'] or any(c['material_error'] for c in required)
    if material and not contradicted:
        score -= WEIGHTS['contradicted']
    if semantic['false_execution_claim']:
        score = 0.
    if semantic['answer_mode'] != 'answer':
        score = min(score, 0.)
    return {'version': LEGACY_VERSION, 'reward': max(0., min(1., score)),
            'components': components, 'eligible': True}


def shape(semantic, rubric, strict_score, *, version=VERSION):
    """Early training: credit supported answer coverage, with no deductions.

    Mistakes remain visible in the strict score and diagnostic components.
    Only relevant required facts earn credit; unrelated true filler earns none.
    """
    result = _legacy_shape(semantic, rubric, strict_score)
    if version == LEGACY_VERSION:
        return result
    if version != VERSION:
        raise ValueError('Unknown training reward version')
    result['version'] = VERSION
    if result['eligible']:
        result['reward'] = result['components']['required_coverage']
    return result


def group_signal(groups):
    """Use whole resolved groups only, exactly as GRPO does."""
    import statistics
    rows = []
    for group in groups:
        valid = bool(group) and all(t.verification and t.verification.status == 'resolved' for t in group)
        shaped = [t.verification.reward for t in group] if valid else []
        strict = [t.verification.diagnostics.get('strict_score', t.verification.reward) for t in group] if valid else []
        strict_complete = bool(strict) and all(x is not None for x in strict)
        rows.append({'task_id': group[0].task_id if group else None, 'eligible': valid,
            'training_rewards': shaped, 'strict_scores': strict,
            'training_variance': statistics.pvariance(shaped) if shaped else None,
            'strict_variance': statistics.pvariance(strict) if strict_complete else None,
            'training_contributes': bool(shaped) and max(shaped)-min(shaped) > 1e-12,
            'strict_contributes': strict_complete and max(strict)-min(strict) > 1e-12})
    return {'groups': rows, 'eligible_groups': sum(r['eligible'] for r in rows),
            'training_contributing_groups': sum(r['training_contributes'] for r in rows),
            'strict_contributing_groups': sum(r['strict_contributes'] for r in rows),
            'excluded_groups': sum(not r['eligible'] for r in rows)}
