"""Training-only error-aware coverage; never substitutes for strict evaluation.

Consumes the complete, source-bound v7 semantic assessment. Evidence references
are checked upstream by the grader; their presence here is an eligibility guard,
not a claim of independent human verification. Coefficients are frozen before
the experiment and inherited from supported-coverage-v3, not fit to selection.
"""
import math

VERSION = 'aligned-coverage-v1'
COVERAGE_VERSION = 'independent-factual-coverage-v3'
WEIGHTS = {'contradicted': .75, 'unsupported': .15,
           'uncited': .05, 'bad_citation': .05}
_UNSET = object()


def feedback(semantic, rubric, strict_score, citation_errors=(), *, factual_coverage=_UNSET):
    """Preserve supported partial credit, then deduct diagnosed answer defects.

    Ambiguous or non-source-backed negative findings are excluded from training,
    rather than converted to model failures. Missing support is a smaller,
    distinct penalty from a source-established contradiction.
    """
    def excluded(reason):
        return {'version': VERSION, 'reward': None, 'components': {},
                'eligible': False, 'reason': reason}

    if (strict_score is None or not semantic['assessment_complete'] or
            semantic['needs_review'] or semantic['disagreements']):
        return excluded('unresolved_assessment')
    if factual_coverage is not _UNSET:
        if factual_coverage is None:
            return excluded('unresolved_factual_coverage')
        if (type(factual_coverage) not in (int, float) or
                not math.isfinite(factual_coverage) or not 0 <= factual_coverage <= 1):
            raise ValueError('Aligned reward requires finite factual coverage in [0,1]')
    specs = {c['id']: c for c in rubric['claims']}
    required = semantic['required_claims']
    if (len(specs) != len(rubric['claims']) or
            len({c['id'] for c in required}) != len(required) or
            {c['id'] for c in required} != set(specs)):
        raise ValueError('Aligned reward requires unique complete rubric findings')
    if not specs or any(type(c['weight']) not in (float, int) or
                        not math.isfinite(c['weight']) or c['weight'] <= 0
                        for c in specs.values()):
        raise ValueError('Aligned reward requires positive finite fixed weights')
    additional = semantic['additional_claims']
    if len({c['id'] for c in additional}) != len(additional):
        raise ValueError('Duplicate additional claim findings')
    available = set(semantic['evidence_keys'])
    negative = [c for c in required + additional
                if c['verdict'] == 'contradicted' or c['material_error']]
    if any(not c['evidence_keys'] or not set(c['evidence_keys']) <= available
           for c in negative):
        return excluded('negative_finding_without_source_evidence')
    global_error = (semantic['critical_error'] or
                    semantic['diagram']['material_error'] or
                    semantic['false_execution_claim'])
    if global_error and not negative:
        return excluded('global_error_without_linked_negative_evidence')
    coverage = sum(specs[c['id']]['weight'] *
                   ({'complete': 1., 'partial': .5, 'absent': 0.}[c['coverage']]
                    if c['verdict'] == 'supported' else 0.)
                   for c in required) / sum(c['weight'] for c in specs.values())
    # Expanded-study integration supplies the exact independent coverage pass
    # used by the control. The default remains useful for source-bound fixtures.
    if factual_coverage is not _UNSET:
        coverage = float(factual_coverage)
    contradicted = sum(c['verdict'] == 'contradicted' or c['material_error']
                       for c in additional)
    # Required/global material errors cannot disappear when the additional
    # assertion extractor did not repeat them. Avoid charging that fallback twice.
    if not contradicted and (global_error or any(c['material_error'] or
                                                c['verdict'] == 'contradicted'
                                                for c in required)):
        contradicted = 1
    unsupported = sum(c['verdict'] not in ('supported', 'contradicted') and
                      not c['material_error'] for c in additional)
    supported_links = {c['claim_id'] for c in semantic['citation_links'] if c['supported']}
    uncited = len(({c['id'] for c in semantic['extracted_claims']} - supported_links) |
                  set(semantic['uncited_claim_ids']))
    bad = len({c['citation_id'] for c in semantic['citation_links'] if not c['supported']})
    # Deterministic citation validation errors were removed from judge evidence.
    # A fixed indicator is used because reason strings are not citation IDs.
    if citation_errors:
        bad = max(1, bad)
    components = {'required_coverage': coverage, 'contradicted': contradicted,
                  'unsupported': unsupported, 'uncited': uncited, 'bad_citation': bad}
    deduction = sum(WEIGHTS[k] * components[k] for k in WEIGHTS)
    reward = max(0., min(1., coverage - deduction))
    if semantic['false_execution_claim'] or semantic['answer_mode'] != 'answer':
        reward = 0.
    return {'version': VERSION, 'reward': reward, 'components': components,
            'eligible': True, 'calibrated': False,
            'definition': {'penalty_weights': dict(WEIGHTS),
                           'coverage_version': COVERAGE_VERSION if factual_coverage is not _UNSET
                           else 'full-semantic-rubric-coverage'}}
