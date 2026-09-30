"""Opt-in correctness objective and separate citation-only diagnostics."""
VERSION = 'correctness-only-v1'

def enabled(config):
    return config.get('environment', {}).get('scoring_policy') == VERSION

def citation_diagnostics(submission, semantic, errors):
    if not submission.get('citations'):
        return {'citation_score':0.0,'citation_status':'missing','citation_errors':list(errors)}
    if errors:
        return {'citation_score':0.0,'citation_status':'invalid','citation_errors':list(errors)}
    if not semantic or not semantic.get('assessment_complete'):
        return {'citation_score':None,'citation_status':'unresolved','citation_errors':[]}
    ids={c['id'] for c in semantic.get('extracted_claims',[])}
    linked={c['claim_id'] for c in semantic.get('citation_links',[]) if c['supported']}
    return {'citation_score':len(ids & linked)/len(ids) if ids else None,
            'citation_status':'assessed' if ids else 'not_applicable','citation_errors':[]}
