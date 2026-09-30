"""Experimental collection adapter for the shared full semantic QA verifier.

Uses qa_eval extraction, assessment, schema/provenance validation and decision
rules. Automated reference admission remains non-human-reviewed; these scores
are research coverage, not calibrated production acceptance or efficiency reward.
"""
from qa_eval.judging import judge, POLICY
from qa_eval.grading import decide
from qa_eval.schema import EVIDENCE, validate
from qa_eval.security import digest
from .admission import blob, sha
from .evidence_compaction import compact

VERSION = 'all-claims-v3'
RELIABLE_VERSION = 'all-claims-v5'
POSITIVE_VERSION = 'all-claims-v6'
VERSIONS = (VERSION, 'all-claims-v4', RELIABLE_VERSION, POSITIVE_VERSION, 'all-claims-v7')
JUDGE_CALLS = 2
EXTRACTION_INSTRUCTION = ('This stage only inventories assertions, not their truth. '
    'Include false and unsupported assertions too. extraction_complete means every '
    'assertion has been enumerated; lack of supporting evidence alone does not make '
    'extraction incomplete. answer_mode describes the submitted candidate answer, '
    'not your own uncertainty. Assessment of truth happens in the next stage. '
    'Request source with path, line ranges and file_sha256 only; omit the optional symbol field. '
    'Every evidence request MUST span at most 600 lines, with 1 <= start_line <= end_line <= line_count '
    'from the catalog. Prefer the cited ranges. Never request an entire large file. '
    'Copy file hashes exactly from the catalog; do not invent paths or hashes.')
EXTRACTION_INSTRUCTION += (' Statements of uncertainty such as "I could not determine the answer" '
    'are not substantive assertions about repository behavior. For a pure abstention, return '
    'claims=[] and answer_mode="abstention". A concrete assertion that a symbol does not exist '
    'is different and must still be extracted.')
ADMISSION_INSTRUCTION = ('The reference rubric is automatically admitted, not human gold. '
    'Pinned source is authoritative. If a required reference claim conflicts with source or '
    'its truth cannot be verified, set needs_review=true; do not turn a defective reference '
    'into a negative reward for the candidate. Audit every additional candidate assertion. '
    'The disagreements field is only for unresolved evidence/reference ambiguity that needs adjudication. '
    'It is NOT a list of ordinary mistakes in the answer. When evidence establishes an answer error, '
    'record it in the claim verdict/material_error and citation_links, with needs_review=false and '
    'disagreements=[] unless a separate genuine ambiguity remains. Unsupported candidate elaboration '
    'or a citation that misses the supporting lines is an answer defect, not automatically a defective rubric. '
    'Never clear genuine uncertainty just to produce a numeric score.')
ADMISSION_INSTRUCTION += (' Evidence identifiers are short aliases (e1, e2, etc.) bound to verified source. '
    'Use only supplied evidence keys, claim IDs and citation IDs, copied exactly. '
    'A supported claim with non-absent coverage MUST include at least one evidence key. '
    'If an assertion is only plausible external knowledge, mark insufficient instead of supported. '
    'Negative claims about the entire repository require adequate evidence; unsupported negative '
    'candidate claims can be marked insufficient without declaring the reference ambiguous.')
ADMISSION_INSTRUCTION += (' Interpret short reference claims in the scope of their cited function or test, '
    'not as a claim about the entire repository unless explicitly global. In tests, fixtures can mean '
    'the arranged test inputs, including inline objects, not only functions decorated with pytest.fixture.')
ADMISSION_INSTRUCTION += (' Grade semantic content, not lexical overlap. A source-supported paraphrase '
    'earns complete coverage when it expresses the whole required fact, including its relevant conditions. '
    'Do not invent an identifier or class from a capitalized ordinary word such as Context. '
    'Naming the helper that implements a context manager operation can fully express that operation '
    'when the answer connects the helper to that context; do not require the reference wording. '
    'Likewise, level_names can denote the instance property self.level_names in an explanation; '
    'do not infer a different local variable unless the answer actually asserts that distinction. '
    'For partial coverage, identify a specific missing or incorrect factual requirement, not merely '
    'a different word or level of abstraction. Do not infer missing steps from source alone. '
    'A correct collection step does not imply that the answer described footprint wrapping or '
    'checking against references. Keep each required fact independent of unrelated answer errors.')
ADMISSION_INSTRUCTION += (' The text of each required claim is its complete scoring criterion. '
    'Use the question only to resolve scope or pronouns, NEVER to add requirements to that claim. '
    'Do not demand implementation detail or a mechanism that the claim itself does not require. '
    'For example, if the required claim says A delegates to B and the answer correctly says that, '
    'coverage is complete even if the broader question asks other things. '
    'If a candidate omits a required claim, use absent/insufficient; do not treat the reference '
    'text as something the candidate asserted or invent a contradiction. '
    'Distinguish processing/traversal order from output-container ordering, and source declaration '
    'order from runtime execution order. Do not infer a Python runtime version from code style. '
    'If supplied source cannot verify a required reference claim, flag reference uncertainty '
    'for review rather than treating an evidence gap as proof the candidate is wrong.')


def normalize_extraction(value):
    """An empty optional symbol means no symbol constraint, not a new assertion.

    Keep raw provider output on disk. All claims, hashes, ranges, completeness
    flags and nonempty symbols still pass through the strict verifier unchanged.
    """
    import copy
    value = copy.deepcopy(value)
    if isinstance(value, dict) and isinstance(value.get('claims'), list):
        for claim in value['claims']:
            if isinstance(claim, dict) and isinstance(claim.get('evidence_requests'), list):
                expanded = []
                for ref in claim['evidence_requests']:
                    if isinstance(ref, dict) and 'symbol' in ref and ref['symbol'] in ('', None):
                        del ref['symbol']
                    # Preserve every requested line; split valid oversized ranges
                    # into bounded reads rather than dropping source or assertions.
                    if (isinstance(ref,dict) and type(ref.get('start_line')) is int and
                            type(ref.get('end_line')) is int and
                            600 <= ref['end_line']-ref['start_line'] < 2400):
                        for start in range(ref['start_line'],ref['end_line']+1,600):
                            expanded.append({**ref,'start_line':start,'end_line':min(start+599,ref['end_line'])})
                    else:
                        expanded.append(ref)
                claim['evidence_requests'] = expanded
    return value


def assess(request, call, version=VERSION, evidence_policy='legacy-v1'):
    row = request['source_row']
    rubric, answer = request['rubric'], request['answer']
    inventory = row['image_result']['snapshot_files']
    commit = row['public']['repository']['commit']

    def evidence_reader(ref):
        validate(ref, EVIDENCE)
        if 'symbol' in ref:
            raise ValueError('Collection evidence requests must use line ranges, not symbols')
        if inventory.get(ref['path']) != ref['file_sha256']:
            raise ValueError('Evidence does not match pinned inventory')
        content = blob(row['snapshot_root'], commit, ref['path'])
        if sha(content) != ref['file_sha256']:
            raise ValueError('Pinned evidence changed')
        lines = content.decode().splitlines()
        a, b = ref['start_line'], ref['end_line']
        if not 1 <= a <= b <= len(lines) or b-a >= 600:
            raise ValueError('Evidence request outside bounded source range')
        key = digest({k: ref[k] for k in ('path','start_line','end_line','file_sha256')})
        return key, {'path':ref['path'], 'start_line':a, 'end_line':b,
                     'text':'\n'.join(f'{i}: {lines[i-1]}' for i in range(a,b+1))}

    evidence = {}
    source_refs = rubric['evidence'] + request.get('answer_evidence', [])
    if evidence_policy == 'definition-context-v1':
        from .definition_evidence import definition_context
        source_refs += definition_context(request)
    elif evidence_policy != 'legacy-v1':
        raise ValueError('Unsupported judge evidence policy')
    for ref in source_refs:
        clean = {k:ref[k] for k in ('path','start_line','end_line','file_sha256')}
        key, text = evidence_reader(clean)
        evidence[key] = text
    # The reference often cites a narrow implementation fragment. Give the judge
    # surrounding source to assess its scope; never alter the candidate's citations.
    for ref in rubric['evidence']:
        count = len(blob(row['snapshot_root'],commit,ref['path']).splitlines())
        context = {k:ref[k] for k in ('path','file_sha256')}
        padding = 40 if row.get('reference', {}).get('evidence_revision') == 'reference-citations-v1' else (120 if version != VERSION else 60)
        padding = min(padding, max(0, (600 - (ref['end_line']-ref['start_line']+1)) // 2))
        context.update(start_line=max(1,ref['start_line']-padding),end_line=min(count,ref['end_line']+padding))
        key,text = evidence_reader(context)
        evidence[key] = text
    # Explicitly scoped catalog: observed and reference files. The reader also
    # permits independently requested, hash-bound evidence in the full inventory.
    known = {ref['path'] for ref in source_refs} | set(request.get('observed_files', {}))
    catalog = [{'path':p,'file_sha256':inventory[p],
                'line_count':len(blob(row['snapshot_root'], commit, p).splitlines())} for p in sorted(known)]
    task = {'id':row['id'], 'question':request['question'], 'repository':row['public']['repository'],
            'human_reviewed':False, 'gold_status':'draft', 'answerability':'answerable',
            'claims':[{'id':c['id'],'text':c['text'],'weight':c['weight'],
                       'separable_subparts':[]} for c in rubric['claims']],
            'critical_errors':['Material falsehood or false claim of code execution'],
            'diagram':{'required':False,'criteria':[],'allowed_abstractions':[]}}
    family = request.get('judge_family', 'qwen')
    experiment = {'training_judge_family':family, 'evaluation_judge_family':family,
                  'judge_version':version, 'judge_command':[], 'judge_timeout_seconds':120,
                  'max_evidence_bytes':96000 if version != VERSION else 48000, 'independent_evaluation':False}
    if answer.get('diagram') is not None:
        raise ValueError('Collection protocol does not support diagrams')
    semantic = judge(task, answer, experiment, 'training' if row['split']=='train' else 'evaluation',
                     None, evidence, None, digest({'commit':commit,'inventory':inventory}),
                     {'execution_records':[],'probes':[]}, call=call,
                     source_catalog=catalog, evidence_reader=evidence_reader,
                     evidence_transform=compact
                         if version != VERSION else None,
                     training_partial_credit=version in (POSITIVE_VERSION, 'all-claims-v7') and row['split'] == 'train')
    score, tier, material, reasons = strict_score(task, answer, semantic)
    from .shaped_reward import shape, VERSION as reward_version, LEGACY_VERSION
    shaped = shape(semantic, rubric, score,
                   version=reward_version if version in (POSITIVE_VERSION, 'all-claims-v7') else LEGACY_VERSION)
    return {'strict_score':score, 'training_feedback':shaped, 'status':'unresolved' if tier=='unresolved' else 'resolved', 'score':score,
            'reason':'; '.join(reasons) or 'All submitted assertions and required claims assessed',
            'claims':semantic['required_claims'], 'claim_count':len(task['claims']),
            'rubric_hash':rubric['rubric_hash'], 'aggregation_version':version,
            'semantic':semantic, 'tier':tier, 'material_error':material,
            'human_reviewed':False,'independent_evaluation':False}


def strict_score(task, answer, semantic):
    tier, coverage, material, reasons = decide(task, answer, semantic, [])
    # Any unsupported/uncited extra assertion blocks strict evaluation credit. Partial
    # coverage alone remains useful, provided every submitted assertion passes.
    extra_ok = all(c['verdict']=='supported' and c['coverage']=='complete'
                   for c in semantic['additional_claims'])
    linked = {c['claim_id'] for c in semantic['citation_links'] if c['supported']}
    citations_ok = (not semantic['uncited_claim_ids'] and
        {c['id'] for c in semantic['extracted_claims']} <= linked and
        all(c['supported'] for c in semantic['citation_links']) and
        {c['id'] for c in answer['citations']} <=
        {c['citation_id'] for c in semantic['citation_links'] if c['supported']})
    score = None if tier=='unresolved' else (coverage if tier!='failed' and extra_ok and citations_ok else 0.0)
    return score, tier, material, reasons
