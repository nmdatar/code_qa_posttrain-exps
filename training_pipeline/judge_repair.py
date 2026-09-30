"""One bounded mechanical repair; never retry a valid unfavorable judgment."""
import copy
import json
import time
from dataclasses import asdict
from qa_eval.schema import validate, unique_ids
from .storage import atomic_json
from .strict_grading import normalize_extraction
from .claim_grading import _unique_object


def validate_response(result, payload, source_bounds=False):
    validate(result, payload['output_schema'])
    if payload['stage'] == 'extract':
        unique_ids(result['claims'], 'extracted claims')
        if not result['extraction_complete']:
            raise ValueError('Incomplete claim extraction')
        if source_bounds:
            catalog = {r['path']:r for r in payload['untrusted']['repository_catalog']}
            for claim in result['claims']:
                for ref in claim['evidence_requests']:
                    a, b = ref['start_line'], ref['end_line']
                    if not 1 <= a <= b or b-a >= 600:
                        raise ValueError('Each evidence range must have ordered endpoints and at most 600 lines')
                    entry = catalog.get(ref['path'])
                    if entry:
                        if ref['file_sha256'] != entry['file_sha256']:
                            raise ValueError('Evidence hash must exactly match the supplied catalog')
                        if b > entry['line_count']:
                            raise ValueError('Evidence range exceeds '+ref['path']+' line_count='+str(entry['line_count']))
        return
    for field, expected in [('required_claims', payload['rubric']['claims']),
                            ('additional_claims', payload['untrusted']['extracted_claims'])]:
        unique_ids(result[field], field)
        if {c['id'] for c in result[field]} != {c['id'] for c in expected}:
            raise ValueError('Omitted or invented '+field)
        for finding in result[field]:
            if set(finding['evidence_keys']) - set(payload['untrusted']['evidence']):
                raise ValueError('Unknown evidence key')
            if finding['verdict'] == 'supported' and finding['coverage'] != 'absent' and not finding['evidence_keys']:
                raise ValueError('Supported assertion lacks verified evidence keys')
    claim_ids = {c['id'] for c in result['required_claims']+result['additional_claims']}
    citation_ids = {c['id'] for c in payload['untrusted']['citations']}
    if set(result['uncited_claim_ids'])-claim_ids or any(x['claim_id'] not in claim_ids or x['citation_id'] not in citation_ids for x in result['citation_links']):
        raise ValueError('Unknown citation or claim ID')


def sample_response(factory, request, payload, queue_seconds):
    from .strict_grading import POLICY
    original = copy.deepcopy(payload)
    attempts = factory.config['judge'].get('repair_attempts', 0)+1
    for attempt in range(attempts):
        began = time.monotonic()
        sample = factory.judge.sample([{'role':'system','content':POLICY},
            {'role':'user','content':json.dumps(payload)}], factory.config['judge']['max_tokens'],
            factory.config['judge'].get('temperature', 0))
        artifact = factory.root/'private'/(request['episode_id']+'.'+payload['stage']+
                    ('' if attempt == 0 else f'.repair-{attempt}')+'.judge-raw.json')
        record = {'generation':asdict(sample), 'request':payload, 'attempt':attempt,
                  'version':factory.reward_version, 'judge_identity':factory.judge.identity,
                  'timing':{'queue_seconds':queue_seconds if payload['stage']=='extract' and attempt==0 else 0,
                            'sampling_seconds':time.monotonic()-began}}
        atomic_json(artifact, record)
        try:
            if sample.stop_reason == 'length':
                raise ValueError('Truncated full-claim audit')
            raw = sample.text.strip()
            if raw.startswith('```json') and raw.endswith('```'):
                raw = raw[7:-3].strip()
            result = json.loads(raw, object_pairs_hook=_unique_object,
                parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
            if payload['stage'] == 'extract':
                result = normalize_extraction(result)
            validate_response(result, original, factory.config.get('environment',{}).get('grading_version') in ('all-claims-v5', 'all-claims-v6', 'all-claims-v7'))
            return result
        except (ValueError, KeyError, TypeError) as exc:
            record['validation_error'] = str(exc)
            atomic_json(artifact, record)
            if attempt+1 == attempts:
                raise
            payload = copy.deepcopy(original)
            payload['instructions'] += (' A previous response failed mechanical validation: '+str(exc)+
                '. Return a complete valid response. Keep every claim and genuine uncertainty. '
                'Do not change a verdict merely to satisfy validation; use insufficient if support is absent. '
                'Keep reasons concise (one short sentence) to fit the response limit.')
            # Failed text is explicitly untrusted and cannot alter the rubric.
            payload['untrusted']['previous_invalid_response'] = sample.text[:4096]
    raise AssertionError('Unreachable')
