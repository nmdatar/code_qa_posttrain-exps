"""Known-answer questions and frozen, hash-verified assertion rubrics."""
import json
from product_api.catalog import RELEASE, open_repository
from training_pipeline.claim_grading import VERSION, reference_rubric, validate_claims
from training_pipeline.storage import digest


def catalog(repositories):
    references = {r['task_id']: r for line in (RELEASE/'private/grading.jsonl').open() if (r := json.loads(line))}
    rows = {}
    for file in (RELEASE/'rl/train.jsonl', RELEASE/'evaluation/development/tasks.jsonl'):
        for line in file.open():
            task = json.loads(line)
            reference = references.get(task['id'], {})
            record = reference.get('record', {})
            count = len(reference.get('reviewed_claims', [])) or len(record.get('claims', []))
            if not count or record.get('answerability', 'answerable') != 'answerable': continue
            source = task['repository']
            name = source['url'].removeprefix('https://github.com/').removesuffix('.git')
            repo = repositories.get(task['environment_id'])
            if repo is None:
                repo = next((r for r in repositories.values() if r['name'] == name and r['commit'] == source['commit']
                    and (r['execution'] or not {'python_probe','run_tests'} & set(task['permitted_tools']))), None)
            if not repo or not repo['ready'] or repo['commit'] != source['commit']: continue
            rows[task['id']] = {'id':task['id'], 'repo_id':repo['id'], 'question':task['user_prompt'],
                'claim_count':count, 'split':task['split'], 'reference_status':reference.get('gold_status',record.get('gold_status','draft')),
                'human_reviewed':reference.get('human_reviewed',record.get('human_reviewed',False)),
                'task':task, 'reference':reference}
    return rows


def public(row):
    return {k:v for k,v in row.items() if k not in ('task','reference')}


def freeze(row, repo_config):
    repo, _ = open_repository(repo_config)
    reference = row['reference']
    record = reference.get('record', {})
    refs = reference.get('verified_evidence', []) or [e for c in record['claims'] for e in c['evidence']]
    evidence, seen = [], set()
    for ref in refs:
        key = (ref['path'], ref['start_line'], ref['end_line'])
        if key in seen: continue
        text, sha = repo.text(ref['path'])
        if sha != ref['file_sha256']: raise ValueError('Benchmark evidence does not match pinned source')
        lines = text.splitlines()
        if not 1 <= ref['start_line'] <= ref['end_line'] <= len(lines): raise ValueError('Invalid reference source range')
        evidence.append({**ref, 'id':'e'+str(len(evidence)+1), 'text':'\n'.join(lines[ref['start_line']-1:ref['end_line']])})
        seen.add(key)
    if reference.get('reviewed_claims'):
        rubric = reference_rubric(reference, evidence)
    else:
        claims = [{k:c[k] for k in ('id','text','weight')} | {'evidence_ids': [e['id'] for e in evidence
            if any(e['path']==r['path'] and e['start_line']<=r['start_line'] and e['end_line']>=r['end_line'] for r in c['evidence'])]}
            for c in record['claims']]
        validate_claims(claims)
        rubric = {'version':VERSION,'claims':claims,'evidence':evidence}
        rubric['rubric_hash'] = digest(rubric)
    return {**public(row), 'commit':repo.commit, 'rubric':rubric,
        'reference_answer':reference.get('reference_answer') or '\n\n'.join(c['text'] for c in rubric['claims']),
        'permitted_tools':row['task']['permitted_tools']}
