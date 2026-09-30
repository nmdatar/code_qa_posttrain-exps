"""Recount frozen collection exports and verify their cross-file bindings."""
import argparse
from collections import Counter
from pathlib import Path
import hashlib,json
from .build import read_jsonl
from .collection import file_sha,load,normalized_question
from .contracts import validate_public_task, canonical_hash


def audit_release(root):
    root=Path(root).resolve();manifest=load(root/'manifest.json')
    actual={str(p.relative_to(root)) for p in root.rglob('*') if p.is_file() and p!=root/'manifest.json'}
    if actual!=set(manifest['artifacts']):raise ValueError('Release file inventory differs from manifest')
    for relative,digest in manifest['artifacts'].items():
        path=(root/relative).resolve()
        if not path.is_relative_to(root) or file_sha(path)!=digest:raise ValueError('Release artifact hash mismatch: '+relative)
    tasks=read_jsonl(root/'public/tasks.jsonl');refs=read_jsonl(root/'private/references.jsonl');qualities=read_jsonl(root/'private/quality.jsonl');envs=read_jsonl(root/'public/environments.jsonl');grading=read_jsonl(root/'private/grading.jsonl')
    def unique(rows,key):
        result={r[key]:r for r in rows}
        if len(result)!=len(rows):raise ValueError('Duplicate '+key)
        return result
    t=unique(tasks,'id');r=unique(refs,'id');q=unique(qualities,'task_id');e=unique(envs,'environment_id');g=unique(grading,'task_id')
    if set(t)!=set(r) or set(t)!=set(q):raise ValueError('Release task IDs mismatch')
    assertions=unique(read_jsonl(root/'private/source-and-behavior-assertions.jsonl'),'task_id')
    if set(assertions)!=set(t):raise ValueError('Task assertions mismatch')
    expected={k for k,v in q.items() if v['grading_reference_available']}
    if set(g)!=expected or set(load(root/'private/reviewed-grading-task-ids.json'))!=expected:raise ValueError('Grading inclusion mismatch')
    families={};questions=set();source_rows={}
    for task in tasks:
        validate_public_task(task);norm=normalized_question(task['user_prompt'])
        if norm in questions:raise ValueError('Duplicate normalized prompt')
        questions.add(norm);environment=e[task['environment_id']]
        if environment['repository']!=task['repository'] or not environment['runtime_checked']:raise ValueError('Environment binding mismatch')
        if environment['capability']=='source_reading' and task['permitted_tools']!=['list_files','search_code','read_file']:raise ValueError('Unexpected execution capability')
        family=task['repository']['family_id'];split=task['split']
        if split!='development' or (family in families and families[family]!=split):raise ValueError('Split leakage')
        families[family]=split;ref=r[task['id']]
        if ref['reference_kind']=='upstream_benchmark_draft':
            if ref['question']!=task['user_prompt'] or ref['repository']!=task['repository']:raise ValueError('Private task binding mismatch')
            if hashlib.sha256(task['user_prompt'].encode()).hexdigest()!=q[task['id']]['question_sha256']:raise ValueError('Question hash mismatch')
            if hashlib.sha256(ref['reference_answer'].encode()).hexdigest()!=q[task['id']]['reference_sha256']:raise ValueError('Original reference hash mismatch')
            if file_sha(root/ref['release_source_file'])!=ref['provenance']['source_file_sha256']:raise ValueError('Source provenance hash mismatch')
            if ref['release_source_file'] not in source_rows:source_rows[ref['release_source_file']]=read_jsonl(root/ref['release_source_file'])
            source=source_rows[ref['release_source_file']][ref['provenance']['source_row_index']]
            if source['question']!=ref['question'] or source['answer']!=ref['reference_answer']:raise ValueError('Upstream row lineage mismatch')
            if 'commit_id' in source and source['commit_id']!=task['repository']['commit']:raise ValueError('Upstream commit mismatch')
            if 'repo' in source and task['repository']['url']!='https://github.com/'+source['repo']:raise ValueError('Upstream repository mismatch')
            if ref['provenance'].get('original_short_commit') and not task['repository']['commit'].startswith(ref['provenance']['original_short_commit']):raise ValueError('Upstream short pin mismatch')
            if task['id'] in g:
                answer=g[task['id']]['reference_answer'];digest=hashlib.sha256(answer.encode()).hexdigest()
                if answer!=ref['selected_reference_answer'] or digest!=g[task['id']]['reference_sha256'] or digest!=q[task['id']]['selected_reference_sha256']:raise ValueError('Selected grading reference mismatch')
                if q[task['id']]['reference_admission']=='quarantined_reference':raise ValueError('Quarantined reference in grading')
    if families!=load(root/'split-manifest.json')['families']:raise ValueError('Family manifest mismatch')
    for environment in envs:
        evidence=root/'private/environment-evidence'/environment['environment_id']
        built=load(evidence/'result.json');attestation=load(evidence/'image-attestation.json');isolation=load(evidence/'isolation-report.json')
        prepared=load(evidence/'source-environment.json')
        if built.get('source_environment_sha256')!=canonical_hash(prepared):raise ValueError('Build recipe/source environment binding mismatch')
        if built['image_digest']!=environment['image_id'] or built['snapshot_sha256']!=environment['snapshot_sha256']:raise ValueError('Published image binding mismatch')
        if attestation['status']!='passed' or isolation['passed'] is not True:raise ValueError('Environment checks not passed')
    for name in ['sft/train.jsonl','rl/train.jsonl','preferences/train.jsonl','evaluation/final_test.jsonl']:
        if read_jsonl(root/name):raise ValueError('Development benchmark records leaked into training/test')
    counts=manifest['counts']
    for key,actual_count in [('total_runnable_tasks',len(t)),('environments',len(e)),('families',len(families)),('grading_reference_available',len(g))]:
        if counts[key]!=actual_count:raise ValueError('Incorrect reported count: '+key)
    if counts['reference_review_states']!=dict(Counter(x['semantic_review'] for x in qualities)):raise ValueError('Incorrect review counts')
    if counts['reference_admission_states']!=dict(Counter(x['reference_admission'] for x in qualities)):raise ValueError('Incorrect admission counts')
    return {'status':'passed','tasks':len(t),'grading_references':len(g),'environments':len(e),'families':len(families),'artifact_hashes_checked':len(manifest['artifacts']),'scope':'Frozen export inventory, hashes, schema, task/environment/source/reference bindings, unique prompts, quality counts, split separation and grading inclusion; not a new semantic review.'}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('release',type=Path);a=p.parse_args();print(json.dumps(audit_release(a.release),indent=2))
