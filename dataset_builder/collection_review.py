"""Bind independent reference reviews to immutable task and source records."""
import hashlib
import json
from pathlib import Path
from .build import read_jsonl, write_json
from .collection import file_sha, load
from .git_source_environment import read_git_file


def review_records(root):
    folder=Path(root)/'reports/task-generation-1000';records={}
    for path in sorted([*folder.glob('semantic-review-[0-9]*.json'),*(folder/'full-review').glob('batch-*.json')]):
        report=load(path)
        if report.get('human_reviewed') is not False:raise ValueError('Agent review cannot claim human approval')
        for row in report['tasks']:
            if row['task_id'] in records:raise ValueError('Duplicate semantic review task')
            records[row['task_id']]={**row,'review_report':str(path.relative_to(root)),'review_report_sha256':file_sha(path)}
    return records


def validate_review(review, reference, quality, checkout, snapshot_files, cache=None):
    if review['task_id']!=reference['id']:raise ValueError('Review task ID mismatch')
    if review['reference_sha256']!=quality['reference_sha256'] or review['question_sha256']!=quality['question_sha256']:raise ValueError('Stale semantic review')
    if review['status'] not in {'supported','needs_review','reject'}:raise ValueError('Unknown review status')
    if not review.get('checked_claims'):raise ValueError('Review has no source-checked claims')
    if review['status']=='supported' and review.get('material_errors'):raise ValueError('Supported review contains material errors')
    cache={} if cache is None else cache
    evidence=[]
    for claim in review['checked_claims']:
        if not claim.get('claim') or not claim.get('reason') or not claim.get('verdict'):raise ValueError('Incomplete claim review')
        span=claim['evidence'];path=span['path']
        if path not in snapshot_files:raise ValueError('Review evidence is not in solver snapshot: '+path)
        key=(str(checkout),reference['repository']['commit'],path)
        if key not in cache:
            data=read_git_file(checkout,key[1],path)
            if hashlib.sha256(data).hexdigest()!=snapshot_files[path]:raise ValueError('Review source hash mismatch')
            cache[key]=len(data.decode('utf-8').splitlines())
        start,end=span['start_line'],span['end_line']
        if not isinstance(start,int) or not isinstance(end,int) or not 1<=start<=end<=cache[key]:raise ValueError(f'Review evidence outside file: {path}:{start}-{end} ({cache[key]} lines)')
        evidence.append({**span,'file_sha256':snapshot_files[path]})
    return {**review,'evidence_integrity':'passed','verified_evidence':evidence,'review_scope':'Central-claim static source review; not exhaustive sentence-level validation or human approval'}


def audit_reviews(root):
    root=Path(root);reviews=review_records(root);verified={};failures=[];cache={}
    for bundle in sorted((root/'data/generated/collection-1000-v1').iterdir()):
        refs=read_jsonl(bundle/'private/references.jsonl');quality={r['task_id']:r for r in read_jsonl(bundle/'private/quality.jsonl')}
        env=load(bundle/'public/environment.json');built=load(bundle/'environment-build/result.json')
        for ref in refs:
            if ref['id'] not in reviews:continue
            try:verified[ref['id']]=validate_review(reviews[ref['id']],ref,quality[ref['id']],Path(env['snapshot_path']),built['snapshot_files'],cache)
            except (ValueError,KeyError,UnicodeDecodeError) as exc:failures.append({'task_id':ref['id'],'review_report':reviews[ref['id']]['review_report'],'error':str(exc)})
    result={'reviewed':len(reviews),'evidence_verified':len(verified),'failures':failures,'verified':verified}
    write_json(root/'reports/task-generation-1000/review-evidence-audit.json',result)
    return result



def correction_records(root):
    records={};folder=Path(root)/'reports/task-generation-1000'
    for path in sorted((folder/'correction-review').glob('worker-*/*.json')):
        report=load(path);queue=load(folder/'correction-queue'/path.parent.name/path.name)
        if report.get('human_reviewed') is not False:raise ValueError('Correction review cannot claim human approval')
        if queue['author_worker']==queue['reviewer_worker']:raise ValueError('Correction was reviewed by its author')
        if report.get('reviewer_worker')!=queue['reviewer_worker']:raise ValueError('Wrong correction reviewer')
        expected={r['task_id']:r for r in queue['tasks']}
        if {r['task_id'] for r in report['tasks']}!=set(expected):raise ValueError('Correction review missing tasks')
        for row in report['tasks']:
            original=expected[row['task_id']]
            for key in ['reference_sha256','question_sha256','original_reference_sha256']:
                if row.get(key)!=original[key]:raise ValueError('Correction review hash mismatch')
            if row['task_id'] in records:raise ValueError('Duplicate correction review')
            records[row['task_id']]={**row,'review_report':str(path.relative_to(root)),'review_report_sha256':file_sha(path),'author_worker':queue['author_worker'],'reviewer_worker':queue['reviewer_worker']}
    return records


def audit_corrections(root):
    root=Path(root);reviews=review_records(root);corrections=correction_records(root);verified={};failures=[];cache={}
    for bundle in sorted((root/'data/generated/collection-1000-v1').iterdir()):
        refs=read_jsonl(bundle/'private/references.jsonl');quality={r['task_id']:r for r in read_jsonl(bundle/'private/quality.jsonl')}
        env=load(bundle/'public/environment.json');built=load(bundle/'environment-build/result.json')
        for ref in refs:
            if ref['id'] not in corrections:continue
            correction=corrections[ref['id']]
            try:
                answer=reviews[ref['id']]['corrected_reference_answer'];q=quality[ref['id']]
                if correction['original_reference_sha256']!=q['reference_sha256']:raise ValueError('Correction points to wrong original answer')
                corrected_quality={**q,'reference_sha256':hashlib.sha256(answer.encode()).hexdigest()}
                verified[ref['id']]=validate_review(correction,{**ref,'reference_answer':answer},corrected_quality,Path(env['snapshot_path']),built['snapshot_files'],cache)
            except (ValueError,KeyError,UnicodeDecodeError) as exc:failures.append({'task_id':ref['id'],'review_report':correction['review_report'],'error':str(exc)})
    result={'reviewed':len(corrections),'evidence_verified':len(verified),'failures':failures,'verified':verified}
    write_json(root/'reports/task-generation-1000/correction-evidence-audit.json',result)
    return result

if __name__=='__main__':
    for name,fn in [('original',audit_reviews),('corrections',audit_corrections)]:
        r=fn(Path.cwd());print(json.dumps({'stage':name,**{k:v for k,v in r.items() if k!='verified'}},indent=2))
