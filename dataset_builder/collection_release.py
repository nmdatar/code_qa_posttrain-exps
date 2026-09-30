"""Package runnable task records with explicit, separate reference quality states."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil

from .build import read_jsonl, write_json, write_jsonl
from .collection import file_sha, load, normalized_question
from .collection_environments import _ready, _bound, _runtime_reusable
from .contracts import canonical_hash, validate_public_task


from .collection_review import review_records, validate_review, correction_records


def audit_import_bundle(bundle):
    manifest=load(bundle/'manifest.json')
    required={'public/tasks.jsonl','public/environment.json','private/references.jsonl','private/quality.jsonl','private/assertions.jsonl'}
    if not required.issubset(manifest['artifacts']):raise ValueError('Incomplete import manifest')
    for relative,digest in manifest['artifacts'].items():
        path=bundle/relative
        if not path.resolve().is_relative_to(bundle.resolve()) or file_sha(path)!=digest:raise ValueError('Bundle artifact changed')
    env=load(bundle/'public/environment.json');built=load(bundle/'environment-build/result.json')
    if not _ready(built,env):raise ValueError('Unready or stale image')
    for name,key,value in [('isolation-report.json','passed',True),('image-attestation.json','status','passed')]:
        report=load(bundle/'private'/name)
        if report.get(key)!=value or not _bound(report,built):raise ValueError('Missing or unbound environment verification')
    runtime=load(bundle/'private/source-runtime-check.json')
    if not _runtime_reusable(runtime,built,bundle):raise ValueError('Source runtime evidence is missing or stale')
    tasks=read_jsonl(bundle/'public/tasks.jsonl');refs=read_jsonl(bundle/'private/references.jsonl');quality=read_jsonl(bundle/'private/quality.jsonl')
    by_id={r['id']:r for r in refs};q_by_id={q['task_id']:q for q in quality}
    if len(by_id)!=len(refs) or {t['id'] for t in tasks}!=set(by_id) or set(by_id)!=set(q_by_id):raise ValueError('Public/private IDs mismatch')
    for t in tasks:
        validate_public_task(t);r=by_id[t['id']];q=q_by_id[t['id']]
        if t['permitted_tools']!=['list_files','search_code','read_file']:raise ValueError('Source import grants execution')
        if t['user_prompt']!=r['question'] or t['repository']!=r['repository'] or t['split']!=r['split']:raise ValueError('Public/private record mismatch')
        if t['repository']!=env['repository'] or t['environment_id']!=env['id']:raise ValueError('Task environment mismatch')
        if hashlib.sha256(r['reference_answer'].encode()).hexdigest()!=q['reference_sha256']:raise ValueError('Reference hash mismatch')
        if hashlib.sha256(t['user_prompt'].encode()).hexdigest()!=q['question_sha256']:raise ValueError('Question hash mismatch')
    return tasks,refs,quality,env,built


def grading_records(private,reviews,corrections):
    grading=[]
    for ref in private:
        if ref['reference_kind']=='local_claims_draft':
            grading.append({'task_id':ref['id'],'kind':'local_claims_and_behavior_assertions','record':ref['record']})
        elif ref.get('selected_reference_answer'):
            selected_review=corrections[ref['id']] if ref.get('correction_status')=='supported' else reviews[ref['id']]
            grading.append({'task_id':ref['id'],'kind':'source_reference_comparison','reference_answer':ref['selected_reference_answer'],
                            'reference_sha256':ref['selected_reference_sha256'],'grading':{**ref['grading'],'reference_field':'reference_answer'},
                            'reviewed_claims':selected_review['checked_claims'],'verified_evidence':selected_review['verified_evidence'],
                            'human_reviewed':False,'gold_status':'source_reviewed_draft'})
    return grading


def package(root,output):
    root,output=Path(root).resolve(),Path(output).resolve()
    if output.exists() and any(output.iterdir()):raise ValueError('Release exists; choose a new version')
    preparation=load(root/'reports/task-generation-1000/preparation.json');reviews=review_records(root);corrections=correction_records(root)
    public=[];private=[];qualities=[];assertions=[];environments={};bundle_index=[];unready=[];review_cache={};source_artifacts={}
    for entry in preparation['environments']:
        bundle=root/entry['bundle']
        try:tasks,refs,quality,env,built=audit_import_bundle(bundle)
        except (OSError,ValueError,KeyError) as exc:
            unready.append({'bundle':entry['bundle'],'task_count':entry['tasks'],'reason':str(exc)});continue
        for r,q in zip(refs,quality):
            if r['id']!=q['task_id']:raise ValueError('Reference/quality order mismatch')
            source=root/r['provenance']['source_file']
            if file_sha(source)!=r['provenance']['source_file_sha256']:raise ValueError('Upstream artifact changed')
            reference={**r,'reference_kind':'upstream_benchmark_draft'}
            source_destination='private/sources/'+r['provenance']['source_file_sha256']+source.suffix
            source_artifacts[source_destination]=source
            reference['release_source_file']=source_destination
            q={**q,'runtime_status':'passed','bundle':entry['bundle'],'difficulty_status':'not_measured','answerability_scope':'Central-claim reference review; no separate exhaustive question-quality adjudication','citation_integrity_scope':'upstream_original_answer'}
            review=reviews.get(r['id'])
            if not review:raise ValueError('Reference review incomplete: '+r['id'])
            if review:
                review=validate_review(review,r,q,Path(env['snapshot_path']),built['snapshot_files'],review_cache)
                reviews[r['id']]=review
                if review['reference_sha256']!=q['reference_sha256'] or review['question_sha256']!=q['question_sha256']:raise ValueError('Stale semantic review')
                q['semantic_review']=review['status'];q['review_report']=review['review_report']
                q['review_scope']=review['review_scope'];q['review_evidence_integrity']='passed'
                q['reference_admission']='source_reviewed_original_draft' if review['status']=='supported' else 'quarantined_reference'
                if review['status']=='supported':reference['selected_reference_answer']=r['reference_answer']
                if review.get('corrected_reference_answer'):
                    reference['proposed_corrected_reference_answer']=review['corrected_reference_answer']
                    reference['correction_status']='proposed_by_reviewer_not_independently_adjudicated'
                    correction=corrections.get(r['id'])
                    if not correction:raise ValueError('Correction review incomplete: '+r['id'])
                    if correction['original_reference_sha256']!=q['reference_sha256']:raise ValueError('Correction original answer mismatch')
                    corrected=review['corrected_reference_answer']
                    corrected_quality={**q,'reference_sha256':hashlib.sha256(corrected.encode()).hexdigest()}
                    correction=validate_review(correction,{**r,'reference_answer':corrected},corrected_quality,Path(env['snapshot_path']),built['snapshot_files'],review_cache)
                    corrections[r['id']]=correction
                    reference['correction_status']=correction['status'];q['correction_review']=correction['status']
                    if correction['status']=='supported':
                        reference['selected_reference_answer']=corrected
                        q['reference_admission']='independently_reviewed_correction_draft'
                q['grading_reference_available']='selected_reference_answer' in reference
                if q['grading_reference_available']:
                    q['selected_reference_sha256']=hashlib.sha256(reference['selected_reference_answer'].encode()).hexdigest()
                    reference['selected_reference_sha256']=q['selected_reference_sha256']
                reference['grading']={**reference['grading'],'enabled':q['grading_reference_available'],'reference_field':'selected_reference_answer'}
            private.append(reference);qualities.append(q)
        assertions.extend(read_jsonl(bundle/'private/assertions.jsonl'))
        public.extend(tasks);environments[env['id']]={'environment_id':env['id'],'repository':env['repository'],
            'capability':'source_reading','backend':'modal','image_id':built['image_digest'],
            'snapshot_sha256':built['snapshot_sha256'],'recipe_sha256':built['recipe_sha256'],'tool_version':built['tool_version'],
            'runtime_checked':True,'bundle':entry['bundle'],
            'source_inventory_mode':built.get('source_inventory_mode','checkout'),
            'source_scope':built.get('source_scope','Pinned primary repository source snapshot'),
            'symbolic_links':built.get('symbolic_links',{}),'submodules':built.get('submodules',{})}
        bundle_index.append((bundle,env['id'],'import'))
    # Include all twelve previously runnable, independently reviewed local tasks.
    for config_name in ['development-batch.json','generation-batch-v2.json']:
        config=load(root/'examples/dataset_sources'/config_name)
        from .batch import audit_batch
        tasks,refs,envs,bundles=audit_batch(config,root)
        public.extend(tasks)
        for r in refs:
            private.append({'id':r['id'],'reference_kind':'local_claims_draft','record':r,'human_reviewed':False})
            qualities.append({'task_id':r['id'],'runtime_status':'passed','semantic_review':'supported_previous_independent_review',
                              'human_reviewed':False,'training_eligible':False,'reference_origin':'locally_generated','reference_admission':'previously_reviewed_local_draft','grading_reference_available':True})
        for env,(bundle,built,verified,isolation) in zip(envs,bundles):
            assertions.extend(read_jsonl(bundle/'private/assertions.jsonl'))
            environments[env['id']]={'environment_id':env['id'],'repository':env['repository'],'capability':built['capability'],
                'backend':'modal','image_id':built['image_digest'],'snapshot_sha256':built['snapshot_sha256'],
                'recipe_sha256':built['recipe_sha256'],'tool_version':built['tool_version'],'runtime_checked':True,
                'bundle':str(bundle.relative_to(root))}
            bundle_index.append((bundle,env['id'],'local'))
    ids=set();questions={};families={}
    for task in public:
        validate_public_task(task)
        if task['id'] in ids:raise ValueError('Duplicate task ID')
        ids.add(task['id']);q=normalized_question(task['user_prompt'])
        if q in questions:raise ValueError('Duplicate normalized question')
        questions[q]=task['id'];family=task['repository']['family_id'];split=task['split']
        if family in families and families[family]!=split:raise ValueError('Repository-family split leakage')
        families[family]=split
        if split!='development':raise ValueError('This collection must not export training/test data')
    if ids!={r['id'] for r in private} or ids!={q['task_id'] for q in qualities}:raise ValueError('Release IDs mismatch')
    output.mkdir(parents=True,exist_ok=True)
    write_jsonl(output/'public/tasks.jsonl',sorted(public,key=lambda r:r['id']))
    write_jsonl(output/'public/environments.jsonl',sorted(environments.values(),key=lambda r:r['environment_id']))
    write_jsonl(output/'private/references.jsonl',sorted(private,key=lambda r:r['id']))
    write_jsonl(output/'private/quality.jsonl',sorted(qualities,key=lambda r:r['task_id']))
    write_jsonl(output/'private/source-and-behavior-assertions.jsonl',assertions)
    for relative,source in source_artifacts.items():
        dest=output/relative;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,dest)
    write_json(output/'private/reviewed-grading-task-ids.json',sorted(q['task_id'] for q in qualities if q['grading_reference_available']))
    grading=grading_records(private,reviews,corrections)
    write_jsonl(output/'private/grading.jsonl',sorted(grading,key=lambda r:r['task_id']))
    write_json(output/'private/semantic-reviews.json',reviews)
    write_json(output/'private/correction-reviews.json',corrections)
    for name in ['corrected-answer-citations.json','corrected-citation-context-review.json','correction-reassignment-log.json']:
        shutil.copyfile(root/'reports/task-generation-1000'/name,output/'private'/name)
    for bundle,env_id,kind in bundle_index:
        prefix=output/'private/environment-evidence'/env_id
        for relative in ['environment-build/result.json','private/isolation-report.json','private/image-attestation.json']:
            dest=prefix/Path(relative).name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(bundle/relative,dest)
        shutil.copyfile(bundle/'public/environment.json',prefix/'source-environment.json')
        recipe_path=bundle/'environment-build/recipe.json'
        if recipe_path.exists():shutil.copyfile(recipe_path,prefix/'recipe.json')
        if kind=='local':
            shutil.copyfile(bundle/'private/verification-report.json',prefix/'verification-report.json')
        if kind=='import':
            shutil.copyfile(bundle/'private/source-runtime-check.json',prefix/'source-runtime-check.json')
    shutil.copyfile(root/'reports/task-generation-1000/repository-activity.json',output/'attribution-repository-activity.json')
    write_json(output/'split-manifest.json',{'families':families,'policy':'All benchmark imports and original development families remain development. Original benchmark splits preserved in private provenance.'})
    write_json(output/'private/quarantine.json',{'acquisition_or_prompt':preparation['rejections'],'runtime':unready,'references':[{'task_id':q['task_id'],'reason':'Reference failed independent review','review_report':q.get('review_report')} for q in qualities if not q['grading_reference_available']]})
    for name in ['sft/train.jsonl','rl/train.jsonl','preferences/train.jsonl','evaluation/final_test.jsonl']:write_jsonl(output/name,[])
    for source,name in [('benchmark-research/sweqa-LICENSE','SWE-QA-LICENSE.txt'),('training-research/repository-license.txt','SWE-QA-Pro-LICENSE.txt'),
                        ('benchmark-research/sweqa-full-pins.json','SWE-QA-FULL-PINS.json'),('benchmark-research/sweqa-card.txt','SWE-QA-CARD.md'),('training-research/probench-card.md','SWE-QA-Pro-CARD.md')]:
        dest=output/'attribution'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(root/'artifacts/task-generation-1000'/source,dest)
    counts={'total_runnable_tasks':len(public),'imported_source_reading':sum(r['reference_kind']=='upstream_benchmark_draft' for r in private),
            'locally_generated_executable':12,'environments':len(environments),'families':len(families),
            'reference_review_states':dict(Counter(q['semantic_review'] for q in qualities)),
            'reference_admission_states':dict(Counter(q['reference_admission'] for q in qualities)),
            'grading_reference_available':sum(q['grading_reference_available'] for q in qualities),
            'reference_corrections_reviewed':len(corrections),
            'upstream_reference_citation_states':dict(Counter(q.get('citation_integrity',{}).get('status','local_claim_evidence') for q in qualities)),
            'human_approved':0,'training_exported':0,'solver_trajectories_new_this_round':0,'unready_environment_tasks':sum(x['task_count'] for x in unready)}
    card=f'''# Repository Q&A collection\n\n{len(public)} runnable development task records, including benchmark imports and 12 locally generated executable tasks.\n\nRunnable means the required source tools/environment are ready. It does not mean every reference was independently reviewed or a solver answered the question. See private/quality.jsonl for per-task states, known citation diagnostics, and per-record central-claim source review. Upstream references are drafts; original references are retained alongside source-reviewed selections and independently reviewed corrections. References that still fail review are quarantined from grading; runnable tasks remain visible for curation. No human-approved gold or training release is claimed.\n\nOriginal benchmarks: SWE-QA (Apache-2.0), https://github.com/peng-weihan/SWE-QA-Bench ; SWE-QA-Pro (MIT), https://github.com/TIGER-AI-Lab/SWE-QA-Pro . Original prompts and answers, dataset revisions, raw artifact hashes and row lineage are retained. Repository code retains its original licenses in the pinned snapshots. Do not publish benchmark results as an untouched held-out evaluation after using these records for development.\n\nThe solver receives only public/tasks.jsonl and the resolved repository environment. Keep private/ and benchmark source artifacts off solver files/indexes. Source-reading images do not install application dependencies or grant code execution. The original 12 tasks retain their executable images and probes.\n\nUse private/grading.jsonl for selected grading references; unresolved original answers are excluded from that file. Private assertions for imports check source integrity and reference provenance; they do not automatically grade natural-language correctness. Reference comparison requires a semantic judge or review. Canonical accepted SFT/RL/preference and final-test files remain empty.\n\nCounts:\n```json\n{json.dumps(counts,indent=2)}\n```\n'''
    (output/'DATASET_CARD.md').write_text(card)
    manifest={'schema_version':'collection-release-1.0','release_id':output.name,'status':'runnable_development_collection_with_explicit_reference_review_states',
        'counts':counts,'human_reviewed':False,'training_eligible':False,
        'artifacts':{str(p.relative_to(output)):file_sha(p) for p in sorted(output.rglob('*')) if p.is_file()}}
    write_json(output/'manifest.json',manifest)
    return manifest


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path.cwd());p.add_argument('--output',type=Path,default=Path('data/releases/repo-qa-1000-v1'));a=p.parse_args()
    import tempfile
    from .collection_audit import audit_release
    destination=(a.root/a.output).resolve()
    if destination.exists():raise ValueError('Release destination already exists')
    destination.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.collection-release-',dir=destination.parent) as temporary:
        staged=Path(temporary)/destination.name
        r=package(a.root,staged)
        audit_release(staged)
        staged.rename(destination)
    print(json.dumps(r['counts'],indent=2))
