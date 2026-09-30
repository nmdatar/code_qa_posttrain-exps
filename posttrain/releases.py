"""Immutable diagnostic releases: independent machine review is not human admission."""
import hashlib,json,os,shutil,tempfile
from pathlib import Path
from .storage import atomic,read,digest
from .data import read_jsonl,adapt_task,validate_artifacts
from .preparation import cached_ready


def bundle_index(roots):
    result={}
    for root in roots:
        for path in Path(root).rglob('public/tasks.jsonl'):
            bundle=path.parent.parent
            if not (bundle/'environment-build/result.json').exists():continue
            for task in read_jsonl(path):
                old=result.get(task['id'])
                if old and old!=bundle:
                    if read(old/'public/environment.json')['id']!=read(bundle/'public/environment.json')['id']:
                        raise ValueError('Task appears in conflicting environments: '+task['id'])
                result[task['id']]=bundle
    return result


def publish(candidates_path,reviews_path,bundle_roots,output,expected_count,split):
    candidates=read_jsonl(candidates_path);reviews={r['task_id']:r for r in read(reviews_path)['reviews']}
    if len(candidates)!=expected_count:raise ValueError('Candidate count does not match intended release')
    if len({c['task']['id'] for c in candidates})!=len(candidates):raise ValueError('Duplicate candidate')
    bundles=bundle_index(bundle_roots);output=Path(output).resolve()
    if output.exists():raise ValueError('Frozen release already exists')
    output.parent.mkdir(parents=True,exist_ok=True)
    staging=Path(tempfile.mkdtemp(prefix='.'+output.name+'-',dir=output.parent))
    public=[];private=[];approved=[];environments={};bindings={};hashes={}
    try:
        for candidate in sorted(candidates,key=lambda c:c['task']['id']):
            task=candidate['task'];ident=task['id'];review=reviews.get(ident,{})
            if task['split']!=split:raise ValueError('Wrong split')
            if review.get('status')!='supported' or review.get('task_sha256')!=digest(task) or review.get('author')==review.get('reviewer'):
                raise ValueError('Missing independently supported exact rubric: '+ident)
            bundle=bundles.get(ident)
            if not bundle:raise ValueError('No prepared environment for '+ident)
            # Existing imported bundles omit our additional source inventory file;
            # validate their original readiness/isolation/attestation/replay directly.
            if not cached_ready(bundle):
                from dataset_builder.collection_environments import _ready,_bound,_runtime_reusable
                env=read(bundle/'public/environment.json');built=read(bundle/'environment-build/result.json')
                iso=read(bundle/'private/isolation-report.json');att=read(bundle/'private/image-attestation.json')
                runtime=read(bundle/'private/source-runtime-check.json')
                if not (_ready(built,env) and _bound(iso,built) and iso.get('passed') and _bound(att,built) and att.get('status')=='passed' and _runtime_reusable(runtime,built,bundle)):
                    raise ValueError('Environment checks incomplete: '+ident)
            env=read(bundle/'public/environment.json');built=read(bundle/'environment-build/result.json')
            visible=next(t for t in read_jsonl(bundle/'public/tasks.jsonl') if t['id']==ident)
            grading={'task_id':ident,'record':task,'reference_answer':candidate['reference_answer'],'human_reviewed':False,'kind':'independent_machine_reviewed_rubric'}
            adapt_task(visible,grading)
            public.append(visible);private.append(grading)
            approved.append({'task_id':ident,'rubric_sha256':digest(task),'author':review['author'],'reviewer':review['reviewer'],'status':'supported','human_reviewed':False})
            bindings[ident]={'bundle':str(bundle),'source_root':env['snapshot_path'],'environment_id':env['id']}
            if env['id'] not in environments:
                environments[env['id']]={'environment_id':env['id'],'repository':env['repository'],'image_id':built.get('image_id') or built['image_digest'],
                                        'runtime_checked':True,'source_environment_sha256':built['source_environment_sha256'],
                                        'recipe':env['recipe'],'snapshot_sha256':built['snapshot_sha256']}
                for relative in ['public/environment.json','environment-build/result.json','private/isolation-report.json','private/image-attestation.json','private/source-runtime-check.json']:
                    dest=staging/'private/environments'/env['id']/relative;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(bundle/relative,dest)
        for name,rows in [('public/tasks.jsonl',public),('public/environments.jsonl',list(environments.values())),('private/grading.jsonl',private),('private/rubric_reviews.jsonl',approved)]:
            path=staging/name;path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in rows))
        attribution=staging/'attribution';attribution.mkdir(parents=True,exist_ok=True)
        if split=='train':
            for file in ['training-source-candidates.json','training-family-separation.json','fresh-candidate-evidence-audit.json']:
                source=Path('reports/posttrain')/file
                if source.exists():shutil.copyfile(source,attribution/file)
            sources=read('reports/posttrain/training-source-candidates.json')
            for repo in sources['repositories']:
                for license in repo.get('license_artifacts',[]):
                    src=Path(license['path'])
                    if hashlib.sha256(src.read_bytes()).hexdigest()!=license['sha256']:raise ValueError('License artifact changed')
                    dest=attribution/'licenses'/repo['repository'].replace('/','--')/license['source_path']
                    dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dest)
        else:
            roots={candidate['source_release'] for candidate in candidates}
            for root in roots:
                for src in (Path(root)/'attribution').glob('*'):
                    if src.is_file():shutil.copyfile(src,attribution/src.name)
        atomic(staging/'private/candidate-provenance.json',candidates)
        atomic(staging/'private/runtime-bindings.json',bindings)
        atomic(staging/'private/independent-review.json',read(reviews_path))
        (staging/'DATASET_CARD.md').write_text(f'# {output.name}\n\n{len(public)} {split} tasks; {len(environments)} ready source environments.\n\nIndependent machine-reviewed private rubrics; human approval and calibrated rewards remain pending. This is a diagnostic release, not approval for longer training or a claim of improved model quality. Source snapshots, tools and evidence are pinned. No teacher trajectories or SFT examples are fabricated.\n')
        for p in sorted(staging.rglob('*')):
            if p.is_file():hashes[str(p.relative_to(staging))]=hashlib.sha256(p.read_bytes()).hexdigest()
        atomic(staging/'manifest.json',{'schema_version':'1','release_id':output.name,'split':split,'tasks':len(public),'environments':len(environments),'diagnostic_only':True,'human_admitted_tasks':0,'artifacts':hashes})
        validate_artifacts(staging);os.rename(staging,output)
        return {'release':str(output),'tasks':len(public),'environments':len(environments),'diagnostic_only':True,'human_admitted_tasks':0}
    except BaseException:
        shutil.rmtree(staging);raise
