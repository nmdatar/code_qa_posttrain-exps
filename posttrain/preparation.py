"""Prepare verified source-reading bundles without executing repository code.

Environment build dispatch is separate and requires a shared conservative cost
reservation. A client timeout and ledger estimate are not provider billing caps.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
from .storage import atomic, digest, read


def git(root, *args):
    environment = dict(os.environ, GIT_NO_REPLACE_OBJECTS='1', GIT_TERMINAL_PROMPT='0',
                       GIT_CONFIG_NOSYSTEM='1', GIT_CONFIG_GLOBAL='/dev/null', GIT_LFS_SKIP_SMUDGE='1')
    return subprocess.run(['git','-c','core.hooksPath=/dev/null','-c','core.fsmonitor=false',
                           '-c','core.autocrlf=false','-C',str(root),*args],
                          capture_output=True, check=True, timeout=180, env=environment).stdout


def fetch_snapshot(repository, repos):
    url, commit = repository['url'], repository['commit']
    if not re.fullmatch(r'https://github.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+',url):
        raise ValueError('Pinned public GitHub repository required')
    if not re.fullmatch(r'[a-f0-9]{40}',commit):
        raise ValueError('Full commit required')
    name = url.removeprefix('https://github.com/').replace('/','--')+'-'+commit[:12]
    root=Path(repos).resolve()/name
    root.mkdir(parents=True,exist_ok=True)
    if not (root/'.git').exists():git(root,'init','-q')
    try:git(root,'cat-file','-e',commit+'^{commit}')
    except subprocess.CalledProcessError:
        git(root,'fetch','--no-tags','--depth=1',url,commit)
    # Checkout serves local inspection only. Git objects remain authoritative.
    git(root,'checkout','--detach','--force',commit)
    if git(root,'rev-parse','HEAD').decode().strip()!=commit:raise ValueError('Snapshot mismatch')
    return root


def prepare_candidates(candidates, reviews, repos, output, families=None):
    from dataset_builder.build import SYSTEM_PROMPT, write_jsonl
    from dataset_builder.collection import RECIPE
    from dataset_builder.contracts import validate_bundle
    from dataset_builder.git_source_environment import inventory
    from qa_eval.source import GitSource
    from qa_eval.deterministic import read_evidence
    candidates=[json.loads(x) for x in Path(candidates).read_text().splitlines() if x.strip()]
    decisions={r['task_id']:r for r in read(reviews)['reviews']}
    grouped={}; excluded=[]
    for candidate in candidates:
        task=candidate['task']; family=task['repository']['family_id']
        if families and family not in families:continue
        review=decisions.get(task['id'],{})
        if review.get('status')!='supported' or review.get('task_sha256')!=digest(task):
            excluded.append({'task_id':task['id'],'reason':'missing_or_stale_supported_independent_review'})
            continue
        if task['split']!='train':raise ValueError('Fresh preparation accepts training assignments only')
        grouped.setdefault((family,task['repository']['commit']),[]).append(candidate)
    reports=[]
    for (family,commit), rows in sorted(grouped.items()):
        repository=rows[0]['task']['repository'];root=fetch_snapshot(repository,repos)
        source=GitSource(root,commit);metadata=inventory(root,commit)
        for row in rows:
            for claim in row['task']['claims']:
                for ref in claim['evidence']:read_evidence(source,ref)
        identity={'repository':repository,'recipe':RECIPE,'source_inventory_mode':'git_objects_primary'}
        environment={'id':'env-'+digest(identity)[:20], 'repository':repository,'snapshot_path':str(root),
                     'recipe':RECIPE,'status':'prepared_not_built','source_inventory_mode':'git_objects_primary',
                     'source_scope':'Primary regular Git blobs only; no repository execution.',
                     'symbolic_links':metadata['symbolic_links'],'submodules':metadata['submodules']}
        private=[r['task'] for r in rows]
        public=[{'schema_version':'1.0','id':t['id'],'system_prompt':SYSTEM_PROMPT+' Source reading only.',
                 'user_prompt':t['question'],'repository':t['repository'],'environment_id':environment['id'],
                 'split':t['split'],'permitted_tools':t['permitted_tools'],'budgets':t['budgets']} for t in private]
        validate_bundle(public,private,[environment])
        bundle=Path(output).resolve()/(family.replace('/','--')+'-'+commit[:12])
        source_hash=digest({'tasks':private,'environment':environment})
        if (bundle/'manifest.json').exists():
            old=read(bundle/'manifest.json')
            if old.get('source_spec_sha256')!=source_hash:raise ValueError('Bundle changed; prepare a new version')
            for relative,sha in old['artifacts'].items():
                if hashlib.sha256((bundle/relative).read_bytes()).hexdigest()!=sha:raise ValueError('Prepared artifact changed')
        else:
            write_jsonl(bundle/'public/tasks.jsonl',public)
            atomic(bundle/'public/environment.json',environment)
            write_jsonl(bundle/'private/tasks.jsonl',private)
            write_jsonl(bundle/'private/references.jsonl',rows)
            write_jsonl(bundle/'private/independent-review.jsonl',[decisions[t['id']] for t in private])
            write_jsonl(bundle/'private/assertions.jsonl',[{'task_id':t['id'],'kind':'source_claim_rubric','claims':t['claims'],'executable_behavior_probes':[]} for t in private])
            atomic(bundle/'private/source-spec.json',{'repository':repository,'tasks':private,'environment':RECIPE,'capability':'source_reading'})
            atomic(bundle/'private/source-inventory.json',metadata)
            manifest={'schema_version':'1.0','repository':repository,'environment_id':environment['id'],'split':'train',
                      'task_count':len(rows),'status':'prepared_source_reading','training_eligible':False,'human_reviewed':False,
                      'source_spec_sha256':source_hash,'source_inventory_sha256':digest(metadata['snapshot_files']),
                      'artifacts':{str(p.relative_to(bundle)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(bundle.rglob('*')) if p.is_file()}}
            atomic(bundle/'manifest.json',manifest)
        reports.append({'family':family,'bundle':str(bundle),'tasks':len(rows),'snapshot_files':len(metadata['snapshot_files']),
                        'status':'prepared_not_built','image_id':None,'paid_dispatch':False})
    return {'environments':reports,'prepared_tasks':sum(r['tasks'] for r in reports),'runtime_ready':0,'excluded_tasks':excluded}


def cached_ready(bundle):
    """Validate all existing runtime evidence before avoiding a paid rebuild."""
    from dataset_builder.collection_environments import _ready, _bound, _runtime_reusable
    bundle=Path(bundle)
    try:
        environment=read(bundle/'public/environment.json')
        built=read(bundle/'environment-build/result.json')
        isolation=read(bundle/'private/isolation-report.json')
        attestation=read(bundle/'private/image-attestation.json')
        runtime=read(bundle/'private/source-runtime-check.json')
        inventory=read(bundle/'private/source-inventory.json')
        if not (_ready(built,environment) and built['snapshot_files']==inventory['snapshot_files']):return None
        if not (_bound(isolation,built) and isolation['passed'] and len(isolation['checks'])==2):return None
        if not (_bound(attestation,built) and attestation['status']=='passed' and attestation['observation']['source_files_checked']==len(built['snapshot_files'])):return None
        if not _runtime_reusable(runtime,built,bundle):return None
        return {'status':'passed','image_id':built.get('image_id',built['image_digest']),'cached':True}
    except (OSError,ValueError,KeyError):return None


def _build_worker(bundle, queue):
    from dataset_builder.collection_environments import process_bundle
    try:
        queue.put(process_bundle(bundle))
    except BaseException as exc:
        queue.put({'status':'quarantined','error':type(exc).__name__})


def build_prepared(bundle, ledger, estimate):
    """Reserve a conservative estimate, not a provider-enforced billing limit.

    Parent wall time is capped at five minutes; sandbox operations independently
    carry provider CPU/memory/time limits. Image builds do not expose equivalent
    hard limits. A timeout retains the whole reservation and stops this batch.
    """
    import multiprocessing
    import time
    if not estimate.get('price_source') or estimate.get('upper_usd') != 1.0:
        raise ValueError('Reviewed $1 operation estimate with price source required')
    cached=cached_ready(bundle)
    if cached:return cached
    metadata={**estimate,'provider_enforced_billing_limit':False,'client_timeout_seconds':300,'bundle':str(Path(bundle).resolve())}
    def operation():
        context=multiprocessing.get_context('spawn');queue=context.Queue()
        worker=context.Process(target=_build_worker,args=(str(bundle),queue))
        started=time.monotonic();worker.start();worker.join(300)
        if worker.is_alive():
            worker.terminate();worker.join(10)
            atomic(Path(bundle)/'private/posttrain-build-dispatch.json',
                   {**metadata,'status':'timeout_remote_outcome_unknown','elapsed_seconds':time.monotonic()-started,
                    'note':'Client worker terminated; remote image build cancellation not confirmed. Stop further paid dispatch and inspect Modal.'})
            raise TimeoutError('Environment preparation exceeded five minutes; remote outcome unknown')
        if worker.exitcode!=0:raise RuntimeError('Environment build worker failed')
        result=queue.get(timeout=5)
        atomic(Path(bundle)/'private/posttrain-build-dispatch.json',
               {**metadata,'status':result['status'],'elapsed_seconds':time.monotonic()-started})
        return result
    return ledger.execute('modal_environment',1.0,operation,metadata)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--candidates',default='reports/posttrain/fresh-task-candidates.jsonl')
    parser.add_argument('--reviews',default='reports/posttrain/fresh-task-independent-review.json')
    parser.add_argument('--repos',default='artifacts/posttrain/repos')
    parser.add_argument('--output',default='data/prepared/posttrain-fresh-v1')
    parser.add_argument('--family',action='append')
    parser.add_argument('--build',action='store_true')
    parser.add_argument('--ledger',default='artifacts/posttrain/spending.json')
    parser.add_argument('--report',default='reports/posttrain/environment-preparation.json')
    args=parser.parse_args()
    result=prepare_candidates(args.candidates,args.reviews,args.repos,args.output,args.family)
    atomic(args.report,result)
    if args.build:
        from .storage import BudgetLedger
        ledger=BudgetLedger(args.ledger)
        for environment in result['environments']:
            estimate={'upper_usd':1.0,'price_source':'https://modal.com/pricing',
                      'verified_at':'2026-09-29','cpu_per_core_second':0.00003942,
                      'memory_per_gib_second':0.00000667,
                      'scope':'Conservative whole build/readiness/isolation/attestation/tool-replay estimate; not metered actual billing.'}
            built=build_prepared(environment['bundle'],ledger,estimate)
            environment.update(status=built['status'],image_id=built.get('image_id'),paid_dispatch=not built.get('cached',False))
            result['runtime_ready']=sum(x['status']=='passed' for x in result['environments'])
            result['budget']=ledger.summary()
            atomic(args.report,result)
            if built['status']!='passed':break
        result['runtime_ready']=sum(x['status']=='passed' for x in result['environments'])
        result['budget']=ledger.summary()
    atomic(args.report,result)
    print(json.dumps(result,indent=2))
    if args.build and result['runtime_ready'] != len(result['environments']):
        raise SystemExit(1)


if __name__=='__main__':main()
