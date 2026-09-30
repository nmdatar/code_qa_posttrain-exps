"""One bounded live check of remote grading, a repository episode and Tinker update."""
import copy
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import time
from .storage import atomic_json, read, digest
from .remote import PACKAGES, verify_bundle, require_remote_worker, controller_reservation


def prepare(output):
    from .config import inputs
    c=read('examples/remote-validation-base.json')
    row=next(r for r in inputs(c)['tasks'] if r['id']=='import-9bf889d14b884ea533ee5165')
    root=Path(output).resolve();root.mkdir(parents=True,exist_ok=False)
    source=Path(__file__).resolve().parents[1]
    for package in PACKAGES:
        shutil.copytree(source/package,root/'code'/package,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    (root/'snapshots').mkdir()
    subprocess.run(['git','-C',row['snapshot_root'],'bundle','create',str(root/'snapshots/source.bundle'),'--all'],check=True,capture_output=True)
    c['run_id']='remote-validation-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    c['output']='/state/artifacts/'+c['run_id']
    c['spend'].update(cap_usd=2.,ledger='/state/artifacts/project-budget/remote-validation.json')
    c['tracking']['mode']='disabled'
    c['model']['checkpoint_ttl_seconds']=3600
    c['execution']['timeout_seconds']=600
    # One short real episode, not a training screen. Only update machinery uses a
    # synthetic positive advantage; never claim policy improvement from this.
    c['limits'].update(max_generations=3,max_output_tokens=1536,latency_seconds=120)
    atomic_json(root/'validation.json',{'row':row})
    m={'schema_version':1,'validation':'remote-grader-update-v1','configs':[c],
       'files':{p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
                for p in root.rglob('*') if p.is_file()}}
    m['bundle_id']=digest(m);atomic_json(root/'bundle.json',m)
    return {'bundle':str(root),'bundle_id':m['bundle_id'],'run_id':c['run_id'],'cap_usd':2}


def worker(bundle='/bundle'):
    from .budget import SpendLedger
    from .collection import CollectionFactory
    from .tinker_backend import TinkerBackend
    from .storage import Tracker
    from .rendering import shifted
    from agent_harness.training_runner import run_episode
    m=verify_bundle(bundle);c=m['configs'][0];require_remote_worker(c)
    root=Path(c['output']);root.mkdir(parents=True,exist_ok=False)
    row=read(Path(bundle)/'validation.json')['row']
    ledger=SpendLedger(c['spend']['ledger'],2,c['spend']['prices'],3600)
    ledger.reserve_external('modal_controller',controller_reservation(c))
    ledger.reserve_external('image_build_and_storage_allowance',.15)
    report={'status':'running','location':'modal-sandbox','started_at':datetime.now(timezone.utc).isoformat(),
            'judge_model':c['judge']['base_model'],'policy_model':c['model']['base_model'],
            'grading_version':c['environment']['grading_version'],'cases':[],
            'optimizer_updates':0,'quality_improvement_claim':False,'cap_usd':2}
    factory=backend=tracker=None
    def save():
        report['reserved_usd']=read(ledger.path)['reserved_usd'];report['actual_billing_usd']=None
        atomic_json(root/'validation-report.json',report)
        subprocess.run(['sync','/state'],check=True)
    try:
        target='/tmp/validation-source'
        subprocess.run(['git','clone','--bare',str(Path(bundle)/'snapshots/source.bundle'),target],check=True,capture_output=True)
        atomic_json(root/'snapshot-map.json',{row['snapshot_root']:target})
        os.environ['QA_SNAPSHOT_MAP']=str(root/'snapshot-map.json')
        factory=CollectionFactory(c,root,ledger)
        factory.prepare_judge()
        refs=row['excerpts'];citations=[]
        for i,ref in enumerate(refs):
            citations.append({'id':'src'+str(i+1),**{k:ref[k] for k in ('path','start_line','end_line','file_sha256')}})
        supported=' '.join(claim['text'] for claim in row['rubric']['claims'])
        for case,text in [('supported',supported),('false_extra',supported+' Additionally, every call deletes all files on the computer, including files outside the repository.')]:
            request={'episode_id':case,'question':row['public']['user_prompt'],
                'answer':{'schema_version':'1.0','task_id':row['id'],'text':text,'citations':citations,'diagram':None},
                'rubric':row['rubric'],'answer_evidence':refs,'source_row':row}
            began=time.monotonic();grade=factory.grade(request)
            passed=grade['status']=='resolved' and (grade['score']>=.99 if case=='supported' else grade['score']==0)
            report['cases'].append({'case':case,'passed':passed,'grade':grade,'elapsed_seconds':time.monotonic()-began})
            save();print(case,'passed',passed,flush=True)
        if not all(r['passed'] for r in report['cases']):
            report['status']='grader_failed';save();return
        backend=TinkerBackend(c['model'],c['limits'],ledger)
        tracker=Tracker(root,c['tracking'],c['run_id'],run_config=c)
        trajectory=run_episode(backend,factory,row,c['limits'],run_id=c['run_id'],stage=0,
            group_id='validation',episode_id='live-repository-episode',experiment_hash=digest(c),temperature=1,tracker=tracker)
        report['episode']={'termination':trajectory.termination,'generations':len(trajectory.generations),
                           'verification':asdict(trajectory.verification) if trajectory.verification else None}
        if not trajectory.generations or trajectory.termination=='infrastructure_error':
            report['status']='episode_failed';save();return
        backend.create_trainer(c['seed'])
        # API/gradient/checkpoint check on sampled tokens, explicitly synthetic
        # advantage. This adapter is disposable, never a research candidate.
        g=trajectory.generations[0]
        datum=shifted(g.prompt,g.tokens,[1/len(g.tokens)]*len(g.tokens),g.logprobs)
        report['update']=backend.update([datum],'importance_sampling',1e-5)
        report['optimizer_updates']=1;save()
        artifacts=backend.save(c['run_id']);report['checkpoint']=artifacts;save()
        backend.verify_artifacts(artifacts);backend.use_sampler(artifacts)
        report['checkpoint_verified']=True;report['status']='passed';save()
    except Exception as exc:
        report['status']='failed';report['error_type']=type(exc).__name__
        # Local validation errors only; never log provider exceptions or secrets.
        if isinstance(exc,ValueError):report['error']=str(exc)[:500]
        save();raise
    finally:
        if factory:factory.close()
        if backend:backend.close()
        if tracker:tracker.finish(report['status'])
        save()
