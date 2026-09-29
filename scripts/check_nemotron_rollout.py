"""One paid repository rollout and frozen-judge check; never creates a trainer."""
import argparse
from dataclasses import asdict
from pathlib import Path
import time
from training_pipeline.config import inputs
from training_pipeline.budget import SpendLedger, current_prices
from training_pipeline.collection import CollectionFactory, modal_reservation
from training_pipeline.storage import read, atomic_json, Tracker, semantic_hash
from training_pipeline.tinker_backend import TinkerBackend
from agent_harness.training_runner import run_episode


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--task-id',default='import-55b0a97c4c4fa3ceab342e86')
    p.add_argument('--execute',action='store_true',help='Explicitly permit this one paid check')
    a=p.parse_args()
    if not a.execute:p.error('Use --execute for the explicitly authorized single check')
    c=read(a.config); data=inputs(c)
    task=next(t for t in data['tasks'] if t['id']==a.task_id)
    root=Path(a.output);root.mkdir(parents=True,exist_ok=False)
    c['run_id']='nemotron-single-rollout-check';c['output']=str(root)
    c['spend']['cap_usd']=5.;c['spend']['ledger']=str(root/'spend.json')
    c['spend']['prices']=current_prices(c['model']['base_model'])
    full_judge_prices=current_prices(c['judge']['base_model'])
    c['judge']['prices']={k:full_judge_prices[k] for k in ('prefill','sample','source','checked_at')}
    c['judge']['prices']['model']=c['judge']['base_model']
    limits=c['limits'];prices=c['spend']['prices'];j=c['judge']
    bound=modal_reservation(c)+limits['max_generations']*(limits['context_tokens']*prices['prefill']+limits['max_tokens_per_call']*prices['sample'])/1e6+(j['context_tokens']*j['prices']['prefill']+j['max_tokens']*j['prices']['sample'])/1e6
    if bound>5:raise ValueError('Single-check reservation exceeds $5')
    atomic_json(root/'config.json',c)
    atomic_json(root/'estimate.json',{'upper_estimate_usd':bound,'cap_usd':5,'episodes':1,'optimizer_updates':0,'checkpoint_saves':0})
    ledger=SpendLedger(root/'spend.json',5,prices,c['model']['checkpoint_ttl_seconds'])
    tracker=Tracker(root,c['tracking'],c['run_id'])
    backend=factory=None;start=time.monotonic()
    try:
        backend=TinkerBackend(c['model'],limits,ledger)
        factory=CollectionFactory(c,root,ledger);factory.prepare_judge()
        trajectory=run_episode(backend,factory,task,limits,run_id=c['run_id'],stage=0,group_id='single-check',episode_id='one-rollout',experiment_hash=semantic_hash(c),temperature=0,tracker=tracker)
        graded=(root/'private/one-rollout.judge.json').exists()
        report={'status':'passed' if graded and trajectory.verification.status=='resolved' else 'unverified',
            'task_id':task['id'],'policy':backend.identity,'judge':factory.judge.identity,
            'live_judge_called':(root/'private/one-rollout.judge-raw.json').exists(),
            'verification':asdict(trajectory.verification),'usage':trajectory.usage,
            'elapsed_seconds':time.monotonic()-start,'optimizer_updates':0,'checkpoint_saves':0,
            'reserved_usd':read(root/'spend.json')['reserved_usd'],'actual_billing_usd':None,
            'note':'One integration check, not judge calibration or an experiment.'}
        atomic_json(root/'report.json',report)
        print(report['status'],report['reserved_usd'],report['elapsed_seconds'])
    finally:
        try:
            if factory:factory.close()
        finally:
            if backend:backend.close()

if __name__=='__main__':main()
