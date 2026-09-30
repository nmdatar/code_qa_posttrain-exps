"""Run one frozen, small judge diagnostic without solver rollouts or training."""
import argparse
import time
from pathlib import Path
from training_pipeline.storage import read, atomic_json, digest
from training_pipeline.budget import SpendLedger, current_prices
from training_pipeline.collection import CollectionFactory


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    args=parser.parse_args()
    fixture=read(args.root/'fixtures.json')
    if digest({k:v for k,v in fixture.items() if k!='fixture_hash'}) != fixture['fixture_hash']:
        raise ValueError('Fixture changed')
    if len(fixture['cases']) > 15: raise ValueError('Diagnostic limited to 15 calls')
    if not args.execute: parser.error('--execute required for paid judge calls')
    # Exclusive marker forbids accidental paid replay, including after interruption.
    c=read(args.config)
    c['run_id']=args.root.name;c['output']=str(args.root)
    fresh=current_prices(c['judge']['base_model'], sampling_only=True)
    c['judge']['prices']=fresh
    c['tracking']['mode']='disabled'
    c['spend']['cap_usd']=1.;c['spend']['ledger']=str(args.root/'spend.json')
    j=c['judge'];bound=len(fixture['cases'])*(j['context_tokens']*fresh['prefill']+j['max_tokens']*fresh['sample'])/1e6
    if bound > 1: raise ValueError('Diagnostic exceeds $1 cap')
    atomic_json(args.root/'config.json',c)
    atomic_json(args.root/'estimate.json',{'cap_usd':1,'upper_estimate_usd':bound,'calls':len(fixture['cases']),
        'allocation':'small grader validation authorized in current chat; within existing $5 grading allocation',
        'optimizer_updates':0,'solver_rollouts':0})
    with (args.root/'started.lock').open('x') as f: f.write('one execution only\n')
    ledger=SpendLedger(args.root/'spend.json',1,c['spend']['prices'],c['model']['checkpoint_ttl_seconds'])
    factory=CollectionFactory(c,args.root,ledger)
    results=[];start=time.monotonic()
    try:
        factory.prepare_judge()
        for case in fixture['cases']:
            row={k:v for k,v in case.items() if k!='request'}
            row['case_id']=case['request']['episode_id'];t=time.monotonic()
            try:
                row['grade']=factory.grade(case['request'])
                g=row['grade'];lo,hi=row['expected_range']
                row['within_expected_range']=g['status']=='resolved' and lo<=g['score']<=hi
            except Exception as exc:
                row['error_type']=type(exc).__name__;row['within_expected_range']=False
            row['elapsed_seconds']=time.monotonic()-t
            results.append(row)
            atomic_json(args.root/'results.json',{'fixture_hash':fixture['fixture_hash'],
                'judge_identity':factory.judge.identity,'results':results,'elapsed_seconds':time.monotonic()-start,
                'reserved_usd':read(args.root/'spend.json')['reserved_usd'],'actual_billing_usd':None,'human_reviewed':False})
            print(row['case_id'],row['family_id'],row['kind'],row.get('grade',row.get('error_type')),flush=True)
            if 'error_type' in row and row['error_type'] not in ('ValueError', 'JSONDecodeError', 'KeyError', 'TypeError'): break
    finally:
        factory.close()

if __name__=='__main__': main()
