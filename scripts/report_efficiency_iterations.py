"""Numeric evaluation progress only; never render or publish charts."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import json
import os
from pathlib import Path
import certifi
os.environ.setdefault('SSL_CERT_FILE',certifi.where())
OUT=Path('reports/efficiency-rl-v2')
CAMPAIGNS = {
    'artifacts/efficiency-rl-v2-bundle.submission.json': ['quality-only', 'efficiency'],
}
ARMS = [arm for arms in CAMPAIGNS.values() for arm in arms]


def status(receipt_path):
    import modal
    r=json.loads(Path(receipt_path).read_text())
    s=modal.Sandbox.from_id(r['sandbox_id'])
    try:
        code=s.poll()
        print(json.dumps({'receipt': receipt_path, 'controller_exit':code}),flush=True)
        if code is not None:
            v=modal.Volume.from_name(r['volume'])
            print(b''.join(v.read_file('campaigns/'+r['bundle_id']+'/status.json')).decode(),flush=True)
            return
        script='''
import json
from pathlib import Path
for root in sorted(Path('/state/artifacts/experiments').glob('efficiency-rl-v2-*')):
 es=[]
 if (root/'events.jsonl').exists():
  for line in (root/'events.jsonl').read_text().splitlines():
   try:es.append(json.loads(line))
   except ValueError:pass
 metrics=['optimizer_step','accepted_answers','scoring_coverage','mean_output_tokens','mean_input_tokens','mean_model_action_seconds','mean_latency_seconds','completion_rate']
 result={'run':root.name,'batches':sum(e['event']=='training_batch' for e in es),'updates':sum(e['event']=='update' for e in es),'trajectories':sum(e['event']=='trajectory' for e in es),'last_event':es[-1]['event'] if es else None,'evals':[{k:e.get(k) for k in metrics} for e in es if e['event']=='evaluation']}
 if (root/'tracking-url.json').exists():result['tracking']=json.loads((root/'tracking-url.json').read_text())
 print(json.dumps(result))
'''
        p=s.exec('python','-c',script,timeout=30)
        print(p.stdout.read(),flush=True);print(p.stderr.read(),flush=True)
    finally:s.detach()


def fetch(receipt_path, arms):
    import modal
    receipt=json.loads(Path(receipt_path).read_text())
    v=modal.Volume.from_name(receipt['volume'])
    paths=['campaigns/'+receipt['bundle_id']+'/status.json']
    for arm in arms:
        root='artifacts/experiments/efficiency-rl-v2-'+arm+'-seed43'
        paths.extend(root+'/'+p for p in ['events.jsonl','tracking-url.json'])
        paths.append('artifacts/project-budget/efficiency-rl-v2/'+arm+'.json')
        paths.append('campaigns/'+receipt['bundle_id']+'/efficiency-rl-v2-'+arm+'-seed43.log')
        for folder in ['evaluations','checkpoints']:
            try:paths.extend(e.path.lstrip('/') for e in v.listdir(root+'/'+folder) if e.path.endswith('.json'))
            except FileNotFoundError:pass
    def download(p):
        raw=b''.join(v.read_file(p));target=OUT/'evidence'/p
        target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
        return {'path':p,'bytes':len(raw)}
    with ThreadPoolExecutor(max_workers=8) as pool:results=list(pool.map(download,paths))
    (OUT/(Path(receipt_path).stem+'-downloads.json')).write_text(json.dumps(results,indent=2))
    print('Downloaded',len(results),'evidence files.',flush=True)


def report(publish=False):
    if publish and (OUT/'corrected-publication.json').exists():
        raise RuntimeError('Corrected final evaluations are published; do not replace them with the original failed evals. See corrected-iteration-results.json.')
    records=[];run_info={}
    for arm in ARMS:
        root=OUT/'evidence/artifacts/experiments'/('efficiency-rl-v2-'+arm+'-seed43')
        events=[json.loads(s) for s in (root/'events.jsonl').read_text().splitlines()]
        evaluation_events=[e for e in events if e['event']=='evaluation']
        end=[e for e in events if e['event']=='run_finish']
        finished=bool(end) and end[-1]['status']=='complete'
        updates=sum(e['event']=='update' and e.get('acknowledged',False) for e in events)
        batches=sum(e['event']=='training_batch' for e in events)
        ledger=json.loads((OUT/'evidence/artifacts/project-budget/efficiency-rl-v2'/(arm+'.json')).read_text())
        run_info[arm]={'updates':updates,'batches':batches,'complete':finished,'reserved_usd':ledger['reserved_usd']}
        baseline=None
        for i,event in enumerate(evaluation_events):
            e=json.loads((root/'evaluations'/Path(event['artifact']).name).read_text())
            assert e['expected']==32 and len(e['results'])==32
            assert len({r['task_id'] for r in e['results']})==32
            ids={r['task_id'] for r in e['results']}
            if baseline is None:baseline=e;base_ids=ids
            assert ids==base_ids
            def reduction(key):
                a,b=baseline.get(key),e.get(key)
                return 100*(a-b)/a if a and b is not None else None
            row={'arm':arm,'optimizer_updates':e['optimizer_step'],
                 'evaluation':'initial' if i==0 else 'final' if finished and i==len(evaluation_events)-1 else 'intermediate',
                 'avg_output_tokens':e['mean_output_tokens'],'output_token_reduction_pct':reduction('mean_output_tokens'),
                 'avg_input_tokens':e['mean_input_tokens'],
                 'avg_model_action_seconds':e['mean_model_action_seconds'],
                 'model_action_time_reduction_pct':reduction('mean_model_action_seconds'),
                 'avg_end_to_end_seconds':e['mean_latency_seconds'],
                 'end_to_end_time_reduction_pct':reduction('mean_latency_seconds'),
                 'accepted':e['accepted_answers'],'assigned':32,'resolved':e['resolved'],
                 'completion_rate':e['completion_rate']}
            records.append(row)
    summary={'runs':run_info,'evaluations':records,'reduction_definition':'Positive = fewer tokens / faster than the same arm at step0. Negative = regression.',
             'runtime_definition':'model_action sums measured generation and action seconds, excluding grading/provisioning/cleanup; end_to_end includes all episode work.',
             'limitations':'Runtime is sensitive to concurrent service load. Interpret token changes with acceptance, completion and grading coverage. Selection only, model-assessed grading. Seed43 is a separate repeat of seed42, not a resume. No causal efficiency claim from raw differences alone.',
             'actual_billing_usd':None}
    (OUT/'iteration-results.json').write_text(json.dumps(summary,indent=2))
    if records:
        with (OUT/'iteration-results.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    if publish:
        assert all(x['complete'] for x in run_info.values()),'Publish final table only when all runs are complete'
        import wandb
        with wandb.init(entity='nmdatar-harvard-university',project='repository-qa-training',id='efficiency-rl-v2-eval-table',name='Efficiency iterations · numeric eval results',group='efficiency-rl-v2',job_type='numeric-summary',dir=str(OUT),settings=wandb.Settings(disable_git=True)) as run:
            columns=list(records[0]);run.log({'eval_iterations':wandb.Table(columns=columns,data=[[r[k] for k in columns] for r in records])})
            run.summary['runs']=run_info;run.summary['limitations']=summary['limitations']
            artifact=wandb.Artifact('efficiency-rl-v2-numeric-results',type='aggregate-evaluation')
            for name in ['iteration-results.json','iteration-results.csv']:artifact.add_file(str(OUT/name),name=name)
            run.log_artifact(artifact).wait()
            (OUT/'wandb-numeric-table.json').write_text(json.dumps({'url':run.url}))
            print('W&B numeric table:',run.url,flush=True)
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--status',action='store_true');p.add_argument('--fetch',action='store_true');p.add_argument('--publish',action='store_true')
    args=p.parse_args()
    if args.status:
        for receipt in CAMPAIGNS:status(receipt)
    else:
        if args.fetch:
            for receipt, arms in CAMPAIGNS.items():fetch(receipt, arms)
        report(args.publish)
