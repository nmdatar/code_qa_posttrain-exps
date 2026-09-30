"""Attach native tool tables to original W&B runs without resuming/changing run state."""
import argparse,hashlib,json
from pathlib import Path
import wandb
from training_pipeline.trajectory_viewer import load_traces, summarize, tool_call_rows

parser=argparse.ArgumentParser();parser.add_argument('--limit',type=int);args=parser.parse_args()
root=Path('reports/trajectory-history')
manifest=json.loads((root/'manifest.json').read_text())
runs=json.loads((root/'wandb-runs.json').read_text())
receipt=root/'wandb-upload.json'
receipts=json.loads(receipt.read_text()) if receipt.exists() else {}
api=wandb.Api(timeout=60)
publisher=wandb.init(entity='nmdatar-harvard-university',project='repository-qa-training',id='trajectory-history-viewers-v1',name='Historical trajectory viewers',job_type='trajectory-viewer',resume='allow',dir=str(root),settings=wandb.Settings(disable_git=True))
for target in runs[:args.limit] if args.limit else runs:
    ident=target['id']
    if receipts.get(ident,{}).get('status')=='uploaded':continue
    archives=[a for a in manifest['archives'] if ident in a['tracking_ids'] or ident==a['run_id']]
    records=[];seen=set()
    # Prefer the largest archive; retain additional episodes without overwriting grades.
    for archive in sorted(archives,key=lambda a:a['episodes'],reverse=True):
        for trace,context in load_traces(archive['source']):
            if trace['episode_id'] not in seen:
                records.append((trace,context));seen.add(trace['episode_id'])
    if not records:raise RuntimeError('No records: '+ident)
    tool_columns,tool_rows=tool_call_rows(records)
    cols=['episode_id','task_id','phase','optimizer_step','score','grading_status','termination','tool_calls','errors']
    data=[[summarize(t,c)[k] for k in cols] for t,c in records]
    artifact=wandb.Artifact('trajectory-inspection-'+hashlib.sha256(ident.encode()).hexdigest()[:20],type='trajectory-inspection',
        description='Readable tool calls and responses. Open tool_calls and filter episode_id. Saved archive snapshot; shared run directories may include training and evaluation phases. Original run state and scores are unchanged.',
        metadata={'source_run':ident,'episodes':len(records),'turns':len(tool_rows),'snapshot':True,
                  'archive_sources':[a['source'] for a in archives],
                  'unreadable_snapshot_files':sum(len(a.get('snapshot_read_errors',[])) for a in archives)})
    artifact.add(wandb.Table(columns=tool_columns,data=tool_rows),'tool_calls')
    artifact.add(wandb.Table(columns=cols,data=data),'trajectories')
    # Include portable HTML pages as an alternative, alongside the native tables.
    for n,a in enumerate(archives):
        for i,page in enumerate(a['pages']):artifact.add_file(str(root/page),name=f'viewer/archive-{n+1}-page-{i+1}.html')
    run=api.run('nmdatar-harvard-university/repository-qa-training/'+ident)
    before=run.state
    uploaded=publisher.log_artifact(artifact)
    uploaded.wait()
    uploaded=run.log_artifact(api.artifact(uploaded.qualified_name))
    run.summary['trajectory_inspection_url']=uploaded.url
    run.summary['trajectory_inspection_episodes']=len(records)
    run.summary.update()
    receipts[ident]={'status':'uploaded','episodes':len(records),'turns':len(tool_rows),'artifact':uploaded.qualified_name,'url':uploaded.url,'original_state':before}
    receipt.write_text(json.dumps(receipts,indent=2))
    print(json.dumps({'run':ident,**receipts[ident]}),flush=True)

publisher.log({'historical_runs':wandb.Table(columns=['run_id','episodes','turns','viewer'],data=[[k,v['episodes'],v['turns'],v['url']] for k,v in receipts.items()])})
publisher.finish()
