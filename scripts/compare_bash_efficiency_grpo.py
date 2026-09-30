"""Read matched runs; publish aggregate held-out charts without uploading answers."""
import argparse
import json
import os
from pathlib import Path
import certifi
os.environ.setdefault('SSL_CERT_FILE',certifi.where())
from training_pipeline.storage import atomic_json, read

OUT=Path('reports/bash-efficiency-grpo-15-v1')
ARMS={'correctness-only':'bash-correctness-grpo-15-v1','token-penalty':'bash-efficiency-grpo-15-v1'}

def fetch():
    import modal
    summary={}
    for arm,name in ARMS.items():
        receipt=read('artifacts/'+name+'-bundle.submission.json')
        sb=modal.Sandbox.from_id(receipt['sandbox_id'])
        try:
            exit_code=sb.poll()
            if exit_code is None:
                source='''
import json
from pathlib import Path
root=Path('/state/artifacts/experiments/NAME')
events=[]
p=root/'events.jsonl'
if p.exists():
 for line in p.read_text().splitlines():
  try: events.append(json.loads(line))
  except ValueError: pass
print(json.dumps({'events':[e for e in events if e['event'] in ('evaluation','training_batch','update','run_finish','experiment_stop','tracking_failure')], 'tracking':json.loads((root/'tracking-url.json').read_text()) if (root/'tracking-url.json').exists() else None}))
'''.replace('NAME',name)
                p=sb.exec('python','-c',source,timeout=30)
                raw=p.stdout.read()
                if p.wait()!=0: raise RuntimeError('Remote read failed')
                data=json.loads(raw)
            else:
                volume=modal.Volume.from_name(receipt['volume'])
                events=[json.loads(x) for x in b''.join(volume.read_file('artifacts/experiments/'+name+'/events.jsonl')).splitlines()]
                data={'events':[e for e in events if e['event'] in ('evaluation','training_batch','update','run_finish','experiment_stop','tracking_failure')]}
            atomic_json(OUT/(arm+'-events.json'),data)
        finally: sb.detach()
        es=data['events']
        summary[arm]={'exit_code':exit_code,'updates':sum(e['event']=='update' and e.get('acknowledged',True) for e in es),'batches':sum(e['event']=='training_batch' for e in es),
                      'evaluations':[{k:v for k,v in e.items() if k in ('optimizer_step','mean_correctness_score','mean_full_correctness_rate','mean_output_tokens','mean_input_tokens','mean_tool_calls','mean_latency_seconds','scoring_coverage','completion_rate')} for e in es if e['event']=='evaluation'],
                      'tracking_failures':sum(e['event']=='tracking_failure' for e in es),'last_event':es[-1]['event'] if es else None}
    atomic_json(OUT/'comparison-status.json',summary)
    print(json.dumps(summary,indent=2),flush=True)
    return summary

def publish():
    import wandb
    metrics=[('mean_correctness_score','Eval correctness (unpenalized)'),('mean_full_correctness_rate','Eval fully correct fraction'),('mean_output_tokens','Eval output tokens per task'),('mean_input_tokens','Eval input tokens per task'),('mean_tool_calls','Eval bash calls per task'),('mean_latency_seconds','Eval latency (service load confounded)'),('output_tokens_per_fully_correct_answer','All-attempt output tokens per fully correct answer'),('scoring_coverage','Eval scoring coverage'),('completion_rate','Eval completion rate')]
    series={a:[e for e in read(OUT/(a+'-events.json'))['events'] if e['event']=='evaluation'] for a in ARMS}
    # Older control instrumentation lacks cost/full-correct; derive only from complete aggregate counters.
    for rows in series.values():
        for e in rows:
            rate=e.get('mean_full_correctness_rate')
            if rate and e.get('mean_output_tokens') is not None:
                e['output_tokens_per_fully_correct_answer']=e['mean_output_tokens']/rate
    with wandb.init(entity='nmdatar-harvard-university',project='repository-qa-training',id='bash-efficiency-grpo-15-v1-comparison',resume='allow',name='Bash GRPO: correctness vs token penalty (15 iterations)',group='bash-efficiency-grpo-15-v1',job_type='comparison',dir=str(OUT),config={'reward':'correctness*(1-0.1*min(output_tokens/6000,1))','batch_size':8,'group_size':8,'max_batches':15,'eval_tasks':32,'single_seed':42},settings=wandb.Settings(disable_git=True)) as run:
        for key,title in metrics:
            curves={a:[e for e in es if e.get(key) is not None] for a,es in series.items()}
            arms=[a for a in ARMS if curves[a]]
            if arms:
                run.log({'eval_comparison/'+key:wandb.plot.line_series(xs=[[e['optimizer_step'] for e in curves[a]] for a in arms],ys=[[e[key] for e in curves[a]] for a in arms],keys=arms,title=title,xname='optimizer update')})
        run.summary.update(read(OUT/'comparison-status.json'))
        atomic_json(OUT/'wandb-comparison.json',{'url':run.url})
        print(run.url,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--publish',action='store_true');args=p.parse_args()
    fetch()
    if args.publish: publish()
