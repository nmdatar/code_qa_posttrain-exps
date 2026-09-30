"""Fetch the final-checkpoint repeat and publish aggregate corrected curves."""
import argparse
import csv
import json
import os
from pathlib import Path
import certifi
os.environ.setdefault('SSL_CERT_FILE', certifi.where())
OUT = Path('reports/efficiency-rl-v2')
NAME = 'efficiency-rl-v2-final-reeval-v1'
ARMS = ['efficiency', 'quality-only']


def remote(status_only=False):
    import modal
    receipt = json.loads((OUT / 'final-reeval-submission.json').read_text())
    sandbox = modal.Sandbox.from_id(receipt['sandbox_id'])
    code = sandbox.poll()
    print('Controller exit:', code, flush=True)
    volume = modal.Volume.from_name(receipt['volume'])
    if status_only and code is None:
        script = """
import json
from pathlib import Path
print(Path('/state/campaigns/efficiency-rl-v2-final-reeval-v1/status.json').read_text())
for p in Path('/state/artifacts/experiments').glob('efficiency-rl-v2-final-reeval-v1-*'):
 if (p/'events.jsonl').exists():
  es=[json.loads(s) for s in (p/'events.jsonl').read_text().splitlines()]
  ts=[e for e in es if e['event']=='trajectory']
  print(json.dumps({'arm':p.name,'tasks':len(ts),'infrastructure_errors':sum(e['termination']=='infrastructure_error' for e in ts),'last_event':es[-1]['event']}))
"""
        p = sandbox.exec('python', '-c', script, timeout=25)
        print(p.stdout.read(), flush=True)
        print(p.stderr.read(), flush=True)
        sandbox.detach()
        return
    sandbox.detach()
    root = 'campaigns/' + NAME
    state = json.loads(b''.join(volume.read_file(root + '/status.json')))
    print(json.dumps(state), flush=True)
    if status_only:
        return
    assert code == 0 and state['status'] == 'complete', 'Wait for both evaluations to finish'
    dest = OUT / 'final-reeval'
    dest.mkdir(exist_ok=True)
    (dest / 'status.json').write_text(json.dumps(state, indent=2))
    for arm in ARMS:
        (dest / (arm + '-report.json')).write_bytes(b''.join(volume.read_file(root + '/' + arm + '-report.json')))
        run_root = 'artifacts/experiments/' + NAME + '-' + arm
        for filename in ['tracking-url.json','events.jsonl']:
            (dest / (arm + '-' + filename)).write_bytes(b''.join(volume.read_file(run_root + '/' + filename)))


def recalculate(publish=False):
    previous = json.loads((OUT / 'iteration-results.json').read_text())
    records = []
    evidence = {}
    all_reports = {}
    for arm in ARMS:
        root = OUT / 'evidence/artifacts/experiments' / ('efficiency-rl-v2-' + arm + '-seed43')
        es = [json.loads(s) for s in (root / 'events.jsonl').read_text().splitlines()]
        original = [json.loads((root / 'evaluations' / Path(e['artifact']).name).read_text()) for e in es if e['event']=='evaluation']
        retry = json.loads((OUT / 'final-reeval' / (arm + '-report.json')).read_text())
        old = original[-1]
        for k in ['checkpoint_id','policy_id','optimizer_step','cohort','data_identity','reward_version']:
            assert retry[k] == old[k], (arm, k)
        assert retry['expected'] == len(retry['results']) == 32
        assert {r['task_id'] for r in retry['results']} == {r['task_id'] for r in old['results']}
        failures = sum(r['termination']=='infrastructure_error' for r in retry['results'])
        assert failures == 0, f'{arm}: infrastructure errors persist; do not publish as corrected'
        all_reports[arm] = original[:-1] + [retry]
        base = original[0]
        for i, e in enumerate(all_reports[arm]):
            def reduction(k):
                return 100*(base[k]-e[k])/base[k] if base.get(k) and e.get(k) is not None else None
            records.append({'arm':arm,'optimizer_updates':e['optimizer_step'],
                'evaluation':'final_reevaluation' if e is retry else 'initial' if i==0 else 'intermediate',
                'avg_output_tokens':e['mean_output_tokens'],'output_token_reduction_pct':reduction('mean_output_tokens'),
                'avg_input_tokens':e['mean_input_tokens'],'input_token_reduction_pct':reduction('mean_input_tokens'),
                'avg_model_action_seconds':e['mean_model_action_seconds'],'model_action_time_reduction_pct':reduction('mean_model_action_seconds'),
                'avg_end_to_end_seconds':e['mean_latency_seconds'],
                'accepted':e['accepted_answers'],'assigned':32,'resolved':e['resolved'],
                'completion_rate':e['completion_rate'], 'infrastructure_errors':sum(r['termination']=='infrastructure_error' for r in e['results'])})
        evidence[arm] = {'original_invalid_final': {'step':old['optimizer_step'], 'avg_output_tokens':old['mean_output_tokens'],
            'infrastructure_errors':sum(r['termination']=='infrastructure_error' for r in old['results'])},
            'replacement_checkpoint_id':retry['checkpoint_id'], 'replacement_tracking':json.loads((OUT / 'final-reeval' / (arm + '-tracking-url.json')).read_text())['url']}
    summary = {'evaluations':records,'provenance':evidence,
               'limitations':'Full32 rerun at exact final saved weights, same frozen science. Rerun concurrency2 and sequential arms, originally8 and parallel arms: runtime is not a controlled speed comparison. Original failed evals remain in raw history; corrected_evaluation curves replace them. Quality is model-assessed and unresolved grades remain explicit.',
               'training_rerun':False}
    (OUT / 'corrected-iteration-results.json').write_text(json.dumps(summary, indent=2))
    with (OUT / 'corrected-iteration-results.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    print(json.dumps({'final_results':[r for r in records if r['evaluation']=='final_reevaluation'],'limitations':summary['limitations']},indent=2),flush=True)
    if not publish:
        return
    import wandb
    receipt = OUT / 'corrected-publication.json'
    assert not receipt.exists(), 'Already published; do not append duplicate curves'
    urls = {}
    for arm in ARMS:
        with wandb.init(entity='nmdatar-harvard-university', project='repository-qa-training',
                        id='efficiency-rl-v2-'+arm+'-seed43', resume='must', dir=str(OUT),
                        settings=wandb.Settings(disable_git=True)) as run:
            run.define_metric('corrected_evaluation/*',step_metric='corrected_optimizer_step')
            for e in all_reports[arm]:
                row={'corrected_optimizer_step':e['optimizer_step']}
                for key in ['mean_output_tokens','mean_input_tokens','mean_model_action_seconds','mean_latency_seconds','mean_compute_units','mean_reward','demonstrated_quality','accepted_rate','scoring_coverage','completion_rate']:
                    if e.get(key) is not None:row['corrected_evaluation/'+key]=e[key]
                run.log(row)
            run.summary['final_evaluation_correction']=evidence[arm]
            run.summary['use_corrected_evaluation_curves']=True
            run.summary['evaluation_correction_limitations']=summary['limitations']
            urls[arm]=run.url
    with wandb.init(entity='nmdatar-harvard-university',project='repository-qa-training',id='efficiency-rl-v2-eval-table',
                    resume='must',dir=str(OUT),settings=wandb.Settings(disable_git=True)) as run:
        columns=list(records[0])
        run.log({'eval_iterations':wandb.Table(columns=columns,data=[[r[k] for k in columns] for r in records])})
        run.summary['final_evaluations_corrected']=True
        run.summary['limitations']=summary['limitations']
        artifact=wandb.Artifact('efficiency-rl-v2-corrected-numeric-results',type='aggregate-evaluation')
        for filename in ['corrected-iteration-results.json','corrected-iteration-results.csv']:
            artifact.add_file(str(OUT / filename),name=filename)
        run.log_artifact(artifact).wait()
        urls['table']=run.url
    receipt.write_text(json.dumps(urls,indent=2))


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--status',action='store_true');p.add_argument('--fetch',action='store_true');p.add_argument('--publish',action='store_true')
    a=p.parse_args()
    if a.status:remote(True)
    else:
        if a.fetch:remote()
        recalculate(a.publish)
