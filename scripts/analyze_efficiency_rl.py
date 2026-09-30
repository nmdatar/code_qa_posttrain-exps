"""Fetch authoritative pilot evidence and publish its matched W&B comparison."""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
import json
from pathlib import Path
import random
import statistics

from training_pipeline.storage import atomic_json

OUT = Path('reports/efficiency-rl-v1')
ARMS = ('quality-only', 'efficiency')


def fetch():
    import modal
    receipt = json.loads(Path('artifacts/efficiency-rl-v1-bundle.submission.json').read_text())
    volume = modal.Volume.from_name(receipt['volume'])
    paths = ['campaigns/'+receipt['bundle_id']+'/status.json']
    for arm in ARMS:
        root = 'artifacts/experiments/efficiency-rl-v1-'+arm+'-seed42'
        paths.extend(root+'/'+p for p in ('events.jsonl', 'tracking-url.json', 'answers.json',
                                        'judge-viewer.html', 'judge-viewer.json', 'trajectory-viewer.html'))
        paths.append('artifacts/project-budget/efficiency-rl-v1/'+arm+'.json')
        paths.append('campaigns/'+receipt['bundle_id']+'/efficiency-rl-v1-'+arm+'-seed42.log')
        for folder in ('evaluations', 'checkpoints'):
            try:
                paths.extend(e.path.lstrip('/') for e in volume.listdir(root+'/'+folder)
                             if e.path.endswith('.json'))
            except FileNotFoundError:
                pass
    def download(path):
        try:
            raw = b''.join(volume.read_file(path))
        except FileNotFoundError:
            return {'path': path, 'missing': True}
        target = OUT/'evidence'/path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
        return {'path': path, 'bytes': len(raw)}
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(download, paths))
    atomic_json(OUT/'downloads.json', results)
    print(json.dumps({'downloaded':sum('bytes' in r for r in results),'missing':sum('missing' in r for r in results)}), flush=True)


def compute(row):
    u = row['usage']
    if any(u.get(k) is None for k in ('input_tokens','output_tokens','tool_seconds')):
        return None
    return u['input_tokens']+2*u['output_tokens']+100*u['tool_seconds']


def accepted(row):
    return row['status'] == 'resolved' and row.get('tier') == 'accepted'


def efficiency_signal(batches, group_size=4):
    """Counterfactual arithmetic on saved rewards, not another training run."""
    def advantages(rewards):
        mean,sd=statistics.fmean(rewards),statistics.pstdev(rewards)
        return [(r-mean)/sd for r in rewards] if sd else [0.]*len(rewards)
    eligible=changed=accepted_tie_broken=0
    for batch in batches:
        rewards=batch['rewards']
        assert len(rewards)%group_size==0
        for start in range(0,len(rewards),group_size):
            shaped=rewards[start:start+group_size]
            quality=[.9 if r>=.9 else r for r in shaped]
            a,b=advantages(shaped),advantages(quality)
            eligible+=1
            different=any(abs(x-y)>1e-9 for x,y in zip(a,b))
            changed+=different
            accepted_tie_broken+=different and all(r>=.9 for r in shaped)
    return {'eligible_groups':eligible,'groups_with_changed_normalized_advantages':changed,
            'all_accepted_ties_broken':accepted_tie_broken,
            'interpretation':'Counterfactual reward arithmetic on the efficiency arm trajectories; not an independently sampled control.'}


def compare(control, treatment):
    a = {r['task_id']:r for r in control['results']}
    b = {r['task_id']:r for r in treatment['results']}
    assert a.keys() == b.keys(), 'Comparison must use identical assigned tasks'
    clusters = defaultdict(list)
    for ident in sorted(a):
        assert a[ident]['family_id'] == b[ident]['family_id']
        clusters[a[ident]['family_id']].append(ident)
    def stats(ids):
        na, nb = sum(accepted(a[i]) for i in ids), sum(accepted(b[i]) for i in ids)
        costs_a, costs_b = [compute(a[i]) for i in ids], [compute(b[i]) for i in ids]
        relative = None
        if na and nb and None not in costs_a and None not in costs_b and sum(costs_a)>0:
            relative = (sum(costs_b)/nb)/(sum(costs_a)/na)-1
        tokens = sum(b[i]['usage']['output_tokens']-a[i]['usage']['output_tokens'] for i in ids)/len(ids)
        return {'acceptance_delta':(nb-na)/len(ids), 'relative_compute_per_acceptance':relative,
                'mean_output_token_delta':tokens}
    observed = stats(list(a))
    rng = random.Random(42)
    keys = sorted(clusters)
    draws = [stats([i for family in rng.choices(keys,k=len(keys)) for i in clusters[family]]) for _ in range(10000)]
    intervals = {}
    undefined = {}
    for k in observed:
        vals = sorted(d[k] for d in draws if d[k] is not None)
        undefined[k] = len(draws)-len(vals)
        # Do not silently condition away zero-acceptance bootstrap samples.
        intervals[k] = [vals[249],vals[9749]] if len(vals)==10000 else None
    return {'tasks':len(a),'repository_clusters':len(keys),'observed':observed,
            'cluster_bootstrap_95':intervals,'undefined_bootstrap_draws':undefined}


def analyze(publish=False):
    summary = {'arms':{}, 'comparisons':{}, 'single_seed':True, 'independent_judge':False,
               'confirmation_used':False, 'actual_billing_usd':None}
    evaluations = {}
    training = {}
    diagnoses = {}
    task_rows = []
    for arm in ARMS:
        root = OUT/'evidence/artifacts/experiments'/('efficiency-rl-v1-'+arm+'-seed42')
        events = [json.loads(x) for x in (root/'events.jsonl').read_text().splitlines()]
        evals = sorted([json.loads(p.read_text()) for p in (root/'evaluations').glob('*.json')],
                       key=lambda e:e['optimizer_step'])
        if len(evals) < 2:
            raise RuntimeError(arm+' has not completed both initial and final evaluation')
        end = [e for e in events if e['event']=='run_finish']
        if not end or end[-1]['status']!='complete':
            raise RuntimeError(arm+' is not a completed run')
        evaluations[arm] = {'initial':evals[0], 'final':evals[-1]}
        answer_table=json.loads((root/'answers.json').read_text())
        answers={r['episode_id']:r for values in answer_table['data']
                 if (r:=dict(zip(answer_table['columns'],values)))}
        initial={r['task_id']:r for r in evals[0]['results']}
        diagnoses[arm]=[]
        for last in evals[-1]['results']:
            first=initial[last['task_id']]
            if accepted(first)!=accepted(last) or last['status']!='resolved' or last['termination']!='completed':
                diagnoses[arm].append({'task_id':last['task_id'],
                    'transition':('lost_acceptance' if accepted(first) and not accepted(last) else
                                  'gained_acceptance' if accepted(last) and not accepted(first) else 'incomplete_or_unresolved'),
                    'initial':{**first,'answer':answers[first['episode_id']]['submission'],
                               'reason':answers[first['episode_id']]['grading_reason']},
                    'final':{**last,'answer':answers[last['episode_id']]['submission'],
                             'reason':answers[last['episode_id']]['grading_reason']}})
        batches = [e for e in events if e['event']=='training_batch']
        training[arm] = batches
        ledger = json.loads((OUT/'evidence/artifacts/project-budget/efficiency-rl-v1'/(arm+'.json')).read_text())
        result = {'acknowledged_updates':sum(e['event']=='update' and e.get('acknowledged',False) for e in events),
                  'attempted_batches':len(batches),'skipped_updates':sum(e['event']=='skipped_update' for e in events),
                  'excluded_groups':sum(e['excluded_groups'] for e in batches),
                  'accepted_training_answers':sum(e.get('accepted_answers',0) for e in batches),
                  # Under the frozen tier formula, only accepted efficiency
                  # rewards can exceed .9. Zip includes only eligible groups.
                  'bonus_bearing_contributing_trajectories':sum(
                      reward > .9 and advantage != 0 for e in batches
                      for reward,advantage in zip(e['rewards'],e['advantages'])),
                  'reserved_usd':ledger['reserved_usd'], 'evaluations':{},
                  'wandb':json.loads((root/'tracking-url.json').read_text())['url']}
        for phase, report in evaluations[arm].items():
            result['evaluations'][phase] = {k:report.get(k) for k in (
                'optimizer_step','expected','resolved','scoring_coverage','completion_rate','demonstrated_quality',
                'accepted_answers','accepted_rate','mean_input_tokens','mean_output_tokens','mean_tool_calls',
                'mean_tool_seconds','mean_compute_units','compute_units_per_accepted_answer',
                'output_tokens_per_accepted_answer','reserved_cost_usd')}
            for r in report['results']:
                task_rows.append({'arm':arm,'phase':phase,'task_id':r['task_id'],'family_id':r['family_id'],
                    'status':r['status'],'tier':r.get('tier'),'accepted':accepted(r),'reward':r['reward'],
                    'input_tokens':r['usage']['input_tokens'],'output_tokens':r['usage']['output_tokens'],
                    'tool_calls':r['usage']['tool_calls'],'tool_seconds':r['usage'].get('tool_seconds'),
                    'compute_units':compute(r),'episode_id':r['episode_id']})
        summary['arms'][arm] = result
        summary['comparisons'][arm+'_before_after'] = compare(evals[0],evals[-1])
    final = compare(evaluations['quality-only']['final'],evaluations['efficiency']['final'])
    summary['efficiency_specific_learning_signal'] = efficiency_signal(training['efficiency'])
    summary['comparisons']['final_efficiency_vs_quality'] = final
    diffs = summary['comparisons']
    summary['difference_in_differences'] = {
        k:diffs['efficiency_before_after']['observed'][k]-diffs['quality-only_before_after']['observed'][k]
        for k in ('acceptance_delta','mean_output_token_delta')}
    ci = final['cluster_bootstrap_95']
    coverage = all(summary['arms'][a]['evaluations']['final']['scoring_coverage']>=.95 for a in ARMS)
    gates = {'scoring_coverage':coverage,
             'quality_noninferiority':ci['acceptance_delta'] is not None and ci['acceptance_delta'][0]>=-.02,
             'compute_reduction':ci['relative_compute_per_acceptance'] is not None and ci['relative_compute_per_acceptance'][1]<-.10}
    summary['gates'] = gates
    summary['decision'] = 'preliminary_signal_requires_replication' if all(gates.values()) else 'no_demonstrated_efficiency_win'
    summary['limitations'] = 'Single seed, 32 selection tasks, same-family model judge with no human calibration. Training bonus may be sparse. Wall time is confounded by parallel service load. Reservations are not invoices. No post-result tuning or confirmation.'
    atomic_json(OUT/'summary.json', summary)
    atomic_json(OUT/'outcome-transitions.json', diagnoses)
    with (OUT/'per-task.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(task_rows[0]));writer.writeheader();writer.writerows(task_rows)
    lines=['\n## Results\n', '**Decision:** '+summary['decision']+'.\n',
           '| Arm | Phase | Accepted | Scored | Output tokens/task | Compute/accepted |',
           '|---|---|---:|---:|---:|---:|']
    for arm in ARMS:
        for phase,e in summary['arms'][arm]['evaluations'].items():
            cost=e['compute_units_per_accepted_answer']
            cost_text=f'{cost:,.1f}' if cost is not None else 'undefined'
            lines.append(f"| {arm} | {phase} | {e['accepted_answers']}/{e['expected']} | {e['resolved']}/{e['expected']} | {e['mean_output_tokens']:.2f} | {cost_text} |")
    lines.append('\nAcknowledged optimizer updates: '+', '.join(
        f"{a}: {summary['arms'][a]['acknowledged_updates']} of {summary['arms'][a]['attempted_batches']} attempted batches" for a in ARMS)+'.')
    lines.append('Recorded combined reservations: $'+format(sum(summary['arms'][a]['reserved_usd'] for a in ARMS),'.2f')+'; not actual provider billing.')
    lines.append('\nSee summary.json for paired repository-cluster intervals, undefined-bootstrap counts and promotion gates; per-task.csv preserves all assigned tasks. '+summary['limitations'])
    text=(OUT/'README.md').read_text().split('\n## Results\n')[0]
    (OUT/'README.md').write_text(text+'\n'.join(lines)+'\n')
    if publish:
        import wandb
        with wandb.init(entity='nmdatar-harvard-university',project='repository-qa-training',
                        id='efficiency-rl-v1-comparison',name='Efficiency RL · matched comparison',
                        group='efficiency-rl-v1',job_type='comparison',dir=str(OUT),
                        config={'design':'matched single-seed GRPO; 16 batches; 32 selection tasks',
                                'reward_bonus_max':.1,'selection_only':True},
                        settings=wandb.Settings(disable_git=True)) as run:
            # This publisher exports aggregate observability only. Per-task
            # rows, answers and diagnostic artifacts remain in the workspace.
            for key,title in [('accepted_rate','Training accepted-answer rate'),
                              ('mean_output_tokens','Training output tokens per attempt'),
                              ('mean_compute_units','Training compute per attempt'),
                              ('mean_accepted_efficiency_bonus','Bonus on accepted training answers'),
                              ('contributing_trajectories','Trajectories contributing to the update')]:
                series = {a:[b for b in training[a] if b.get(key) is not None] for a in ARMS}
                keys = [a for a in ARMS if series[a]]
                if keys:
                    run.log({'learning/'+key:wandb.plot.line_series(
                        xs=[[b['attempted_batches'] for b in series[a]] for a in keys],
                        ys=[[b[key] for b in series[a]] for a in keys],keys=keys,
                        title=title,xname='attempted batch')})
            for key,title in [('accepted_rate','Held-out accepted-answer rate'),
                              ('mean_output_tokens','Held-out output tokens per attempt'),
                              ('compute_units_per_accepted_answer','All-attempt compute per accepted answer'),
                              ('scoring_coverage','Held-out scoring coverage')]:
                data=[[arm+' / '+phase,summary['arms'][arm]['evaluations'][phase][key]]
                      for arm in ARMS for phase in ('initial','final')]
                table=wandb.Table(columns=['condition',key],data=data)
                run.log({'comparison/'+key:wandb.plot.bar(table,'condition',key,title=title)})
            run.summary.update(summary)
            artifact=wandb.Artifact('efficiency-rl-v1-results',type='experiment-comparison')
            for name in ('summary.json',):
                artifact.add_file(str(OUT/name),name=name)
            run.log_artifact(artifact).wait()
            atomic_json(OUT/'wandb-comparison.json',{'url':run.url})
            print(run.url,flush=True)
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fetch',action='store_true');p.add_argument('--publish',action='store_true')
    p.add_argument('--fetch-only',action='store_true')
    args=p.parse_args()
    if args.fetch or args.fetch_only:fetch()
    if not args.fetch_only:analyze(args.publish)
