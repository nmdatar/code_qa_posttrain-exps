"""Paired, descriptive SFT evaluation with task and repository resampling."""
import argparse,collections,json,random,statistics
from pathlib import Path
from training_pipeline.storage import atomic_json


def summarize(root, report):
    rows=[]
    for r in report['results']:
        t=json.loads((root/'trajectories'/(r['episode_id']+'.json')).read_text())
        obs=[e['value'] for e in t['events'] if e['kind']=='observation']
        errors=[o for o in obs if isinstance(o,dict) and str(o.get('error','')).startswith('Invalid action:')]
        bad_json=0
        for g in t['generations']:
            try:
                if not isinstance(json.loads(g['text']),dict):bad_json+=1
            except ValueError:bad_json+=1
        rows.append({**r,'generated_actions':len(t['generations']),'invalid_actions':len(errors),'invalid_json_objects':bad_json,
            'error_types':dict(collections.Counter(o.get('detail',o['error']) for o in errors)),
            'tool_failures':sum('exit_code' in o and o['exit_code']!=0 for o in obs if isinstance(o,dict)),
            'clipped_observations':sum('[output truncated]' in m['content'] for m in t['events'][0]['messages']),
            'demonstrated_strict_reward':r['reward'] if r['status']=='resolved' else 0})
    def total(k):return sum(r[k] for r in rows)
    metrics={'tasks':len(rows),'resolved':report['resolved'],'coverage':report['scoring_coverage'],
        'strict_full_passes':sum(r['reward']==1 for r in rows),'strict_quality':report['demonstrated_quality'],
        'completed':sum(r['termination']=='completed' for r in rows),'budget_exhausted':sum(r['termination']=='budget_exhausted' for r in rows),
        'generated_actions':total('generated_actions'),'invalid_actions':total('invalid_actions'),
        'invalid_action_rate':total('invalid_actions')/total('generated_actions'),
        'invalid_json_objects':total('invalid_json_objects'),'tool_failures':total('tool_failures'),'clipped_observations':total('clipped_observations'),
        'output_tokens':sum(r['usage']['output_tokens'] for r in rows),'tool_calls':sum(r['usage']['tool_calls'] for r in rows),
        'wall_seconds':report['wall_seconds']}
    return {'metrics':metrics,'tasks':rows,'policy_id':report['policy_id'],'optimizer_step':report['optimizer_step'],'environment':report['environment'],'reward_version':report['reward_version'],'data_identity':report['data_identity'],'cohort':report['cohort']}


def compare(a,b):
    for key in ['environment','reward_version','data_identity','cohort']:
        if a[key]!=b[key]:raise ValueError('Scientific conditions differ: '+key)
    x={r['task_id']:r for r in a['tasks']};y={r['task_id']:r for r in b['tasks']}
    if x.keys()!=y.keys():raise ValueError('Task mismatch')
    ids=sorted(x);families=collections.defaultdict(list)
    for i in ids:families[x[i]['family_id']].append(i)
    def deltas(sample):
        quality=statistics.mean(y[i]['demonstrated_strict_reward']-x[i]['demonstrated_strict_reward'] for i in sample)
        rate=lambda rows:sum(rows[i]['invalid_actions'] for i in sample)/sum(rows[i]['generated_actions'] for i in sample)
        return quality,rate(y)-rate(x)
    output={'after_minus_before':dict(zip(['strict_quality','invalid_action_rate'],deltas(ids)))}
    for cluster in [False,True]:
        rng=random.Random(42);draws=[[],[]];units=sorted(families) if cluster else ids
        for _ in range(10000):
            selected=rng.choices(units,k=len(units));sample=[i for f in selected for i in families[f]] if cluster else selected
            q,e=deltas(sample);draws[0].append(q);draws[1].append(e)
        output['repository_clustered_95pct' if cluster else 'paired_task_95pct']={k:[sorted(v)[249],sorted(v)[9749]] for k,v in zip(['strict_quality_delta','invalid_action_rate_delta'],draws)}
    am,bm=a['metrics'],b['metrics'];drop=am['invalid_action_rate']-bm['invalid_action_rate']
    output['exploratory_gate']={'coverage':min(am['coverage'],bm['coverage'])>=.95,'absolute_error_reduction':drop>=.02,'relative_error_reduction':am['invalid_action_rate']>0 and drop/am['invalid_action_rate']>=.25,'quality_point_estimate':bm['strict_quality']>=am['strict_quality']-.02}
    output['meets_all_exploratory_gates']=all(output['exploratory_gate'].values())
    output['confirmed_improvement']=False
    return output

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',default='reports/sft-tool-warmup-v1/results.json');a=p.parse_args();root=Path(a.root)
    reports=sorted([json.loads(p.read_text()) for p in (root/'evaluations').glob('*.json')],key=lambda r:r['optimizer_step'])
    assert len(reports)==2 and [r['optimizer_step'] for r in reports]==[0,4]
    before,after=[summarize(root,r) for r in reports]
    result={'before':before,'after':after,'comparison':compare(before,after)}
    other=Path('reports/grpo-v6-parallel/baseline/summary.json')
    if other.exists():
        ext=Path(json.loads(other.read_text())['evaluation_artifact']);r=json.loads(ext.read_text());external=summarize(ext.parent.parent,r)
        result['external_baseline']=external;result['external_comparison']=compare(external,after)
    atomic_json(a.output,result);print(json.dumps({k:v for k,v in result.items() if k.endswith('comparison')},indent=2))
