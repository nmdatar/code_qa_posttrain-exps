"""Report the paired correctness-only GRPO pilot without conflating citation scores."""
import hashlib
import csv
import json
from pathlib import Path
from statistics import fmean
from training_pipeline.storage import atomic_json, read
from training_pipeline.trajectory_viewer import load_traces, render_report

RUN = 'bash-correctness-grpo-v1'
OUT = Path('reports') / RUN
ROOT = Path('artifacts') / (RUN+'-results') / 'artifacts/experiments' / RUN

def main():
    events = [json.loads(x) for x in (ROOT/'events.jsonl').read_text().splitlines()]
    evaluations = [e for e in events if e['event']=='evaluation']
    if len(evaluations)!=2: raise ValueError('Expected completed before and after evaluations')
    reports = [read(ROOT/'evaluations'/Path(e['artifact']).name) for e in evaluations]
    plan = read(Path('configs/experiments')/RUN/'subset.json')
    expected = set(plan['evaluation_ids'])
    traces = [read(p) for p in (ROOT/'trajectories').glob('*.json')]
    by_id = {t['episode_id']:t for t in traces}
    assert len(by_id)==len(traces), 'Duplicate episode IDs'
    training=[t for t in traces if t['split']=='train']
    assert set(plan['training_ids']).isdisjoint(expected), 'Training/evaluation overlap'
    assert all(t['task_id'] in set(plan['training_ids']) for t in training)
    for t in traces:
        v=t.get('verification')
        if v and v['status']=='resolved':
            assert v['diagnostics']['reward_applied']=='correctness'
            assert v['reward']==v['diagnostics']['correctness_score']
    groups={}
    for t in training: groups.setdefault(t['group_id'],[]).append(t)
    assert all(len(g)==8 and len({t['task_id'] for t in g})==1 for g in groups.values())
    phases = {}
    for label, report in zip(['baseline','posttrained'],reports):
        assert {r['task_id'] for r in report['results']}==expected
        scored = [r['reward'] for r in report['results'] if r['status']=='resolved']
        diagnostics = [by_id[r['episode_id']]['verification']['diagnostics'] for r in report['results']]
        citations = [d['citation_score'] for d in diagnostics if d.get('citation_score') is not None]
        phases[label] = dict(optimizer_step=report['optimizer_step'],tasks=len(expected),resolved=len(scored),
            correctness_mean=fmean(scored) if scored else None, full_correctness=sum(x==1 for x in scored),
            citation_mean=fmean(citations) if citations else None,citation_measurement_count=len(citations),
            checkpoint_id=report['checkpoint_id'],
            tool_calls_mean=fmean(by_id[r['episode_id']]['usage']['tool_calls'] for r in report['results']),
            zero_tool_attempts=sum(by_id[r['episode_id']]['usage']['tool_calls']==0 for r in report['results']))
    pairs=[]
    left,right=({r['task_id']:r for r in report['results']} for report in reports)
    for ident in plan['evaluation_ids']:
        a,b=left[ident],right[ident]
        x=a['reward'] if a['status']=='resolved' else None
        y=b['reward'] if b['status']=='resolved' else None
        pairs.append(dict(task_id=ident,before=x,after=y,delta=y-x if x is not None and y is not None else None))
    matched=[p for p in pairs if p['delta'] is not None]
    deltas=[p['delta'] for p in matched]
    batches=[e for e in events if e['event']=='training_batch']
    updates=[e for e in events if e['event']=='update']
    assert len(updates)==reports[-1]['optimizer_step']
    assert len(batches)==3 and len(training)==192
    ledger=read(Path('artifacts')/(RUN+'-results')/'artifacts/project-budget'/RUN/'run.json')
    result=dict(phases=phases,paired=dict(resolved=len(deltas),mean_delta=fmean(deltas) if deltas else None,
        improved=sum(d>0 for d in deltas),worsened=sum(d<0 for d in deltas),tied=sum(d==0 for d in deltas),
        before_mean=fmean(p['before'] for p in matched) if matched else None,
        after_mean=fmean(p['after'] for p in matched) if matched else None),
        training=dict(attempted_batches=len(batches),optimizer_updates=len(updates),
        attempts=sum(t['split']=='train' for t in traces),
        distinct_tasks=len({t['task_id'] for t in traces if t['split']=='train'}),
        excluded_groups=sum(e.get('excluded_groups',0) for e in batches),
        zero_variance_groups=sum(e.get('zero_variance_groups',0) for e in batches)),
        reservation_usd=ledger['reserved_usd'],tracking_failures=sum(e['event']=='tracking_failure' for e in events),
        limitations='Single seed, three-batch pilot on 32 validation tasks; required-fact coverage is the objective. Citation diagnostics and extra-claim penalties do not affect reward. Reservations are not provider invoices.')
    atomic_json(OUT/'summary.json',result)
    atomic_json(OUT/'training-batches.json',batches)
    with (OUT/'paired-results.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(pairs[0]));w.writeheader();w.writerows(pairs)
    bundle=Path('artifacts')/(RUN+'-bundle')
    rel='inputs/0/release/private/grading.jsonl'
    raw=(bundle/rel).read_bytes()
    assert hashlib.sha256(raw).hexdigest()==read(bundle/'bundle.json')['files'][rel]
    refs={r['task_id']:r for r in map(json.loads,raw.decode().splitlines())}
    records=[]
    for trace,context in load_traces(ROOT):
        diag=(trace.get('verification') or {}).get('diagnostics',{})
        facts={c['id']:c for c in (diag.get('training_coverage') or {}).get('claims',[])}
        assertions=[]
        for i,claim in enumerate(refs[trace['task_id']]['reviewed_claims'],1):
            ident='c'+str(i)
            text=claim['claim']+('\nQualification: '+claim['reason'] if claim['verdict']=='qualified' else '')
            assertions.append({'id':ident,'assertion':text,'recorded_correctness_assessment':facts.get(ident),
                               'reference_source':[claim['evidence']]+claim.get('supporting_evidence',[])})
        context={**context,'grading_review':{'required_assertions':assertions,
            'answer_checks':{'citation_score':diag.get('citation_score'),'citation_status':diag.get('citation_status'),
                             'citation_links':(diag.get('semantic') or {}).get('citation_links')}}}
        records.append((trace,context))
    (OUT/'trajectories.html').write_text(render_report(records,'Bash-only correctness GRPO · before, training, after'))
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
