"""Paired initial-versus-final validation, with missing grades kept explicit."""
import csv
import json
import random
from pathlib import Path

from training_pipeline.storage import atomic_json, read
from training_pipeline.trajectory_viewer import write_report


def main():
    out=Path('reports/fast-grpo-selected-v1')
    root=Path('artifacts/fast-grpo-selected-v1-results/artifacts/experiments/fast-grpo-selected-v1')
    plan=read('configs/experiments/fast-grpo-selected-v1/subset.json')
    events=[json.loads(line) for line in (root/'events.jsonl').read_text().splitlines()]
    evaluations=[e for e in events if e.get('event')=='evaluation']
    if len(evaluations)!=2:
        raise ValueError('Expected completed initial and final evaluations; inspect failure archive')
    reports=[]
    for event in evaluations:
        name=Path(event['artifact']).name
        reports.append(read(root/'evaluations'/name))
    before,after=reports
    assert before['optimizer_step']==0
    expected=set(plan['evaluation_ids'])
    assert all({r['task_id'] for r in report['results']}==expected for report in reports)
    traces=[read(p) for p in (root/'trajectories').glob('*.json')]
    assert all(t['task_id'] in set(plan['training_ids'])|expected for t in traces)
    byid={t['episode_id']:t for t in traces}
    assert len(byid)==len(traces)
    phases={}
    for name,report in zip(('baseline','grpo'),reports):
        rows=report['results']
        values=[r['reward'] for r in rows if r['status']=='resolved']
        phases[name]={'optimizer_step':report['optimizer_step'],'policy_id':report['policy_id'],
            'checkpoint_id':report['checkpoint_id'],'tasks':len(rows),'resolved':len(values),
            'strict_passes':sum(v==1 for v in values),
            'mean_reward_resolved':sum(values)/len(values) if values else None,
            'demonstrated_quality':sum(values)/len(rows),
            'evaluation_seconds':report['wall_seconds']}
    a={r['task_id']:r for r in before['results']};b={r['task_id']:r for r in after['results']}
    paired=[];deltas=[]
    for ident in plan['evaluation_ids']:
        x,y=a[ident],b[ident]
        v=x['reward'] if x['status']=='resolved' else None
        w=y['reward'] if y['status']=='resolved' else None
        delta=w-v if v is not None and w is not None else None
        if delta is not None:deltas.append(delta)
        paired.append({'task_id':ident,'baseline':v,'grpo':w,'delta':delta,
                       'baseline_episode':x['episode_id'],'grpo_episode':y['episode_id']})
    rng=random.Random(42)
    boot=sorted(sum(rng.choices(deltas,k=len(deltas)))/len(deltas) for _ in range(10000)) if deltas else []
    updates=[e for e in events if e.get('event')=='update']
    batches=[e for e in events if e.get('event')=='training_batch']
    ledger=read('artifacts/fast-grpo-selected-v1-results/artifacts/project-budget/fast-grpo-selected-v1/run.json')
    result={'status':'complete','model':'Qwen/Qwen3.5-4B','phases':phases,
            'paired':{'resolved':len(deltas),'improved':sum(v>0 for v in deltas),
                      'worsened':sum(v<0 for v in deltas),'tied':sum(v==0 for v in deltas),
                      'mean_delta':sum(deltas)/len(deltas) if deltas else None,
                      'task_bootstrap_95_interval':[boot[250],boot[9749]] if boot else None},
            'training':{'scheduled_updates':4,'acknowledged_updates':len(updates),'attempted_batches':len(batches),
                        'trajectories':sum(t['split']=='train' for t in traces),
                        'distinct_tasks':len({t['task_id'] for t in traces if t['split']=='train'}),
                        'contributing_trajectories':sum(e.get('contributing_trajectories',0) for e in updates),
                        'excluded_groups':sum(e.get('excluded_groups',0) for e in batches),
                        'zero_variance_groups':sum(e.get('zero_variance_groups',0) for e in batches)},
            'recorded_trajectories':len(traces),'reserved_usd':ledger['reserved_usd'],
            'tracking_failures':sum(e.get('event')=='tracking_failure' for e in events),
            'confirmation_used':False,'live_media_uploads':False,
            'limitations':'Single-seed four-batch screen on 16 validation tasks; no statistical-quality claim. Evaluation uses the unchanged strict grader, whereas GRPO trains on factual coverage. Unknown grades stay null. Same-family uncalibrated judge.'}
    assert len(updates)==after['optimizer_step']
    atomic_json(out/'summary.json',result)
    atomic_json(out/'training-batches.json',batches)
    with (out/'paired-results.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(paired[0]));writer.writeheader();writer.writerows(paired)
    write_report(root,out/'trajectories.html')
    (out/'README.md').write_text(f'''# Fast baseline versus GRPO on selected tasks

Qwen3.5-4B, rank 8, seed 42. Train on 32 tasks drawn by deterministic family-balanced sampling from the selected 274-task score band. All 31 training families are represented. Evaluate 16 fixed tasks from the existing 32-task validation cohort before and after four scheduled GRPO batches; the 85-task confirmation cohort remains untouched. The initial untrained checkpoint supplies the baseline. Each batch contains 8 tasks × 4 attempts; learning rate is 1e-5.

| Metric | Baseline | After GRPO |
|---|---:|---:|
| Optimizer step | 0 | {after['optimizer_step']} |
| Resolved validation grades | {phases['baseline']['resolved']}/16 | {phases['grpo']['resolved']}/16 |
| Strict passes | {phases['baseline']['strict_passes']}/16 | {phases['grpo']['strict_passes']}/16 |
| Demonstrated strict score (all 16 tasks) | {phases['baseline']['demonstrated_quality']:.4f} | {phases['grpo']['demonstrated_quality']:.4f} |

Acknowledged updates: **{len(updates)}** out of 4 scheduled. Recorded trajectories: **{len(traces)}**. Conservative reservations: **${ledger['reserved_usd']:.2f}**, against a $250 cap; these are not invoices.

`paired-results.csv` keeps every validation task and unresolved grades. `summary.json` includes paired differences, the descriptive task-bootstrap interval, contributing trajectories and zero-variance/excluded groups. This tiny, single-seed screen cannot establish a general improvement. The train reward is factual coverage; the validation score also enforces strict correctness/citation gates. The grader is uncalibrated and related to the training-data difficulty model.

Efficiency: 32 rollout workers / 16 judges, no intermediate validation, one shared baseline/training setup, and final-only local viewers. Every raw trajectory was retained; slow live dashboard media uploads were disabled. `trajectories.html` exposes prompts, model outputs, tool calls, observations and final grades. Checkpoint manifests and private grading artifacts are retained in the local archive; remote sampler checkpoints use the configured two-day retention.

The subset manifest pins both task lists and the source selected pool. Held-out leakage, tampering, concurrency, optimizer batching and trace retention checks passed before launch (43 tests). Inputs and the remote code bundle are frozen for reproducibility; no other experiment was modified or resumed.
''')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
