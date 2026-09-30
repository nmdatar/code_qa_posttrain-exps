"""Compare frozen GRPO and REINFORCE archives without relabeling judgments."""
import argparse,json,random,statistics
from pathlib import Path


def load(root):
    rows=[json.loads(l) for l in (root/'events.jsonl').read_text().splitlines()]
    evaluations=[r for r in rows if r['event']=='evaluation']
    if len(evaluations)!=2 or rows[-1].get('status')!='complete':raise ValueError('Complete initial/final run required')
    batches=[r for r in rows if r['event']=='training_batch']
    outcomes=[]
    for e in evaluations:
        outcomes.append({r['task_id']:r for r in rows if r['event']=='trajectory' and r['phase']=='evaluation' and r['optimizer_step']==e['optimizer_step']})
    training=[r['task_id'] for r in rows if r['event']=='trajectory' and r['phase']=='training']
    return {'rows':rows,'evaluations':evaluations,'outcomes':outcomes,'batches':batches,'training_tasks':training}


def summarize(grpo,reinforce):
    a,b=load(grpo),load(reinforce)
    from collections import Counter
    if Counter(a['training_tasks'])!=Counter(b['training_tasks']):raise ValueError('Training task/attempt mismatch')
    for key in ['data_identity','environment','reward_version','cohort']:
        if a['evaluations'][0][key]!=b['evaluations'][0][key]:raise ValueError('Unmatched '+key)
    ids=sorted(a['outcomes'][0])
    for run in [a,b]:
        for out in run['outcomes']:
            if sorted(out)!=ids:raise ValueError('Selection task mismatch')
    result={}
    for name,run in [('grpo',a),('reinforce',b)]:
        result[name]={'batches':len(run['batches']),'updates':sum(r['event']=='update' and r.get('acknowledged',False) for r in run['rows']),
          'training_attempts':len(run['training_tasks']),'resolved_training':sum(r['resolved_trajectories'] for r in run['batches']),
          'contributing_trajectories':sum(r['contributing_trajectories'] for r in run['batches']),
          'zero_variance_groups':sum(r['zero_variance_groups'] for r in run['batches']),
          'initial':{k:run['evaluations'][0][k] for k in ['demonstrated_quality','resolved','scoring_coverage']},
          'final':{k:run['evaluations'][-1][k] for k in ['demonstrated_quality','resolved','scoring_coverage','checkpoint_id']}}
        for label,outs in zip(['initial','final'],run['outcomes']):result[name][label]['strict_passes']=sum(r['reward']==1 for r in outs.values())
    credit=lambda run,stage,task:run['outcomes'][stage][task]['reward'] or 0
    diffs=[(credit(b,1,t)-credit(b,0,t))-(credit(a,1,t)-credit(a,0,t)) for t in ids]
    rng=random.Random(42);samples=sorted(statistics.mean(rng.choices(diffs,k=len(diffs))) for _ in range(10000))
    result['difference_in_changes']={'reinforce_minus_grpo':statistics.mean(diffs),'paired_task_bootstrap_95pct':[samples[249],samples[9749]]}
    result['limits']='Sequential single-seed comparison, same nominal LR but different advantage scale. Unknown grades contribute no demonstrated credit, not proof of wrongness. Bootstrap does not capture repeated judge/generation variability. Confirmation untouched.'
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--grpo',required=True,type=Path);p.add_argument('--reinforce',required=True,type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    result=summarize(a.grpo,a.reinforce);a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
