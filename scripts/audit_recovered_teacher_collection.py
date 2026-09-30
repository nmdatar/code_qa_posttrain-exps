"""Offline slot-exact union and unchanged complete-SFT admission after recovery."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from training_pipeline.collection_recovery import verified_archive_inventory,merge_recovery_inventory
from training_pipeline.storage import read,atomic_json,digest
from scripts.prepare_complete_sft_study import prepare


def audit(args):
    output=Path(args.report)
    if output.exists():raise ValueError('Use a fresh merged audit report')
    manifest=read(args.recovery_manifest)
    if digest({k:v for k,v in manifest.items() if k!='manifest_hash'})!=manifest['manifest_hash']:
        raise ValueError('Frozen recovery manifest hash changed')
    original_config,original,original_proof=verified_archive_inventory(args.original_root,args.original_report)
    recovery_config,recovered,recovery_proof=verified_archive_inventory(args.recovery_root,args.recovery_report)
    if original_proof!=manifest['source']:raise ValueError('Original archive provenance differs from recovery freeze')
    if recovery_config['benchmark']['manifest_hash']!=manifest['manifest_hash']:raise ValueError('Recovery run used another task manifest')
    merged=merge_recovery_inventory(manifest,original,recovered,recovery_config)
    rows=[];terminations=Counter();statuses=Counter();strict=0;unresolved=[];reward_total=0
    for item in merged:
        t=item['trace'];v=t.get('verification')or{};d=v.get('diagnostics')or{}
        terminations[t['termination']]+=1;statuses[v.get('status','missing')]+=1
        strict+=d.get('strict_score')==1
        if v.get('status')=='resolved':reward_total+=v['reward']
        else:unresolved.append(t['episode_id'])
        rows.append({k:item[k] for k in ('task_id','attempt_slot','origin','path','sha256')}|{'episode_id':t['episode_id'],'termination':t['termination'],'verification_status':v.get('status'),'strict_score':d.get('strict_score')})
    sft=prepare(args.config,None,args.destination,trajectory_paths=[Path(r['path']) for r in merged])
    result={'kind':'recovered-teacher-slot-union-audit-v1','recovery_manifest_hash':manifest['manifest_hash'],
            'source_proofs':{'original':original_proof,'recovery':recovery_proof},'merged_attempts':len(merged),
            'distinct_training_tasks':len({r['task_id'] for r in merged}),
            'origin_counts':dict(Counter(r['origin'] for r in merged)),
            'excluded_replaced_source_episodes':[s['replaces_episode_id'] for s in manifest['slots'] if s['replaces_episode_id']],
            'termination_counts':dict(terminations),'verification_counts':dict(statuses),
            'resolved_coverage':statuses['resolved']/len(merged),'strict_passing_trajectories':strict,
            'mean_demonstrated_training_reward':reward_total/len(merged),'unresolved_episode_ids':unresolved,
            'slot_rows':rows,'sft_admission':sft,'paid_calls':0,'confirmation_used':False}
    atomic_json(output,result)
    return {k:v for k,v in result.items() if k not in {'slot_rows','excluded_replaced_source_episodes','unresolved_episode_ids','source_proofs'}}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('config','original-root','original-report','recovery-root','recovery-report','recovery-manifest','destination','report'):
        p.add_argument('--'+name,required=True)
    print(json.dumps(audit(p.parse_args()),indent=2))
