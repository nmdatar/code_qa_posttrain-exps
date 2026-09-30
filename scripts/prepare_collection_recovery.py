"""Freeze only missing/infrastructure-failed teacher slots; never submit work."""
import argparse
import copy
import json
from collections import Counter
from pathlib import Path
from training_pipeline.benchmark import task_manifest,selected_tasks
from training_pipeline.collection_recovery import create_manifest,verified_archive_inventory,scientific_identity
from training_pipeline.collection import load_collection
from training_pipeline.config import validate_config
from training_pipeline.remote import operation_estimate,controller_reservation
from training_pipeline.storage import read,atomic_json


def prepare(config_path,run_root,archive_report,destination):
    root=Path(destination)
    if root.exists():raise ValueError('Recovery destination exists; use a fresh immutable output directory')
    local=read(config_path)
    source,inventory,provenance=verified_archive_inventory(run_root,archive_report)
    if scientific_identity(local)!=scientific_identity(source):raise ValueError('Local/source scientific conditions differ')
    original=read(local['benchmark']['task_manifest'])
    if original['manifest_hash']!=source['benchmark']['manifest_hash']:raise ValueError('Original cohort hash differs')
    data=load_collection(local['environment'])
    if original!=task_manifest(data,len(original['task_ids'])):raise ValueError('Original cohort no longer matches training data')
    manifest=create_manifest(original,source,inventory,provenance)
    if not manifest['slots']:raise ValueError('No collection slots need sampling')
    root.mkdir(parents=True)
    mp=root/'recovery-tasks.json';atomic_json(mp,manifest)
    config=copy.deepcopy(local)
    config['run_id']=source['run_id']+'-recovery-v1'
    config['output']='artifacts/experiments/'+config['run_id']
    config['benchmark']={'task_manifest':str(mp),'manifest_hash':manifest['manifest_hash'],'attempts':1}
    # Continue accounting to the authoritative original ledger, not a fresh cap.
    config['spend']['ledger']=source['spend']['ledger'].removeprefix('/state/')
    config['spend']['cap_usd']=source['spend']['cap_usd']
    config['tracking']['notes']='Recovery only:preserve all completed/semantic outcomes;sample frozen missing/infra-failed slots. No optimizer replay.'
    config['tracking']['tags']=[*local['tracking'].get('tags',[]),'collection-recovery-v1']
    validate_config(config);tasks=selected_tasks(config,data)
    cp=root/'teacher-recovery.json';atomic_json(cp,config)
    estimate=operation_estimate(config)
    summary={'original_task_count':len(original['task_ids']),'original_attempt_slots':len(original['task_ids'])*2,
             'archived_trajectories':len(inventory),'retained_attempts':len(manifest['retained']),
             'recovery_episodes':len(tasks),'recovery_reasons':dict(Counter(s['reason'] for s in manifest['slots'])),
             'retained_dispositions':dict(Counter(s['disposition'] for s in manifest['retained'])),
             'retained_unresolved':sum(s['verification_status']!='resolved' for s in manifest['retained']),
             'config':str(cp),'manifest_hash':manifest['manifest_hash'],'estimate':estimate,
             'additional_upper_bound_usd':estimate['upper_estimate_usd']+controller_reservation(config),
             'paid_calls':0,'confirmation_used':False}
    atomic_json(root/'recovery-audit.json',summary)
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True);p.add_argument('--run-root',required=True)
    p.add_argument('--archive-report',required=True);p.add_argument('--destination',required=True)
    args=p.parse_args();print(json.dumps(prepare(args.config,args.run_root,args.archive_report,args.destination),indent=2))
