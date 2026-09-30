"""Frozen sampling-only recovery slots; preserve outcomes independently of score.

Original attempts have random episode IDs, not positional IDs. Stable synthetic
slots are assigned from sorted episode IDs and their provenance is retained.
Only absent or infrastructure-failed slots without a final answer are sampled.
"""
import copy
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from .contracts import ConfigurationError
from .storage import digest, read

VERSION = 'training-collection-recovery-v1'


def scientific_identity(config):
    environment = {k:v for k,v in config['environment'].items() if k not in {'release','modal_prices'}}
    return digest({'model':config['model']['base_model'], 'environment':environment,
                   'judge':config.get('judge'), 'limits':config['limits'],
                   'training_reward':config.get('training_reward'), 'temperature':1})


def has_final_answer(trace):
    if isinstance(trace.get('submission'),dict) and trace['submission'].get('text'):
        return True
    for event in trace.get('events',[]):
        if event.get('kind') == 'parsed_action' and 'answer' in event.get('value',{}):
            return True
    generations=trace.get('generations',[])
    if not generations:
        generations=[e for e in trace.get('events',[]) if e.get('kind')=='generation']
    if generations:
        try:
            value=json.loads(generations[-1]['text'])
            return isinstance(value,dict) and 'answer' in value
        except (ValueError,KeyError,TypeError):
            pass
    return False


def disposition(trace):
    # An answer awaiting a judge/cleanup retry must never trigger more sampling.
    if has_final_answer(trace):
        return 'preserve-answer'
    if trace.get('termination') != 'infrastructure_error':
        if trace.get('termination') not in {'completed','agent_error','budget_exhausted'}:
            raise ConfigurationError('Unknown collection termination; inspect before recovery')
        return 'preserve-terminal-outcome'
    return 'replace-infrastructure-failure'


def create_manifest(original, source_config, traces, source):
    """Pure partition construction; caller must verify complete archive first."""
    if source_config.get('group_retries',1) != 0 or source_config['benchmark']['attempts'] != 2:
        raise ConfigurationError('Recovery requires original two-attempt, no-retry benchmark')
    indexed=defaultdict(list)
    ids=set()
    for item in traces:
        trace=item['trace']
        if (trace['task_id'] not in original['task_ids'] or trace['split']!='train'
                or trace['policy_id']!='base:'+source_config['model']['base_model']
                or trace['run_id']!=source_config['run_id'] or trace['episode_id'] in ids):
            raise ConfigurationError('Recovery source task/policy/run/episode mismatch')
        ids.add(trace['episode_id']);indexed[trace['task_id']].append(item)
    retained=[];slots=[];inventory=[]
    for task_id in original['task_ids']:
        items=sorted(indexed[task_id],key=lambda item:item['trace']['episode_id'])
        if len(items)>2:
            raise ConfigurationError('More than two archived attempts; inspect retries before recovery')
        for ordinal in range(2):
            slot={'task_id':task_id,'attempt_slot':ordinal}
            if ordinal>=len(items):
                slots.append({**slot,'reason':'missing','replaces_episode_id':None});continue
            item=items[ordinal];t=item['trace'];action=disposition(t)
            record={**slot,'episode_id':t['episode_id'],'sha256':item['sha256'],
                    'path':item['path'],'termination':t['termination'],
                    'verification_status':(t.get('verification') or {}).get('status'),
                    'disposition':action}
            inventory.append(record)
            if action.startswith('preserve'):
                retained.append(record)
            else:
                slots.append({**slot,'reason':'infrastructure_error','replaces_episode_id':t['episode_id']})
    value={'kind':VERSION,'data_identity':original['data_identity'],'original_manifest':original,
           'source':source,'source_scientific_identity':scientific_identity(source_config),
           'original_attempts':2,'retained':retained,'slots':slots,'source_inventory':inventory,
           'task_ids':[s['task_id'] for s in slots],
           'note':'Stable synthetic attempt slots, not historical launch ordinals. Never replace semantic failures or completed answers awaiting grading.'}
    return {**value,'manifest_hash':digest(value)}


def validate_manifest(manifest,config,data):
    from .benchmark import task_manifest
    if manifest.get('kind')!=VERSION:
        raise ConfigurationError('Unsupported recovery manifest')
    value={k:v for k,v in manifest.items() if k!='manifest_hash'}
    if digest(value)!=manifest['manifest_hash'] or config['benchmark']['manifest_hash']!=manifest['manifest_hash']:
        raise ConfigurationError('Recovery manifest hash mismatch')
    original=manifest['original_manifest']
    if original!=task_manifest(data,len(original['task_ids'])) or manifest['data_identity']!=data['identity']:
        raise ConfigurationError('Recovery original task cohort mismatch')
    if (config.get('execution',{}).get('operation')!='benchmark' or config['benchmark']['attempts']!=1
            or config.get('group_retries',1)!=0 or manifest['original_attempts']!=2
            or scientific_identity(config)!=manifest['source_scientific_identity']):
        raise ConfigurationError('Recovery requires matched sampling-only one-attempt configuration')
    all_slots={(task,ordinal) for task in original['task_ids'] for ordinal in range(2)}
    slot_keys=[(s['task_id'],s['attempt_slot']) for s in manifest['slots']]
    kept_keys=[(s['task_id'],s['attempt_slot']) for s in manifest['retained']]
    if (len(set(slot_keys+kept_keys))!=len(slot_keys+kept_keys) or set(slot_keys+kept_keys)!=all_slots
            or manifest['task_ids']!=[s['task_id'] for s in manifest['slots']]):
        raise ConfigurationError('Recovery slots overlap, duplicate or fail to partition original attempts')
    inventory=manifest['source_inventory'];by_slot={(r['task_id'],r['attempt_slot']):r for r in inventory}
    if len(by_slot)!=len(inventory) or len({r['episode_id'] for r in inventory})!=len(inventory):
        raise ConfigurationError('Duplicate recovery source evidence')
    for record in manifest['retained']:
        if record!=by_slot.get((record['task_id'],record['attempt_slot'])) or not record['disposition'].startswith('preserve'):
            raise ConfigurationError('Recovery retained record missing or changed')
    for slot in manifest['slots']:
        prior=by_slot.get((slot['task_id'],slot['attempt_slot']))
        if slot['reason']=='missing':
            if prior is not None or slot['replaces_episode_id'] is not None:
                raise ConfigurationError('Missing slot already has source evidence')
        elif slot['reason']=='infrastructure_error':
            if (not prior or prior['termination']!='infrastructure_error'
                    or prior['disposition']!='replace-infrastructure-failure'
                    or slot['replaces_episode_id']!=prior['episode_id']):
                raise ConfigurationError('Recovery attempts to replace preserved outcome')
        else:raise ConfigurationError('Unsupported recovery reason')
    if not set(by_slot)<=all_slots:
        raise ConfigurationError('Source inventory outside original cohort')
    if not slot_keys:
        raise ConfigurationError('No missing collection slots; no new sampling needed')
    indexed={t['id']:t for t in data['tasks']}
    return [indexed[s['task_id']] for s in manifest['slots']]


def verified_archive_inventory(run_root, checksums):
    """Require an authoritative completed download manifest and hash every input."""
    run_root=Path(run_root).resolve()
    checked=read(checksums)
    # Archive tools store either an unwrapped path/hash map or a checksums member.
    if 'files' in checked and 'root' in checked:
        if checked.get('controller_exit') is None or any(f.get('parse_error') for f in checked['files']):
            raise ConfigurationError('Archive controller still active or parse errors present')
        checked={str(Path(checked['root']) / f['path']):f['sha256'] for f in checked['files']}
    elif 'checksums' in checked:checked=checked['checksums']
    if not isinstance(checked,dict) or not checked:
        raise ConfigurationError('Missing completed archive checksum inventory')
    normalized={str(Path(k).resolve()):v for k,v in checked.items()}
    def verify(path):
        path=Path(path).resolve();raw=path.read_bytes();actual=hashlib.sha256(raw).hexdigest()
        if normalized.get(str(path))!=actual:
            raise ConfigurationError('Archive input absent from final checksums or changed: '+str(path))
        return raw,actual
    source_raw,source_hash=verify(run_root/'config.json');config=json.loads(source_raw)
    event_raw,event_hash=verify(run_root/'events.jsonl')
    events=[json.loads(line) for line in event_raw.splitlines() if line.strip()]
    expected={e['episode_id'] for e in events if e.get('event')=='trajectory'}
    records=[]
    for path in sorted((run_root/'trajectories').glob('*.json')):
        raw,sha=verify(path);trace=json.loads(raw)
        if path.stem!=trace['episode_id']:raise ConfigurationError('Episode filename mismatch')
        records.append({'path':str(path),'sha256':sha,'trace':trace})
    actual={r['trace']['episode_id'] for r in records}
    inventoried={Path(k).stem for k in normalized if Path(k).parent==run_root/'trajectories'}
    if not expected<=actual or actual!=inventoried:
        raise ConfigurationError('Incomplete archived trajectories; finish download before planning recovery')
    return config,records,{'run_id':config['run_id'],'config_sha256':source_hash,'events_sha256':event_hash,
                           'archive_checksums_sha256':hashlib.sha256(Path(checksums).read_bytes()).hexdigest(),
                           'archive_root':str(run_root),'archived_trajectory_count':len(records)}


def run_recovery_slots(pipeline,tasks,manifest):
    """Persist the precise original attempt slot in each new trajectory group ID."""
    from .concurrency import ordered_map
    pipeline.tracker.context={'phase':'benchmark','optimizer_step':pipeline.state['optimizer_step']}
    policy=pipeline.backend.policy_id
    if len(tasks)!=len(manifest['slots']):raise ConfigurationError('Recovery job/slot mismatch')
    jobs=list(zip(tasks,manifest['slots']))
    def sample(job):
        task,slot=job
        if task['id']!=slot['task_id']:raise ConfigurationError('Recovery task/slot mismatch')
        group_id='recovery-'+manifest['manifest_hash'][:16]+'-'+task['id']+'-'+str(slot['attempt_slot'])
        trace=pipeline.rollout(task,group_id,1)
        if pipeline.backend.policy_id!=policy or trace.policy_id!=policy:
            raise ConfigurationError('Policy changed during recovery')
        return [trace]
    return ordered_map(sample,jobs,pipeline.config.get('concurrency',{}).get('rollouts',1))


def merge_recovery_inventory(manifest,original_records,recovery_records,recovery_config):
    """Return exactly one final outcome per original slot, never failed predecessors."""
    if scientific_identity(recovery_config)!=manifest['source_scientific_identity']:
        raise ConfigurationError('Recovered run scientific identity changed')
    original={r['trace']['episode_id']:r for r in original_records}
    expected_original={r['episode_id']:r for r in manifest['source_inventory']}
    if len(original)!=len(original_records) or set(original)!=set(expected_original):
        raise ConfigurationError('Original archive differs from frozen recovery inventory')
    for episode,item in original.items():
        if item['sha256']!=expected_original[episode]['sha256']:
            raise ConfigurationError('Original recovery source hash changed')
    slots={(s['task_id'],s['attempt_slot']):s for s in manifest['slots']}
    group_slots={'recovery-'+manifest['manifest_hash'][:16]+'-'+task+'-'+str(ordinal):(task,ordinal)
                 for task,ordinal in slots}
    merged=[]
    for record in manifest['retained']:
        item=original[record['episode_id']]
        if disposition(item['trace'])!=record['disposition']:
            raise ConfigurationError('Retained source disposition changed')
        merged.append({**item,'task_id':record['task_id'],'attempt_slot':record['attempt_slot'],'origin':'preserved'})
    seen=set();episode_ids=set(original)
    for item in recovery_records:
        t=item['trace'];slot=group_slots.get(t.get('group_id'))
        if (slot is None or slot in seen or t['task_id']!=slot[0]
                or t['episode_id'] in episode_ids or t['split']!='train'
                or t['policy_id']!='base:'+recovery_config['model']['base_model']
                or t['run_id']!=recovery_config['run_id']):
            raise ConfigurationError('Recovered episode does not uniquely bind an intended slot')
        seen.add(slot);episode_ids.add(t['episode_id'])
        merged.append({**item,'task_id':slot[0],'attempt_slot':slot[1],'origin':'recovery'})
    if seen!=set(slots):raise ConfigurationError('Recovery archive is missing intended slots')
    expected={(task,ordinal) for task in manifest['original_manifest']['task_ids'] for ordinal in range(2)}
    actual={(r['task_id'],r['attempt_slot']) for r in merged}
    if len(merged)!=len(expected) or actual!=expected:
        raise ConfigurationError('Merged collection does not partition original attempts')
    return sorted(merged,key=lambda r:(r['task_id'],r['attempt_slot']))
