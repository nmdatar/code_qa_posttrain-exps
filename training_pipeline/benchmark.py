"""Frozen training-task throughput benchmark; no trainer, optimizer or checkpoint."""
import copy
import hashlib
import time
from collections import defaultdict
from pathlib import Path
from .storage import digest, read, atomic_json
from .config import inputs
from .contracts import ConfigurationError


def task_manifest(data, count=16):
    families=defaultdict(list)
    for task in data['tasks']:
        families[task['family_id']].append(task['id'])
    total=sum(map(len,families.values()))
    if not 0 < count <= total:raise ConfigurationError('Invalid throughput cohort size')
    allocation={f:len(ids)*count//total for f,ids in families.items()}
    for f in sorted(families,key=lambda f:(-(len(families[f])*count % total),f))[:count-sum(allocation.values())]:
        allocation[f]+=1
    ids=[]
    for f in sorted(families):
        ids.extend(sorted(families[f],key=lambda i:(hashlib.sha256(f'throughput-v1|{f}|{i}'.encode()).hexdigest(),i))[:allocation[f]])
    value={'kind':'training-throughput-v1','data_identity':data['identity'],'task_ids':ids,'families':allocation}
    return {**value,'manifest_hash':digest(value)}


def selected_tasks(config, data):
    spec=config.get('benchmark')
    if not spec:raise ConfigurationError('Benchmark needs a pinned training-task manifest')
    manifest=read(spec['task_manifest'])
    from .task_expansion import VERSION as expansion_version, validate as validate_expansion
    if manifest.get('kind') == expansion_version:
        return validate_expansion(manifest, config, data)
    from .collection_recovery import VERSION as recovery_version, validate_manifest
    if manifest.get('kind') == recovery_version:
        return validate_manifest(manifest, config, data)
    expected=task_manifest(data,len(manifest['task_ids']))
    if manifest!=expected or manifest['manifest_hash']!=spec['manifest_hash']:
        raise ConfigurationError('Throughput cohort identity mismatch')
    indexed={t['id']:t for t in data['tasks']}
    tasks=[indexed[i] for i in manifest['task_ids']]
    if any(t['split']!='train' for t in tasks):raise ConfigurationError('Benchmark is training-only')
    return tasks


def benchmark(config, backend=None, factory=None):
    from .orchestrator import Pipeline
    data=inputs(config);tasks=selected_tasks(config,data)
    config=copy.deepcopy(config)
    attempts=config['benchmark']['attempts']
    # Price all requested episodes and full-group retries, without training/storage.
    config['evaluation']={'every':0,'temperature':1,
        'max_tasks':len(tasks)*attempts*(1+config.get('group_retries',1))}
    p=Pipeline(config,backend=backend,factory=factory,data=data)
    status='failed'
    try:
        p.setup(True,job_type='benchmark')
        ledger=getattr(p.backend,'ledger',None)
        before=read(ledger.path)['reserved_usd'] if ledger else None
        started=time.monotonic()
        recovery_manifest=read(config['benchmark']['task_manifest'])
        from .collection_recovery import VERSION as recovery_version, run_recovery_slots
        groups=(run_recovery_slots(p,tasks,recovery_manifest) if recovery_manifest.get('kind')==recovery_version
                else p.groups(tasks,attempts,1))
        elapsed=time.monotonic()-started
        traces=[read(path) for path in (p.root/'trajectories').glob('*.json')]
        resolved=[t for t in traces if t.get('verification',{}).get('status')=='resolved']
        grades=[read(path) for path in (p.root/'private').glob('*.judge-raw.json')]
        report={'kind':'throughput-benchmark','model_identity':p.backend.identity,
            'policy_id':p.backend.policy_id,'reward_version':p.factory.reward_version,
            'data_identity':data['identity'],'task_manifest':config['benchmark'],
            'concurrency':config.get('concurrency',{'rollouts':1,'judges':1}),
            'expected_episodes':len(tasks)*attempts,'attempted_episodes':len(traces),
            'resolved_episodes':len(resolved),'scoring_coverage':len(resolved)/len(traces) if traces else 0,
            'completion_rate':sum(t['termination']=='completed' for t in traces)/len(traces) if traces else 0,
            'excluded_groups':sum(any(t.verification is None or t.verification.status=='unresolved' for t in g) for g in groups),
            'wall_seconds':elapsed,'episodes_per_second':len(traces)/elapsed if elapsed else None,
            'usage':{k:sum(t['usage'].get(k,0) for t in traces) for k in ('input_tokens','output_tokens','tool_calls')},
            'reserved_cost_usd':read(ledger.path)['reserved_usd']-before if ledger else None,
            'timing_seconds':{k:sum(t['usage'].get(k,0) for t in traces) for k in
                ('provision_seconds','generation_seconds','action_seconds','verification_seconds','cleanup_seconds')},
            'judge_queue_seconds':sum(g.get('timing',{}).get('queue_seconds',0) for g in grades),
            'judge_sampling_seconds':sum(g.get('timing',{}).get('sampling_seconds',0) for g in grades),
            'actual_billing_usd':None,'optimizer_updates':0,'checkpoint_saves':0}
        from .shaped_reward import group_signal
        report['reward_signal'] = group_signal(groups)
        atomic_json(p.root/'reward-signal.json',report['reward_signal'])
        atomic_json(p.root/'benchmark.json',report)
        p.tracker.event('throughput_benchmark',**report)
        status='complete';return report
    finally:
        try:
            if p.factory and hasattr(p.factory,'close'):p.factory.close()
        finally:
            if p.backend and hasattr(p.backend,'close'):p.backend.close('success' if status=='complete' else 'errored')
            if p.tracker:p.tracker.finish(status)
