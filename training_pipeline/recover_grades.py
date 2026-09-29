"""Recover exactly one complete, unused on-policy group after judge syntax repair."""
from dataclasses import asdict
from pathlib import Path
from .storage import read, atomic_json, load_checkpoint, semantic_hash, digest
from .config import inputs
from .contracts import Trajectory, Generation, VerificationResult, ConfigurationError
from .collection import parse_judge, VERSION, modal_reservation
from .strategies import grpo_batch
from .budget import SpendLedger
from .tinker_backend import TinkerBackend
from .orchestrator import Pipeline


def prepare(source, output):
    source=Path(source)
    config=read(source/'config.json')
    events=[__import__('json').loads(x) for x in (source/'events.jsonl').read_text().splitlines()]
    if any(e['event']=='update' for e in events) or not any(e['event']=='run_finish' and e['status']=='complete' for e in events):
        raise ConfigurationError('Recovery requires a finished run with no optimizer updates')
    if len(config['stages'])!=1 or config['stages'][0]['max_batches']!=1:
        raise ConfigurationError('Recovery supports exactly one attempted batch')
    checkpoints=[load_checkpoint(p) for p in (source/'checkpoints').glob('ckpt-*.json') if not p.name.endswith('.pending.json')]
    initial=next(c for c in checkpoints if c['state']['stage']==0 and c['state']['optimizer_step']==0)
    data=inputs(config)
    trajectories=[]; evidence=[]
    for path in (source/'trajectories').glob('*.json'):
        item=read(path)
        if item['split']!='train':continue
        if item['policy_id']!=initial['artifacts']['sampler'] or item['experiment_hash']!=semantic_hash(config):
            raise ConfigurationError('Recovery policy or configuration mismatch')
        row=next(r for r in data['tasks'] if r['id']==item['task_id'])
        if item['task_hash']!=digest(row):raise ConfigurationError('Task changed')
        verification=item['verification']
        if verification['status']=='unresolved':
            if verification['reasons']!=['judge_invalid_or_context_overflow:JSONDecodeError']:
                raise ConfigurationError('Not a recoverable judge syntax error')
            raw=read(source/'private'/(item['episode_id']+'.judge-raw.json'))
            if raw['version']!=VERSION or raw['request']['answer']!=item['submission']:
                raise ConfigurationError('Judge response binding changed')
            result,repaired=parse_judge(raw['generation']['text'])
            if not repaired or result['status']!='resolved':raise ConfigurationError('Judge remains unresolved')
            verification=asdict(VerificationResult('resolved',result['score'],VERSION,[result['reason']],
                {'calibrated':False,'cached_judge_syntax_repaired':True}))
            evidence.append({'episode_id':item['episode_id'],'score':result['score'],'raw_artifact':str(source/'private'/(item['episode_id']+'.judge-raw.json'))})
        item['generations']=[Generation(**g) for g in item['generations']]
        item['verification']=VerificationResult(**verification)
        trajectories.append(Trajectory(**item))
    if len(trajectories)!=config['stages'][0]['group_size'] or len({t.group_id for t in trajectories})!=1:
        raise ConfigurationError('Not one complete group')
    rows,stats=grpo_batch([trajectories])
    if not rows:raise ConfigurationError('Recovered group has no contribution')
    config['run_id']=Path(output).parent.name
    config['output']=str(output)
    return config,data,initial,trajectories,rows,stats,evidence


def recover(source, output):
    config,data,initial,trajectories,rows,stats,evidence=prepare(source,output)
    settings=config['spend']; prior=read(settings['ledger'])
    ledger=SpendLedger(settings['ledger'],settings['cap_usd'],prior['prices'],prior['ttl_seconds'])
    l=config['limits']; n=config['evaluation']['max_tasks']
    upper=(ledger.estimate('train',input_tokens=sum(len(r.input_tokens) for r in rows))+
        ledger.estimate('checkpoint',ttl_seconds=config['model']['checkpoint_ttl_seconds'])+
        n*(modal_reservation(config)+l['max_generations']*ledger.estimate('sample',input_tokens=l['context_tokens'],output_tokens=l['max_tokens_per_call'])+
           ledger.estimate('sample',input_tokens=l['context_tokens'],output_tokens=512)))
    if prior['reserved_usd']+upper>ledger.cap:raise ConfigurationError('Recovery exceeds remaining ceiling')
    # Exclusive durable guard: even an ambiguous failure must never reissue this group's update.
    with (Path(source)/'grade-recovery-started.json').open('x') as f:
        __import__('json').dump({'output':str(output),'upper_estimate_usd':upper},f)
    backend=TinkerBackend(config['model'],config['limits'],ledger)
    pipeline=Pipeline(config,backend=backend,data=data)
    pipeline.setup(create_run=True)
    atomic_json(Path(output)/'recovery.json',{'source':str(source),'initial_checkpoint':initial['id'],
        'prior_reserved_usd':prior['reserved_usd'],'upper_estimate_usd':upper,'grade_repairs':evidence,'stats':stats})
    status='failed'
    try:
        backend.create_trainer(config['seed']);backend.load(initial['artifacts'],'resume')
        for key,value in initial['identity'].items():
            if backend.identity.get(key)!=value:raise ConfigurationError('Restored backend identity mismatch')
        picked=pipeline._batch(data['tasks'],1)
        if picked[0]['id']!=trajectories[0].task_id:raise ConfigurationError('Recovered group does not match original cursor')
        for trajectory in trajectories:pipeline.tracker.trajectory(trajectory)
        result=backend.update(rows,'importance_sampling',config['stages'][0]['learning_rate'])
        pipeline.tracker.event('update',optimizer_step=1,recovered_unused_group=True,**result,**stats)
        pipeline.state.update(stage=1,optimizer_step=1,stage_updates=0,stage_batches=0,attempted_batches=1,cursor=0,order=[],needs_fresh_optimizer=False)
        pipeline.last_manifest=initial
        pipeline.commit(); evaluation=pipeline.evaluate()
        status='complete'
        return {'checkpoint':str(pipeline.last_path),'updates':1,'evaluation':evaluation,'stats':stats}
    finally:
        pipeline.tracker.finish(status)
        try:
            if hasattr(pipeline.factory,'close'):pipeline.factory.close()
        finally:backend.close('success' if status=='complete' else 'errored')
