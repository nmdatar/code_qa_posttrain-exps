"""Prepare immutable inputs and submit detached Modal training campaigns.

The laptop only packages inputs/submits/reads artifacts. One named controller
sandbox owns the persistent campaign ledgers; parallel arms run as subprocesses
in that sandbox, so existing process locks protect shared spending state.
"""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from .contracts import ConfigurationError
from .storage import atomic_json, read, digest

APP = 'repository-qa-training'
VOLUME = APP + '-state-v2'
SECRET = APP + '-credentials'
CONTROLLER = 'experiment-controller'
PACKAGES = ('training_pipeline', 'agent_harness', 'qa_eval', 'dataset_builder')


def validate_execution(spec):
    if not isinstance(spec, dict) or set(spec)-{'cohort','checkpoint'} != {'kind','operation','timeout_seconds','cpu','memory_mib'}:
        raise ConfigurationError('Invalid Modal execution settings')
    if spec['kind'] != 'modal' or spec['operation'] not in {'run','fork','benchmark','baseline','evaluate'}:
        raise ConfigurationError('Production experiments require Modal execution')
    if 'cohort' in spec and (spec['operation'] not in ('baseline','evaluate') or spec['cohort'] not in ('selection','confirmation')):
        raise ConfigurationError('Cohort selection is only supported for baseline jobs')
    if spec['operation'] in {'evaluate', 'fork'}:
        if 'checkpoint' not in spec:
            raise ConfigurationError('Evaluation/fork requires a project-relative checkpoint or best pointer')
        relative(spec['checkpoint'])
    elif 'checkpoint' in spec:
        raise ConfigurationError('Checkpoint is only supported for evaluation or fork')
    for name, ceiling in [('timeout_seconds',86400),('cpu',8),('memory_mib',32768)]:
        if type(spec[name]) is not int or not 1 <= spec[name] <= ceiling:
            raise ConfigurationError('Invalid Modal resource limit: '+name)


def require_remote_worker(config):
    if config.get('execution',{}).get('kind') != 'modal' or os.environ.get('QA_MODAL_WORKER') != '1':
        raise ConfigurationError('Collection experiments run only on Modal. Use python -m training_pipeline.remote prepare / submit; local validation remains available.')


def relative(value):
    p = Path(value)
    if p.is_absolute() or not p.parts or '..' in p.parts:
        raise ConfigurationError('Remote outputs and ledgers require safe project-relative paths')
    return p.as_posix()


def resolve_checkpoint_pointer(state_root, checkpoint):
    """Resolve one latest/best pointer, confined to the mounted state volume."""
    root = Path(state_root).resolve()
    path = root / relative(checkpoint)
    if not path.resolve().is_relative_to(root):
        raise ConfigurationError('Checkpoint escapes state volume')
    value = read(path)
    keys = [key for key in ('manifest', 'path') if key in value]
    if keys:
        if len(keys) != 1 or 'manifest_hash' in value:
            raise ConfigurationError('Ambiguous checkpoint pointer')
        target = value[keys[0]]
        if not isinstance(target, str) or not target:
            raise ConfigurationError('Invalid checkpoint pointer')
        path = Path(target) if Path(target).is_absolute() else root / relative(target)
        if not path.resolve().is_relative_to(root):
            raise ConfigurationError('Checkpoint pointer escapes state volume')
    # Reject nested pointers and corrupt manifests before a subprocess starts.
    from .storage import load_checkpoint
    load_checkpoint(path)
    return path


def controller_reservation(config):
    s = config['execution']; p = config['environment']['modal_prices']
    return 2*s['timeout_seconds']*(s['cpu']*p['cpu_core_second']+s['memory_mib']/1024*p['gib_second'])


def operation_estimate(config):
    from .launch import estimate
    c = copy.deepcopy(config)
    if c['execution']['operation']=='benchmark':
        from .benchmark import selected_tasks
        from .config import inputs
        count = len(selected_tasks(c, inputs(c)))
        c['evaluation'] = {'every':0,'temperature':1,'max_tasks':count*c['benchmark']['attempts']*(1+c.get('group_retries',1))}
    return estimate(c, c['spend']['prices'], evaluation_only=c['execution']['operation'] not in {'run','fork'})


def prepare(config_paths, output, parallel_training=False, budget_plan=None, autoresearch=False):
    from .config import inputs, resolve_run_config
    from .budget import ledger_lock
    output = Path(output).resolve()
    if output.exists():
        raise ConfigurationError('Bundle already exists; choose a fresh path')
    configs = [resolve_run_config(read(path)) for path in config_paths]
    if autoresearch and parallel_training:
        raise ConfigurationError('Autoresearch runs one training candidate at a time')
    allocation = read(budget_plan) if budget_plan else None
    if allocation:
        amounts = list(allocation['ledger_caps'].values()) + list(allocation['reserves_usd'].values())
        ceiling = allocation.get('project_ceiling_usd')
        if (type(ceiling) not in (int,float) or not math.isfinite(ceiling) or ceiling <= 0
                or any(type(v) not in (int,float) or not math.isfinite(v) or v < 0 for v in amounts)
                or sum(amounts) > ceiling):
            raise ConfigurationError('Allocation must fit its explicit project ceiling')
        for c in configs:
            if allocation['ledger_caps'].get(c['spend']['ledger']) != c['spend']['cap_usd']:
                raise ConfigurationError('Config does not match the authorized ledger allocation')
    if not configs:
        raise ConfigurationError('At least one config is required')
    if len({c['run_id'] for c in configs}) != len(configs):
        raise ConfigurationError('Duplicate run IDs')
    if parallel_training:
        for c in configs:
            checkpoint = c.get('execution',{}).get('checkpoint')
            if checkpoint and any(Path(checkpoint).is_relative_to(Path(other['output'])) for other in configs):
                raise ConfigurationError('Dependent checkpoint jobs must run sequentially')
    if any(c['run_id']=='auto' or not re.fullmatch(r'[A-Za-z0-9_-]+',c['run_id']) for c in configs):
        raise ConfigurationError('Unsafe run ID')
    resources = configs[0].get('execution')
    for c in configs:
        validate_execution(c.get('execution'))
        if any(c['execution'][k]!=resources[k] for k in ('timeout_seconds','cpu','memory_mib')):
            raise ConfigurationError('One campaign must use identical controller resource limits')
        relative(c['output']); relative(c['spend']['ledger'])
        if Path(c['output']).exists():
            raise ConfigurationError('Run output already exists; version the config before relaunch')
        if c['environment'].get('grading_version') not in ('all-claims-v3', 'all-claims-v4', 'all-claims-v5', 'all-claims-v6', 'all-claims-v7'):
            raise ConfigurationError('Remote production runs require the full-claim grader')
        if c['judge']['base_model']!='Qwen/Qwen3.5-397B-A17B':
            raise ConfigurationError('Initial remote experiments pin the validated Qwen 397B judge')
    datasets = [inputs(c) for c in configs]
    plans = [operation_estimate(c) for c in configs]
    ledgers = {}
    for c, plan in zip(configs,plans):
        key = c['spend']['ledger']
        if key not in ledgers:
            with ledger_lock(key):
                ledgers[key] = read(key) if Path(key).exists() else None
        prior = ledgers[key]['reserved_usd'] if ledgers[key] else 0
        total = prior + sum(p['upper_estimate_usd']+controller_reservation(x)
                           for x,p in zip(configs,plans) if x['spend']['ledger']==key)
        if total > c['spend']['cap_usd']:
            raise ConfigurationError(f'Campaign reservations exceed existing cap for {key}: {total:.2f}')
    output.mkdir(parents=True)
    root = Path(__file__).resolve().parents[1]
    for package in PACKAGES:
        shutil.copytree(root/package, output/'code'/package, ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    snapshots = {}
    for data in datasets:
        for row in data.get('snapshot_tasks', data['tasks'])+data['development']:
            source = row['snapshot_root']
            snapshots.setdefault(source, set()).add(row['public']['repository']['commit'])
    for source, commits in sorted(snapshots.items()):
        key = hashlib.sha256(source.encode()).hexdigest()
        target = output/'snapshots'/(key+'.bundle'); target.parent.mkdir(exist_ok=True)
        # Preserve exact objects/commits. No user credentials, git config, or
        # working-tree files are copied into the remote bundle.
        subprocess.run(['git','-C',source,'bundle','create',str(target),'--all'],check=True,
                       stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,timeout=300)
        for commit in commits:
            subprocess.run(['git','-C',source,'cat-file','-e',commit+'^{commit}'],check=True,
                           stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
    for i,c in enumerate(configs):
        folder = output/'inputs'/str(i); folder.mkdir(parents=True)
        shutil.copytree(c['environment']['release'],folder/'release')
        c['environment']['release'] = f'/bundle/inputs/{i}/release'
        if 'task_subset' in c:
            shutil.copyfile(c['task_subset']['manifest'], folder/'task-subset.json')
            c['task_subset']['manifest'] = f'/bundle/inputs/{i}/task-subset.json'
        if 'supervised' in c:
            source = Path(c['supervised']['manifest']).resolve()
            shutil.copytree(source.parent, folder/'supervised')
            c['supervised']['manifest'] = f'/bundle/inputs/{i}/supervised/{source.name}'
        if 'harness' in c:
            shutil.copyfile(c['harness']['source_manifest'], folder/'harness-sources.json')
            c['harness']['source_manifest'] = f'/bundle/inputs/{i}/harness-sources.json'
        for spec,key in [(c['evaluation'],'cohort_manifest'),(c.get('benchmark',{}),'task_manifest')]:
            if key in spec:
                shutil.copyfile(spec[key],folder/(key+'.json'))
                spec[key] = f'/bundle/inputs/{i}/{key}.json'
        c['output'] = '/state/'+relative(c['output'])
        c['spend']['ledger'] = '/state/'+relative(c['spend']['ledger'])
    hashes = {p.relative_to(output).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(output.rglob('*')) if p.is_file()}
    manifest = {'schema_version':1,'configs':configs,'ledger_seeds':ledgers,
                'snapshots':{s:hashlib.sha256(s.encode()).hexdigest()+'.bundle' for s in snapshots},
                'parallel_training':parallel_training,'files':hashes,
                'note':'Same frozen Qwen judge in both research roles; not independent evaluation.'}
    if autoresearch:
        manifest['autoresearch'] = {'version':'stability-v1','candidate_limit':len(configs)}
    if allocation:
        manifest['budget_allocation'] = allocation
    manifest['bundle_id'] = digest(manifest)
    atomic_json(output/'bundle.json',manifest)
    return {'bundle':str(output),'bundle_id':manifest['bundle_id'],'runs':[c['run_id'] for c in configs],
            'upper_estimate_usd':sum(p['upper_estimate_usd']+controller_reservation(c) for c,p in zip(configs,plans))}


def verify_bundle(root):
    root = Path(root).resolve(); m = read(root/'bundle.json')
    ident = m.pop('bundle_id')
    if digest(m)!=ident:
        raise ConfigurationError('Bundle manifest changed')
    m['bundle_id']=ident
    for name, expected in m['files'].items():
        p = root/relative(name)
        if p.is_symlink() or not p.resolve().is_relative_to(root) or hashlib.sha256(p.read_bytes()).hexdigest()!=expected:
            raise ConfigurationError('Bundle contents changed: '+name)
    actual = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
    if actual != set(m['files'])|{'bundle.json'}:
        raise ConfigurationError('Unexpected files in bundle')
    return m


def isolated_names(ident):
    import base64
    if not re.fullmatch('[a-f0-9]{64}', ident):
        raise ConfigurationError('Invalid bundle ID')
    # Lossless 256-bit identity, within Modal's 64-character name limit.
    token = base64.b32encode(bytes.fromhex(ident)).decode('ascii').rstrip('=').lower()
    return 'qa-exp-' + token, 'qa-state-' + token


def isolated_resources(manifest):
    """Use campaign-private storage: process locks cannot protect shared volumes."""
    ident = manifest['bundle_id']
    if not re.fullmatch('[a-f0-9]{64}', ident):
        raise ConfigurationError('Invalid bundle ID')
    if any(key in manifest for key in ('validation', 'tool_sft_probe',
                                       'atomic_claim_experiment', 'reward_validation')):
        raise ConfigurationError('Special campaigns require the shared controller')
    configs = manifest['configs']
    if not configs or any(c['execution']['operation'] not in ('run', 'baseline', 'benchmark')
                          or 'checkpoint' in c['execution'] for c in configs):
        raise ConfigurationError('Isolated campaigns require fresh independent runs')
    seeds = manifest.get('ledger_seeds', {})
    if any(seed is not None for seed in seeds.values()):
        raise ConfigurationError('Isolated campaigns cannot copy existing ledger history')
    allocation = manifest.get('budget_allocation')
    if not allocation:
        raise ConfigurationError('Isolated campaigns require an explicit fresh budget allocation')
    keys = set()
    for c in configs:
        path = c['spend']['ledger']
        if not path.startswith('/state/'):
            raise ConfigurationError('Expected a prepared ledger path')
        key = relative(path[len('/state/'):]); keys.add(key)
        if allocation['ledger_caps'].get(key) != c['spend']['cap_usd']:
            raise ConfigurationError('Isolated ledger allocation mismatch')
    if set(seeds) != keys or set(allocation['ledger_caps']) != keys:
        raise ConfigurationError('Isolated budget must contain only this campaign ledgers')
    amounts = list(allocation['ledger_caps'].values()) + list(allocation['reserves_usd'].values())
    ceiling = allocation.get('project_ceiling_usd')
    if (type(ceiling) not in (int, float) or not math.isfinite(ceiling) or ceiling <= 0
            or any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in amounts)
            or sum(amounts) > ceiling):
        raise ConfigurationError('Invalid isolated budget ceiling')
    return isolated_names(ident)


def submit(bundle, isolated=False):
    import modal
    from modal.exception import NotFoundError
    bundle=Path(bundle).resolve(); m=verify_bundle(bundle)
    controller, volume_name = isolated_resources(m) if isolated else (CONTROLLER, VOLUME)
    receipt=bundle.parent/(bundle.name+'.submission.json')
    if receipt.exists():
        raise ConfigurationError('Submission already attempted; inspect its receipt instead of retrying')
    try:
        active=modal.Sandbox.from_name(APP,controller)
        running=active.poll() is None
        active.detach()
        if running:
            raise ConfigurationError('A remote campaign is active; wait before submitting another')
    except NotFoundError:
        pass
    image=(modal.Image.debian_slim(python_version='3.11').apt_install('git')
           .pip_install('modal==1.5.5','tinker==0.30.4','transformers==5.17.0','jinja2==3.1.6','wandb==0.30.0')
           .add_local_dir(bundle,'/bundle',copy=True)
           .env({'PYTHONPATH':'/bundle/code','QA_MODAL_WORKER':'1','PYTHONUNBUFFERED':'1',
                 'PYTHONDONTWRITEBYTECODE':'1'}))
    settings=m['configs'][0]['execution']
    # Write intent before dispatch: a lost response is never blindly retried.
    atomic_json(receipt,{'status':'submission_pending','bundle_id':m['bundle_id'],'app':APP,'name':controller,'volume':volume_name,'isolated':isolated})
    sb=modal.Sandbox.create('python','-m','training_pipeline.remote','worker',
        app=modal.App.lookup(APP,create_if_missing=True),name=controller,image=image,
        timeout=settings['timeout_seconds'],cpu=(settings['cpu'],settings['cpu']),
        memory=(settings['memory_mib'],settings['memory_mib']),
        secrets=[modal.Secret.from_name(SECRET,required_keys=['TINKER_API_KEY','MODAL_TOKEN_ID','MODAL_TOKEN_SECRET','WANDB_API_KEY'])],
        volumes={'/state':modal.Volume.from_name(volume_name,create_if_missing=True,version=2)},workdir='/state',
        tags={'bundle_id':m['bundle_id'][:63]})
    result={'status':'submitted','sandbox_id':sb.object_id,'bundle_id':m['bundle_id'],'volume':volume_name,'controller':controller,'isolated':isolated}
    atomic_json(receipt,result)
    sb.detach()
    return result


def reconcile_ledger(remote, seed, allocated_cap=None):
    """Cloud history must extend the imported local history, never reset it."""
    if seed is None:
        return
    if remote is None:
        return
    for k in ('cap','prices','ttl_seconds'):
        if remote[k]!=seed[k]:
            if k == 'cap' and remote[k] == allocated_cap:
                continue
            raise ConfigurationError('Local and cloud ledger settings diverged')
    n=len(seed['reservations'])
    if remote['reservations'][:n]!=seed['reservations'] or remote['reserved_usd']<seed['reserved_usd']:
        raise ConfigurationError('Local and cloud ledger histories diverged; reconcile before launch')


def reallocate_ledger(path, cap, allocation_hash):
    """Explicit campaign allocation changes caps, never spending history."""
    from .budget import ledger_lock
    with ledger_lock(path):
        state = read(path)
        if state['reserved_usd'] > cap:
            raise ConfigurationError('Allocation is below existing reservations')
        if state['cap'] != cap:
            state.setdefault('allocation_history',[]).append({'previous_cap':state['cap'],
                'cap':cap,'allocation_hash':allocation_hash})
            state['cap'] = cap
            atomic_json(path,state)


def worker(bundle='/bundle', state_root='/state', snapshot_root='/tmp/snapshots'):
    if os.environ.get('QA_MODAL_WORKER')!='1':
        raise ConfigurationError('Worker entrypoint is only for the Modal image')
    from concurrent.futures import ThreadPoolExecutor
    from .budget import SpendLedger
    m=verify_bundle(bundle)
    if m.get('validation') == 'archive-restore-v1':
        from .archive_validation import worker as archive_worker
        return archive_worker(bundle)
    if m.get('validation') == 'remote-grader-update-v1':
        from .remote_validation import worker as validate_worker
        return validate_worker(bundle)
    root=Path(state_root)/'campaigns'/m['bundle_id']
    root.mkdir(parents=True,exist_ok=False)
    atomic_json(root/'status.json',{'status':'starting','bundle_id':m['bundle_id']})
    try:
        mapping={}
        for source,name in m['snapshots'].items():
            target=Path(snapshot_root)/Path(name).stem; target.parent.mkdir(parents=True,exist_ok=True)
            subprocess.run(['git','clone','--bare',str(Path(bundle)/'snapshots'/name),str(target)],
                           check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
            mapping[source]=str(target)
        atomic_json(root/'snapshot-map.json',mapping)
        os.environ['QA_SNAPSHOT_MAP']=str(root/'snapshot-map.json')
        # Import existing spending once; never replace cloud reservations.
        for key,seed in m['ledger_seeds'].items():
            path=Path(state_root)/relative(key)
            remote=read(path) if path.exists() else None
            cap = m.get('budget_allocation',{}).get('ledger_caps',{}).get(key)
            reconcile_ledger(remote,seed,cap)
            if remote is None and seed is not None:
                atomic_json(path,seed)
            if cap is not None and path.exists():
                reallocate_ledger(path,cap,digest(m['budget_allocation']))
        plans=[operation_estimate(c) for c in m['configs']]
        for c in m['configs']:
            path=Path(c['spend']['ledger'])
            prior=read(path)['reserved_usd'] if path.exists() else 0
            required=sum(p['upper_estimate_usd']+controller_reservation(x)
                         for x,p in zip(m['configs'],plans) if x['spend']['ledger']==str(path))
            if prior+required>c['spend']['cap_usd']:
                raise ConfigurationError('Cloud campaign exceeds remaining shared ledger balance')
        for c in m['configs']:
            if Path(c['output']).exists():
                raise ConfigurationError('Remote run output already exists')
            ledger=SpendLedger(c['spend']['ledger'],c['spend']['cap_usd'],c['spend']['prices'],
                               c['model']['checkpoint_ttl_seconds'])
            ledger.reserve_external('modal_controller',controller_reservation(c),bundle_id=m['bundle_id'])
        subprocess.run(['sync',str(state_root)],check=True)
        if m.get('autoresearch'):
            from .autoresearch import worker as autoresearch_worker
            state = autoresearch_worker(m['configs'], root)
            atomic_json(root/'status.json', {'status':state['status'], 'autoresearch':True,
                'incumbent':state.get('incumbent'), 'stopped_run':state.get('stopped_run'),
                'state':str(root/'autoresearch-state.json')})
            return
        if m.get('tool_sft_probe'):
            from .tool_sft_probe import execute
            if len(m['configs']) != 1:
                raise ConfigurationError('Paired interface probe requires one controller config')
            execute(m['configs'][0], m, root)
            atomic_json(root/'status.json', {'status':'complete','output':m['configs'][0]['output']})
            return
        if m.get('atomic_claim_experiment'):
            from .atomic_claim_experiment import execute
            c = m['configs'][0]
            if read(c['spend']['ledger'])['reserved_usd'] + m['atomic_claim_experiment']['reservation_limit_usd'] > c['spend']['cap_usd']:
                raise ConfigurationError('Atomic claim experiment exceeds shared cap')
            execute(c, m, root)
            atomic_json(root/'status.json', {'status':'experiment_finished','output':c['output']})
            return
        if m.get('reward_validation'):
            from .reward_validation import run as validate_reward
            if len(m['configs']) != 1 or m['configs'][0]['execution']['operation'] != 'benchmark':
                raise ConfigurationError('Reward calibration must precede one benchmark only')
            c = m['configs'][0]
            if read(c['spend']['ledger'])['reserved_usd'] + plans[0]['upper_estimate_usd'] + 5. > c['spend']['cap_usd']:
                raise ConfigurationError('Calibration plus benchmark exceeds shared cap')
            if not validate_reward(c, read(Path(bundle)/m['reward_validation']['fixture']), root/'reward-validation'):
                atomic_json(root/'status.json', {'status':'calibration_failed','benchmark_launched':False})
                return
        def run(c):
            name=c['run_id']; path=root/(name+'.json'); atomic_json(path,c)
            with (root/(name+'.log')).open('x') as log:
                command=[sys.executable,'-m','training_pipeline',c['execution']['operation'],'--config',str(path)]
                if c['execution']['operation'] in {'evaluate','fork'}:
                    checkpoint = resolve_checkpoint_pointer(state_root, c['execution']['checkpoint'])
                    if c['execution']['operation'] == 'evaluate':
                        command=[sys.executable,'-m','training_pipeline','evaluate','--config',str(path),
                                 '--checkpoint',str(checkpoint),'--output',c['output']]
                    else:
                        command.extend(['--checkpoint',str(checkpoint)])
                if 'cohort' in c['execution']:
                    command.extend(['--cohort',c['execution']['cohort']])
                p=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
            result={'run_id':name,'exit_code':p.returncode,'output':c['output']}
            atomic_json(root/(name+'.result.json'),result)
            return result
        results=[]
        # Benchmarks never overlap each other or training.
        benchmarks=[c for c in m['configs'] if c['execution']['operation']=='benchmark']
        rest=[c for c in m['configs'] if c['execution']['operation']!='benchmark']
        for c in benchmarks:
            result=run(c); results.append(result)
            if result['exit_code']:
                raise RuntimeError('Benchmark failed; remaining campaign not launched')
        if m['parallel_training'] and rest:
            with ThreadPoolExecutor(max_workers=min(4,len(rest))) as pool:
                results.extend(pool.map(run,rest))
        else:
            results.extend(run(c) for c in rest)
        atomic_json(root/'status.json',{'status':'complete' if all(r['exit_code']==0 for r in results) else 'failed',
                                      'results':results})
    except BaseException as exc:
        atomic_json(root/'status.json',{'status':'failed','error_type':type(exc).__name__})
        raise
    finally:
        subprocess.run(['sync',str(state_root)],check=True)


def main():
    p=argparse.ArgumentParser(description=__doc__); subs=p.add_subparsers(dest='command',required=True)
    q=subs.add_parser('prepare');q.add_argument('--configs',nargs='+',required=True);q.add_argument('--output',required=True);q.add_argument('--parallel-training',action='store_true');q.add_argument('--budget-plan');q.add_argument('--autoresearch',action='store_true')
    q=subs.add_parser('submit');q.add_argument('--bundle',required=True);q.add_argument('--isolated',action='store_true',help='Fresh campaign with private controller, storage and budget')
    q=subs.add_parser('status');q.add_argument('--sandbox-id',required=True)
    q=subs.add_parser('download');q.add_argument('--bundle-id',required=True);q.add_argument('--output',required=True);q.add_argument('--isolated',action='store_true',help='Read the campaign-private volume')
    subs.add_parser('worker'); a=p.parse_args()
    if a.command=='prepare':result=prepare(a.configs,a.output,a.parallel_training,a.budget_plan,a.autoresearch)
    elif a.command=='submit':result=submit(a.bundle,a.isolated)
    elif a.command=='worker':worker();return
    elif a.command=='status':
        import modal
        sb=modal.Sandbox.from_id(a.sandbox_id);code=sb.poll()
        sb.detach()
        result={'sandbox_id':a.sandbox_id,'status':'running' if code is None else 'exited','exit_code':code}
    else:
        import modal
        from modal.volume import FileEntryType
        if not re.fullmatch('[a-f0-9]{64}',a.bundle_id):raise ValueError('Invalid bundle ID')
        volume=modal.Volume.from_name(isolated_names(a.bundle_id)[1] if a.isolated else VOLUME);out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
        # Copy campaign logs plus all authoritative run artifacts and ledgers.
        count=0
        for prefix in ('campaigns/'+a.bundle_id,'artifacts'):
            for entry in volume.listdir(prefix,recursive=True):
                if entry.type != FileEntryType.FILE:continue
                target=out/relative(entry.path.lstrip('/'));target.parent.mkdir(parents=True,exist_ok=True)
                with target.open('xb') as f:
                    for part in volume.read_file(entry.path):f.write(part)
                count+=1
        result={'files':count,'output':str(out.resolve())}
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
