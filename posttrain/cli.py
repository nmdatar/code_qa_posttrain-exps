"""Post-training experiments, readiness inspection, and recovery."""
import argparse
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import sys
from .config import load_config
from .storage import read,atomic,BudgetLedger


def doctor(config=None, probe=False):
    modules={}
    for module,package in [('tinker','tinker'),('tinker_cookbook','tinker-cookbook'),('modal','modal'),('wandb','wandb')]:
        try: version=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:version=None
        modules[module]={'available':importlib.util.find_spec(module) is not None,'version':version}
    credentials={name:bool(os.environ.get(name)) for name in ['TINKER_API_KEY','WANDB_API_KEY','QA_TRAIN_JUDGE_MODEL','QA_TRAIN_JUDGE_BASE_URL','QA_EVAL_JUDGE_MODEL','QA_EVAL_JUDGE_BASE_URL']}
    result={'dependencies':modules,'process_configuration_present':credentials,'paid_calls':0,
            'note':'No training or paid calls. Presence checks do not prove account access. Credentials are never printed.'}
    from .services import tinker_auth_status,wandb_auth_status
    result['authentication']={'tinker':tinker_auth_status(),'wandb':wandb_auth_status()}
    if probe:
        from .services import probe_tinker_capabilities,fetch_pricing,rank_training_candidates
        try:
            capabilities=probe_tinker_capabilities();pricing=fetch_pricing()
            result['tinker_capabilities']=capabilities
            result['model_candidates']=rank_training_candidates(capabilities,pricing)
            result['pricing']={'source':pricing['source'],'sha256':pricing['sha256'],'retrieved_at':pricing['retrieved_at']}
        except Exception as exc:result['probe']={'status':'failed','error_type':type(exc).__name__}
    if config:
        from .runner import readiness,freeze_data
        try:result['readiness']=readiness(freeze_data(load_config(config)))
        except (ValueError,OSError,KeyError) as exc:result['readiness']={'status':'blocked','blockers':[str(exc)]}
    return result


def environment_check(bundle,task,output,ledger_path,bound):
    from .environments import ModalEnvironment
    budget=BudgetLedger(ledger_path)
    env=ModalEnvironment(bundle,task,output,budget,bound)
    # Exact same tool capability surface as training; no execution capability invented.
    listing=env.call('list_files',{'path':'.','limit':5})
    observation=listing.get('observation',{})
    try:files=json.loads(observation.get('stdout','{}')).get('files',[])
    except (ValueError,TypeError):files=[]
    checks={'list_files':listing}
    if files:
        checks['read_file']=env.call('read_file',{'path':files[0],'start_line':1,'end_line':3})
        checks['search_code']=env.call('search_code',{'query':'a','path':'.','limit':5})
    result={'status':'passed' if files and all(x.get('status')=='ok' for x in checks.values()) else 'incomplete',
            'scope':'source tool canaries; executable probe readiness is separate','task_id':task,'checks':checks}
    atomic(Path(output)/'posttrain-readiness.json',result)
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);commands=p.add_subparsers(dest='command',required=True)
    q=commands.add_parser('doctor');q.add_argument('--config');q.add_argument('--output');q.add_argument('--probe',action='store_true')
    q=commands.add_parser('data');s=q.add_subparsers(dest='action',required=True);q=s.add_parser('validate');q.add_argument('--release',required=True);q.add_argument('--output')
    q=commands.add_parser('env');s=q.add_subparsers(dest='action',required=True);q=s.add_parser('check');q.add_argument('--bundle',required=True);q.add_argument('--task',required=True);q.add_argument('--output',required=True);q.add_argument('--budget-ledger',required=True);q.add_argument('--call-upper-usd',type=float,required=True)
    q=commands.add_parser('run');q.add_argument('--config',required=True)
    q=commands.add_parser('resume');q.add_argument('--checkpoint',required=True);q.add_argument('--config')
    q=commands.add_parser('fork');q.add_argument('--checkpoint',required=True);q.add_argument('--config',required=True)
    q=commands.add_parser('evaluate');q.add_argument('--checkpoint',required=True);q.add_argument('--config',required=True)
    q=commands.add_parser('status');q.add_argument('--run',required=True)
    q=commands.add_parser('report');q.add_argument('--run',required=True);q.add_argument('--output')
    q=commands.add_parser('tracking');s=q.add_subparsers(dest='action',required=True);q=s.add_parser('sync');q.add_argument('--run',required=True);q.add_argument('--mode',choices=['online','offline','disabled']);q.add_argument('--project')
    q=commands.add_parser('calibration');s=q.add_subparsers(dest='action',required=True)
    q=s.add_parser('export');q.add_argument('--examples',required=True);q.add_argument('--output',required=True)
    q=s.add_parser('import');q.add_argument('--packet',required=True);q.add_argument('--decisions',required=True);q.add_argument('--output',required=True)
    args=p.parse_args(argv)
    try:
        result=dispatch(args)
        if getattr(args,'output',None) and args.command in ('doctor','data'):atomic(args.output,result)
        print(json.dumps(result,indent=2,default=str,allow_nan=False))
        if isinstance(result,dict) and result.get('readiness',{}).get('status')=='blocked':return 2
        return 0
    except (ValueError,OSError,KeyError,ImportError,RuntimeError) as exc:
        # Config/schema errors are actionable; remote exception bodies may contain secrets.
        print(json.dumps({'status':'error','type':type(exc).__name__,
                          'detail':str(exc) if isinstance(exc,(ValueError,FileNotFoundError)) else 'See durable run events; remote error details withheld'}),file=sys.stderr)
        return 2


def dispatch(a):
    if a.command=='doctor':return doctor(a.config,a.probe)
    if a.command=='data':
        from .data import inventory_release,validate_artifacts
        return {**inventory_release(a.release),'integrity':validate_artifacts(a.release)}
    if a.command=='env':return environment_check(a.bundle,a.task,a.output,a.budget_ledger,a.call_upper_usd)
    if a.command in ('run','resume','fork'):
        from .runner import run
        from .checkpoints import load_checkpoint
        config=a.config
        if a.command=='resume' and not config:config=load_checkpoint(a.checkpoint)['config']
        return run(config,getattr(a,'checkpoint',None),resume=a.command=='resume')
    if a.command=='evaluate':
        from .runner import Coordinator,make_backend,freeze_data,readiness
        from .checkpoints import load_checkpoint
        from .storage import lock
        c=freeze_data(load_config(a.config));m=load_checkpoint(a.checkpoint)
        for key in ('backend','model','lora_rank','renderer_name'):
            if c[key]!=m['config'][key]:raise ValueError('Evaluation checkpoint identity mismatch: '+key)
        if Path(c['budget_ledger']).resolve()!=Path(m['config']['budget_ledger']).resolve():
            raise ValueError('Checkpoint evaluation must retain the shared spending ledger')
        check=readiness(c)
        if check['blockers']:raise ValueError('Readiness blocked: '+'; '.join(check['blockers']))
        directory=Path(c['artifacts_root'])/c['run_id']
        if (directory/'config.json').exists():raise ValueError('Standalone evaluation requires a new run ID')
        directory.mkdir(parents=True,exist_ok=True)
        with lock(directory/'.writer.lock',blocking=False):
            atomic(directory/'config.json',c)
            ledger=BudgetLedger(c['budget_ledger'],c['budget_cap_usd'],c['budget_reserve_usd'])
            backend=make_backend(c,ledger);backend.load(m['references'],purpose='evaluate')
            return Coordinator(c,directory,backend,ledger).evaluation(m['state'],a.checkpoint)
    if a.command=='status':
        d=Path(a.run)
        from .tracking import EventLog
        events=EventLog(d).read()
        configuration=read(d/'config.json') if (d/'config.json').exists() else {}
        ledger_path=Path(configuration.get('budget_ledger','__missing__'))
        budget=None
        if ledger_path.is_file():
            ledger=read(ledger_path)
            charged=sum(c['charged_usd'] for c in ledger['calls'].values())
            budget={'scope':'shared campaign, not this run alone','charged_or_reserved_usd':charged,
                    'available_for_dispatch_usd':max(0,ledger['cap_usd']-ledger['reserve_usd']-charged),
                    'actual_billing_complete':all(c.get('actual_usd') is not None for c in ledger['calls'].values())}
        return {'run':str(d),'state':read(d/'state.json') if (d/'state.json').exists() else None,
                'latest_checkpoint':read(d/'latest-checkpoint.json') if (d/'latest-checkpoint.json').exists() else None,
                'budget':budget,'evaluations':[e for e in events if e['kind']=='evaluation'],
                'failures':[e for e in events if e['kind'] in {'episode_failure','group_unresolved','update_unknown','run_interrupted'}],
                'events':len(events),'last_event':events[-1] if events else None}
    if a.command=='report':
        from .report import create_report
        return {'report':str(create_report(a.run,a.output))}
    if a.command=='tracking':
        from .tracking import sync_tracking
        c=read(Path(a.run)/'config.json')
        return sync_tracking(a.run,mode=a.mode or c['tracking_mode'],project=a.project or c['tracking_project'])
    if a.command=='calibration':
        from .calibration import export_packet,import_reviews
        from .data import read_jsonl
        return export_packet(read_jsonl(a.examples),a.output) if a.action=='export' else import_reviews(a.packet,a.decisions,a.output)
    raise ValueError('Unknown command')
