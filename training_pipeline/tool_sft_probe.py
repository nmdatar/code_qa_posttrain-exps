"""Sampling-only secondary interface diagnostic on paired, frozen public contexts."""
import json, re, time
from pathlib import Path
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor
from agent_harness.repository_tools import command
from qa_eval.schema import validate, SUBMISSION
from .collection import CollectionEpisode
from .storage import atomic_json, read, digest
from .config import inputs
from .budget import SpendLedger
from .tinker_backend import TinkerBackend


def check(text, row, observed, config):
    episode=CollectionEpisode.__new__(CollectionEpisode)
    episode.row=row;episode.observed_files=observed;episode.factory=SimpleNamespace(config=config)
    strict_json=True
    try:
        if not isinstance(json.loads(text),dict):strict_json=False
    except ValueError:strict_json=False
    try:
        a=episode.parse_action(text)
        if not isinstance(a,dict):raise ValueError('Action is not an object')
        if set(a)=={'answer'}:
            answer=a['answer'];validate(answer,SUBMISSION)
            if answer['task_id']!=row['id']:raise ValueError('Task binding')
            for ref in answer['citations']:
                if observed.get(ref['path'])!=ref['file_sha256']:raise ValueError('Unobserved citation')
                if ref['start_line']<1 or ref['end_line']<ref['start_line']:raise ValueError('Invalid citation range')
            kind='answer'
        else:
            if set(a)!={'tool','arguments'} or a['tool'] not in row['public']['permitted_tools']:raise ValueError('Invalid tool envelope')
            command(a['tool'],a['arguments'],row['image_result']['snapshot_files'],paginate_reads=True)
            if a['tool']=='read_file':
                lo=a['arguments'].get('start_line',1);hi=a['arguments'].get('end_line',lo+119)
                if lo<1 or hi<lo:raise ValueError('Invalid line range')
            kind=a['tool']
        return {'valid':True,'strict_json_object':strict_json,'action_kind':kind,'error':None}
    except (ValueError,KeyError,TypeError) as e:
        return {'valid':False,'strict_json_object':strict_json,'action_kind':None,'error':type(e).__name__+': '+str(e)[:300]}


def execute(config, manifest, campaign):
    spec=manifest['tool_sft_probe'];fixture=read(Path('/bundle')/spec['fixture'])
    data=inputs(config);tasks={r['id']:r for r in data['development']}
    root=Path(config['output']);root.mkdir(parents=True,exist_ok=False)
    atomic_json(root/'config.json',config);atomic_json(root/'fixture.json',fixture)
    ledger=SpendLedger(config['spend']['ledger'],config['spend']['cap_usd'],config['spend']['prices'],read(config['spend']['ledger'])['ttl_seconds'])
    maximum=len(fixture['contexts'])*len(fixture['policies'])*fixture['samples_per_context']
    worst=maximum*(config['limits']['context_tokens']*config['spend']['prices']['prefill']+512*config['spend']['prices']['sample'])/1e6
    if worst>spec['reservation_cap_usd']:raise ValueError('Probe exceeds preregistered cap')
    backends={};before=read(ledger.path)['reserved_usd'];results=[]
    try:
        for name,artifacts in fixture['policies'].items():
            backend=TinkerBackend(config['model'],config['limits'],ledger);backend.load(artifacts,'evaluate');backends[name]=backend
        jobs=[(name,ctx,rep) for ctx in fixture['contexts'] for rep in range(fixture['samples_per_context']) for name in sorted(backends)]
        def sample(job):
            name,ctx,rep=job;started=time.monotonic()
            g=backends[name].sample(ctx['messages'],512,1)
            validation=check(g.text,tasks[ctx['task_id']],ctx['observed_files'],config)
            return {'policy':name,'context_id':ctx['id'],'task_id':ctx['task_id'],'family_id':tasks[ctx['task_id']]['family_id'],'replicate':rep,'text':g.text,'stop_reason':g.stop_reason,'input_tokens':len(g.prompt),'output_tokens':len(g.tokens),'latency_seconds':time.monotonic()-started,**validation}
        with ThreadPoolExecutor(max_workers=8) as pool:
            for row in pool.map(sample,jobs):
                results.append(row)
                with (root/'samples.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        atomic_json(root/'report.json',{'status':'complete','samples':len(results),'fixture_hash':digest(fixture),'reserved_usd':read(ledger.path)['reserved_usd']-before,'actual_billing_usd':None,'results':results})
    finally:
        for backend in backends.values():backend.close('success' if len(results)==maximum else 'errored')
