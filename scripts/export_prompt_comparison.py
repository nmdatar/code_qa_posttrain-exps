"""Export a timestamped, read-only presentation snapshot of both prompt arms."""
import os, json, hashlib
from pathlib import Path
from datetime import datetime, timezone
import certifi
os.environ.setdefault('SSL_CERT_FILE',certifi.where())
import modal
from training_pipeline.storage import read, atomic_json

ARMS={'Original prompt':'bash-correctness-grpo-15-v1', 'Qwen397 subsections':'bash-prompt-decomposition-grpo-7-v1'}
EVENTS={'evaluation','training_batch','training_step_timing','update','run_finish','experiment_stop','tracking_failure'}

def fetch(name):
    receipt=read('artifacts/'+name+'-bundle.submission.json')
    sb=modal.Sandbox.from_id(receipt['sandbox_id'])
    try:
        code=sb.poll()
        if code is None:
            source='''
import json
from pathlib import Path
root=Path('/state/artifacts/experiments/NAME')
def read(p): return json.loads(p.read_text())
events=[]
for line in (root/'events.jsonl').read_text().splitlines():
 try: events.append(json.loads(line))
 except ValueError: pass
out={'events':[e for e in events if e['event'] in EVENTS],
'evaluations':[read(p) for p in (root/'evaluations').glob('*.json')],
'prompts':[{k:v for k,v in read(p).items() if k in ('task_id','question','subsections','identity')} for p in (root/'prompt-decomposition').glob('*.json') if p.name!='manifest.json'],
'config':read(root/'config.json')}
print(json.dumps(out))
'''.replace('NAME',name).replace('EVENTS',repr(EVENTS))
            p=sb.exec('python','-c',source,timeout=45)
            raw=p.stdout.read()
            if p.wait()!=0: raise RuntimeError('Remote snapshot failed')
            data=json.loads(raw)
        else:
            vol=modal.Volume.from_name(receipt['volume'])
            prefix='artifacts/experiments/'+name
            def get(path): return b''.join(vol.read_file(path))
            events=[json.loads(line) for line in get(prefix+'/events.jsonl').splitlines()]
            paths=[e.path.lstrip('/') for e in vol.listdir(prefix+'/evaluations') if e.path.endswith('.json')]
            data={'events':[e for e in events if e['event'] in EVENTS],
                  'evaluations':[json.loads(get(p)) for p in paths],
                  'prompts':[], 'config':json.loads(get(prefix+'/config.json'))}
            if 'prompt_decomposition' in data['config']:
                for entry in vol.listdir(prefix+'/prompt-decomposition'):
                    if entry.path.endswith('.json') and not entry.path.endswith('/manifest.json'):
                        value=json.loads(get(entry.path.lstrip('/')))
                        data['prompts'].append({k:v for k,v in value.items() if k in ('task_id','question','subsections','identity')})
        data.update(controller_exit=code, receipt=receipt)
        return data
    finally: sb.detach()

if __name__=='__main__':
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out=Path('reports/bash-prompt-decomposition-grpo-7-v1/presentation')/stamp
    out.mkdir(parents=True,exist_ok=False)
    snapshot={'fetched_at_utc':stamp,'arms':{label:fetch(name) for label,name in ARMS.items()}}
    atomic_json(out/'snapshot.json',snapshot)
    atomic_json(out/'provenance.json',{'sha256':hashlib.sha256((out/'snapshot.json').read_bytes()).hexdigest(),'source':'Modal authoritative experiment records','read_only':True})
    print(str(out))
