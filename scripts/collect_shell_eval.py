"""Inspect or download only this approved campaign's artifacts."""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import modal
from modal.exception import NotFoundError
from modal.volume import FileEntryType
from training_pipeline.remote import VOLUME
from training_pipeline.storage import atomic_json


def collect(bundle, output, download=False):
    manifest=json.loads((bundle/'bundle.json').read_text())
    receipt=json.loads(bundle.with_name(bundle.name+'.submission.json').read_text())
    sandbox=modal.Sandbox.from_id(receipt['sandbox_id'])
    try: code=sandbox.poll()
    finally: sandbox.detach()
    volume=modal.Volume.from_name(VOLUME)
    campaign='campaigns/'+manifest['bundle_id']
    def read(path):
        try: return b''.join(volume.read_file(path)).decode()
        except (NotFoundError,FileNotFoundError): return None
    status=read(campaign+'/status.json')
    result={'controller_exit':code,'campaign':json.loads(status) if status else None,'runs':[]}
    for config in manifest['configs']:
        prefix=config['output'].removeprefix('/state/')
        raw=read(prefix+'/events.jsonl')
        try:
            events=[json.loads(line) for line in raw.splitlines() if line.strip()] if raw else []
            snapshot_complete=True
        except json.JSONDecodeError:
            if code is not None: raise
            events=[];snapshot_complete=False
        item={'run_id':config['run_id'],'trajectories':sum(e['event']=='trajectory' for e in events) if snapshot_complete else None,'snapshot_complete':snapshot_complete}
        item['milestones']=[{k:e[k] for k in ['event','status','demonstrated_quality','scoring_coverage','completion_rate'] if k in e} for e in events if e['event'] in ['evaluation','run_end']]
        result['runs'].append(item)
    if download:
        if code is None: raise ValueError('Wait until the controller exits before final download')
        prefixes=[campaign]+[c['output'].removeprefix('/state/') for c in manifest['configs']]
        entries=[]
        for prefix in prefixes:
            try: entries.extend(e.path.lstrip('/') for e in volume.listdir(prefix,recursive=True) if e.type==FileEntryType.FILE)
            except (NotFoundError,FileNotFoundError): pass
        entries.extend(c['spend']['ledger'].removeprefix('/state/') for c in manifest['configs'])
        def save(name):
            relative=Path(name)
            if relative.is_absolute() or '..' in relative.parts: raise ValueError('Unsafe artifact path')
            try: raw=b''.join(volume.read_file(name))
            except (NotFoundError,FileNotFoundError): return False
            target=output/relative;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
            return True
        with ThreadPoolExecutor(max_workers=8) as pool:
            result['downloaded_files']=sum(pool.map(save,sorted(set(entries))))
    atomic_json(output/'campaign-status.json',result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,default=Path('artifacts/shell-only-v1-bundle'));p.add_argument('--output',type=Path,default=Path('artifacts/shell-only-v1-results'));p.add_argument('--download',action='store_true');a=p.parse_args()
    print(json.dumps(collect(a.bundle,a.output,a.download),indent=2))
