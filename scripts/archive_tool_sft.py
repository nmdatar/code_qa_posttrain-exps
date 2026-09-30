"""Download only this completed experiment, verify JSON and keep file hashes."""
import concurrent.futures,hashlib,json,argparse
from pathlib import Path
import modal
from modal.volume import FileEntryType
from training_pipeline.remote import reconcile_ledger
from training_pipeline.storage import atomic_json
from training_pipeline.budget import ledger_lock
parser=argparse.ArgumentParser();parser.add_argument('--probe',action='store_true');args=parser.parse_args()
suffix='probe-' if args.probe else ''
run_id='sft-tool-prefix-v1-probe' if args.probe else 'sft-tool-prefix-v1-seed42'
report=Path('reports/sft-tool-warmup-v1');receipt=json.loads((report/(suffix+'submission.json')).read_text())
s=modal.Sandbox.from_id(receipt['sandbox_id']);code=s.poll();s.detach();assert code is not None,'Still running'
v=modal.Volume.from_name('repository-qa-training-state-v2');root=Path('artifacts/sft-tool-prefix-v1-probe-results' if args.probe else 'artifacts/sft-tool-prefix-v1-results')
paths=[e.path for prefix in ['campaigns/'+receipt['bundle_id'],'artifacts/experiments/'+run_id] for e in v.listdir(prefix,recursive=True) if e.type==FileEntryType.FILE]
def get(p):
    target=root/p.lstrip('/');target.parent.mkdir(parents=True,exist_ok=True)
    raw=b''.join(v.read_file(p))
    if target.suffix=='.json':json.loads(raw)
    if target.suffix=='.jsonl':
        for line in raw.splitlines():
            if line.strip():json.loads(line)
    target.write_bytes(raw)
    return str(target),hashlib.sha256(raw).hexdigest()
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:checksums=dict(pool.map(get,paths))
atomic_json(report/(suffix+'checksums.json'),checksums)
p=Path('artifacts/project-budget/02-direct-grpo.json');remote=json.loads(b''.join(v.read_file(str(p))))
with ledger_lock(p):
    reconcile_ledger(remote,json.loads(p.read_text()));atomic_json(p,remote)
before=json.loads((report/('budget-after.json' if args.probe else 'budget-before.json')).read_text())
atomic_json(report/(suffix+'budget-after.json'),{'reserved_usd':remote['reserved_usd'],'cap_usd':remote['cap'],'incremental_reservations_usd':remote['reserved_usd']-before['reserved_usd'],'actual_provider_billing_usd':None,'controller_exit':code})
atomic_json(report/(suffix+'archive.json'),{'controller_exit':code,'files':len(checksums),'root':str(root.resolve())});print('archived',len(checksums),'files','exit',code)
