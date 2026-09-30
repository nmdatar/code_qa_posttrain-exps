"""Submit once, only after the shared controller is free and budget reconciled."""
import json
from pathlib import Path
import modal
from modal.exception import NotFoundError
from training_pipeline.remote import APP,CONTROLLER,VOLUME,reconcile_ledger,submit,operation_estimate,controller_reservation
from training_pipeline.budget import ledger_lock
from training_pipeline.storage import atomic_json
report=Path('reports/sft-tool-warmup-v1')
try:
    active=modal.Sandbox.from_name(APP,CONTROLLER);code=active.poll();active.detach()
    if code is None:
        print('WAIT: existing shared controller still active');raise SystemExit(0)
except NotFoundError:pass
path=Path('artifacts/project-budget/02-direct-grpo.json')
v=modal.Volume.from_name(VOLUME);remote=json.loads(b''.join(v.read_file(str(path))))
with ledger_lock(path):
    reconcile_ledger(remote,json.loads(path.read_text()))
    atomic_json(path,remote)
config=json.loads(Path('configs/experiments/sft-tool-warmup-v1/train.json').read_text())
upper=operation_estimate(config)['upper_estimate_usd']+controller_reservation(config)
assert remote['reserved_usd']+upper<=remote['cap']
atomic_json(report/'budget-before.json',{'reserved_usd':remote['reserved_usd'],'cap_usd':remote['cap'],'incremental_upper_usd':upper,'combined_upper_usd':remote['reserved_usd']+upper,'project_ceiling_usd':1000})
result=submit('artifacts/sft-tool-prefix-v1-bundle');atomic_json(report/'submission.json',result);print(json.dumps(result,indent=2))
