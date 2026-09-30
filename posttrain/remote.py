"""Optional detached Modal CPU coordinator. Import and plan never contact Modal."""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from pathlib import Path, PurePosixPath
import math


@dataclass(frozen=True)
class RemoteConfig:
    app_name: str = 'repo-qa-posttrain'
    runs_volume: str = 'repo-qa-posttrain-runs'
    private_volume: str = 'repo-qa-posttrain-private'
    secret_names: tuple[str, ...] = field(default_factory=tuple)
    timeout_seconds: int = 3600
    cpu: float = 2.0
    memory_mb: int = 4096
    original_workspace: str | None = None

    def validate(self):
        if self.runs_volume == self.private_volume:
            raise ValueError('Private grading and run artifacts need separate volumes')
        if not 1 <= self.timeout_seconds <= 86400 or not math.isfinite(self.cpu) or self.cpu <= 0 or self.memory_mb < 256:
            raise ValueError('Invalid coordinator resources')
        if self.original_workspace:
            original = PurePosixPath(self.original_workspace)
            if ('..' in original.parts or len(original.parts) < 4 or
                    not any(original.is_relative_to(prefix) for prefix in ('/Users', '/home'))):
                raise ValueError('Original workspace alias must be a nested /Users or /home path')


def _under(path, mount):
    p = PurePosixPath(str(path))
    return p.is_absolute() and '..' not in p.parts and p.is_relative_to(mount) and p != PurePosixPath(mount)


def validate_remote_plan(run_config, config_path, envelope_usd, coordinator_upper_usd, remote=None):
    """Validate a bounded allocation without reading credentials or using services."""
    from .config import load_config
    remote = remote or RemoteConfig()
    remote.validate()
    c = load_config(run_config)
    if not _under(config_path, '/private-grading'):
        raise ValueError('Staged run config must be private under /private-grading')
    if not _under(c['artifacts_root'], '/runs') or not _under(c['budget_ledger'], '/runs'):
        raise ValueError('Remote artifacts and child spending ledger must be absolute /runs paths')
    if c['budget_ledger'].startswith(c['artifacts_root'].rstrip('/') + '/' + c['run_id'] + '/'):
        raise ValueError('Keep shared child ledger outside run artifact directory')
    if not all(math.isfinite(x) for x in (envelope_usd, coordinator_upper_usd)):
        raise ValueError('Finite remote allocation required')
    if not 0 < coordinator_upper_usd < envelope_usd <= 20:
        raise ValueError('Positive coordinator estimate inside total envelope <=$20 required')
    if c['budget_cap_usd'] > envelope_usd - coordinator_upper_usd + 1e-9:
        raise ValueError('Child cap plus coordinator estimate exceeds reserved remote envelope')
    if c['budget_ledger'] == 'artifacts/posttrain/spending.json':
        raise ValueError('Never copy the live local spending ledger')
    return {'schema_version':'1.0','status':'offline_plan_not_deployed','run_config':c,
            'remote':asdict(remote),'config_path':str(config_path),
            'envelope_usd':envelope_usd,'coordinator_upper_usd':coordinator_upper_usd,
            'child_cap_usd':c['budget_cap_usd'],'provider_enforced_billing_cap':False,
            'staging':{'workspace':'/private-grading/workspace','config':str(config_path),
                       'run_artifacts':c['artifacts_root'],'child_ledger':c['budget_ledger'],
                       'never_stage':['live parent spending ledger','API key files','local credential stores']}}


def _coordinator_command(config_path, checkpoint=None):
    import sys
    if not _under(config_path, '/private-grading'):
        raise ValueError('Config must be uploaded to private volume')
    if checkpoint:
        if not _under(checkpoint, '/runs'):
            raise ValueError('Resume only a checkpoint already in the remote runs volume')
        return [sys.executable,'-m','posttrain','resume','--checkpoint',checkpoint,'--config',config_path]
    return [sys.executable,'-m','posttrain','run','--config',config_path]


def build_modal_app(config: RemoteConfig | None = None):
    """Construct only. Launch is explicit; solver sandboxes inherit no volumes."""
    import modal
    config = config or RemoteConfig()
    config.validate()
    app = modal.App(config.app_name)
    runs = modal.Volume.from_name(config.runs_volume, create_if_missing=True)
    private = modal.Volume.from_name(config.private_volume, create_if_missing=True)
    requirements = Path(__file__).resolve().parents[1] / 'requirements-posttrain.txt'
    if not requirements.is_file():
        raise ValueError('Remote image build requires repository checkout with requirements-posttrain.txt')
    image = (modal.Image.debian_slim(python_version='3.11')
             .pip_install_from_requirements(str(requirements))
             .add_local_python_source('posttrain','dataset_builder','qa_eval'))

    @app.function(image=image, volumes={'/runs':runs,'/private-grading':private},
                  secrets=[modal.Secret.from_name(name) for name in config.secret_names],
                  cpu=(config.cpu,config.cpu),memory=(config.memory_mb,config.memory_mb),timeout=config.timeout_seconds,
                  max_containers=1,retries=0)
    def coordinator(config_path: str, allocation: dict, checkpoint: str | None = None):
        import json
        import subprocess
        import threading
        from .storage import digest
        c = json.loads(Path(config_path).read_text())
        plan = validate_remote_plan(c,config_path,allocation['envelope_usd'],
                                    allocation['coordinator_upper_usd'],config)
        if digest(plan['run_config']) != allocation['run_config_sha256']:
            raise ValueError('Staged config differs from locally approved allocation')
        workspace = Path('/private-grading/workspace')
        if not workspace.is_dir():raise ValueError('Private workspace not staged')
        # Preserve immutable manifests' recorded source paths without rewriting
        # or invalidating image/environment hash bindings.
        if config.original_workspace:
            alias = Path(config.original_workspace)
            if alias.exists() and alias.resolve()!=workspace.resolve():
                raise ValueError('Original workspace alias already occupied')
            alias.parent.mkdir(parents=True,exist_ok=True)
            if not alias.exists():alias.symlink_to(workspace,target_is_directory=True)
        # A fresh launch cannot reuse a copied ledger; resume retains its own
        # existing remote child ledger and therefore its recorded charges.
        child = Path(plan['run_config']['budget_ledger'])
        if not checkpoint and child.exists():raise ValueError('Fresh remote launch requires a new child ledger path')
        if checkpoint and not child.is_file():raise ValueError('Resume requires existing remote child ledger')
        stopped=threading.Event()
        def flush():
            while not stopped.wait(15):runs.commit()
        thread=threading.Thread(target=flush,daemon=True);thread.start()
        try:
            result=subprocess.run(_coordinator_command(config_path,checkpoint),cwd=workspace,check=False)
            return {'exit_code':result.returncode,'config_path':config_path,'run_id':c['run_id']}
        finally:
            stopped.set();thread.join(timeout=20);runs.commit()
    return app,coordinator


def launch_remote(config_path, budget, upper_usd, config=None, checkpoint=None,
                  *, run_config=None, coordinator_upper_usd=None):
    """Reserve the ENTIRE remote envelope in parent ledger before any dispatch.

    The remote child ledger is new and receives only its allocated sub-budget.
    Parent retains full charge even if launch fails or remote billing is unknown.
    It is never copied to Modal. Resume conservatively reserves another envelope.
    """
    from .storage import digest
    if run_config is None or coordinator_upper_usd is None:
        raise ValueError('Local run config and coordinator estimate required for envelope validation')
    plan=validate_remote_plan(run_config,config_path,upper_usd,coordinator_upper_usd,config)
    _coordinator_command(config_path,checkpoint)
    allocation={'envelope_usd':upper_usd,'coordinator_upper_usd':coordinator_upper_usd,
                'run_config_sha256':digest(plan['run_config'])}
    def dispatch():
        app,coordinator=build_modal_app(config)
        with app.run(detach=True):
            call=coordinator.spawn(config_path,allocation,checkpoint)
            return {'function_call_id':call.object_id,'status':'dispatched','allocation':allocation}
    return budget.execute('modal-remote-envelope',upper_usd,dispatch,
                          {**allocation,'config_path':config_path,'checkpoint':checkpoint,
                           'child_ledger':plan['run_config']['budget_ledger'],
                           'provider_enforced_billing_cap':False})


def main():
    import argparse
    import json
    from .storage import BudgetLedger,read
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['plan','launch'])
    parser.add_argument('--config',required=True,help='Local copy of staged remote experiment JSON')
    parser.add_argument('--remote-config',required=True,help='Absolute uploaded config path under /private-grading')
    parser.add_argument('--envelope-usd',required=True,type=float)
    parser.add_argument('--coordinator-upper-usd',required=True,type=float)
    parser.add_argument('--parent-ledger',default='artifacts/posttrain/spending.json')
    parser.add_argument('--runs-volume',default='repo-qa-posttrain-runs')
    parser.add_argument('--private-volume',default='repo-qa-posttrain-private')
    parser.add_argument('--secret',action='append',default=[])
    parser.add_argument('--original-workspace')
    parser.add_argument('--checkpoint',help='Existing remote /runs checkpoint only')
    parser.add_argument('--timeout-seconds',type=int,default=3600)
    args=parser.parse_args()
    remote=RemoteConfig(runs_volume=args.runs_volume,private_volume=args.private_volume,
                        secret_names=tuple(args.secret),original_workspace=args.original_workspace,
                        timeout_seconds=args.timeout_seconds)
    c=read(args.config)
    plan=validate_remote_plan(c,args.remote_config,args.envelope_usd,args.coordinator_upper_usd,remote)
    if args.action=='plan':result=plan
    else:
        if not Path(args.parent_ledger).is_file():raise ValueError('Use an existing parent ledger; do not reset the task budget')
        result=launch_remote(args.remote_config,BudgetLedger(args.parent_ledger),args.envelope_usd,
                             remote,args.checkpoint,run_config=c,coordinator_upper_usd=args.coordinator_upper_usd)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
