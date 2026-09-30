"""Observe runtime versions and verify every source file inside the actual image."""
import json
from pathlib import Path
from .build import write_json
from .contracts import canonical_hash
from .environment import ModalBackend, DockerBackend, RunLimits

PROGRAM = r'''
import hashlib, importlib.metadata, json, pathlib, platform, shutil, subprocess, sys
expected=json.load(sys.stdin)
root=pathlib.Path('/workspace')
mismatches=[]
for name,wanted in expected.items():
    path=root/name
    if not path.resolve().is_relative_to(root):
        mismatches.append(name); continue
    try: actual=hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError: actual=None
    if actual!=wanted: mismatches.append(name)
inventory={'python':platform.python_version(), 'python_packages':sorted(
    [{'name':d.metadata['Name'],'version':d.version} for d in importlib.metadata.distributions()],
    key=lambda d:(d['name'] or '').lower())}
if shutil.which('node'):
    inventory['node']=subprocess.check_output(['node','--version'],text=True).strip()
artifact=pathlib.Path('/opt/query-core.cjs')
if artifact.is_file(): inventory['query_core_bundle_sha256']=hashlib.sha256(artifact.read_bytes()).hexdigest()
print(json.dumps({'source_files_checked':len(expected),'mismatches':mismatches,'inventory':inventory},sort_keys=True))
if mismatches:sys.exit(1)
'''


def attest(bundle):
    bundle=Path(bundle)
    built=json.loads((bundle/'environment-build/result.json').read_text())
    if built['status']!='ready': raise ValueError('Environment not ready')
    backend=ModalBackend() if built['backend']=='modal' else DockerBackend()
    result=backend.run(built['image_digest'],['python','-I','-c',PROGRAM],
                       RunLimits(timeout_seconds=60),stdin=json.dumps(built['snapshot_files']))
    success=result['exit_code']==0 and not result['timed_out'] and not result['truncated']
    observed=json.loads(result['stdout']) if success else None
    report={'environment_hash':canonical_hash(built),'image_id':built['image_digest'],
            'status':'passed' if success and not observed['mismatches'] else 'not_passed',
            'observation':observed,'execution':result}
    write_json(bundle/'private/image-attestation.json',report)
    return report


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--bundle',required=True);args=p.parse_args()
    report=attest(args.bundle);print(json.dumps(report,indent=2))
    if report['status']!='passed':raise SystemExit(1)
