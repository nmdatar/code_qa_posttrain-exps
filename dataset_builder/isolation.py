"""Observed sandbox isolation checks; not a security certification."""
import json
from pathlib import Path
from .build import write_json
from .contracts import canonical_hash
from .environment import ModalBackend, DockerBackend, RunLimits

PROBE = '''import os,json,socket
from pathlib import Path
r={'nonroot':os.getuid()!=0,'gold_absent':not Path('/workspace/private').exists() and not Path('/workspace/data/generated').exists(),'fresh_tmp':not Path('/tmp/dataset-marker').exists()}
try:
 Path('/workspace/forbidden-write').write_text('x');r['source_readonly']=False
except PermissionError:r['source_readonly']=True
s=socket.socket();s.settimeout(2)
try:s.connect(('1.1.1.1',443));r['network_blocked']=False
except OSError:r['network_blocked']=True
finally:s.close()
Path('/tmp/dataset-marker').write_text('attempt')
print(json.dumps(r,sort_keys=True))
'''


def check_isolation(bundle):
    root = Path(bundle)
    built = json.loads((root / 'environment-build/result.json').read_text())
    backend = ModalBackend() if built['backend'] == 'modal' else DockerBackend()
    results = [backend.run(built['image_digest'], ['python', '-'], RunLimits(timeout_seconds=10), stdin=PROBE) for _ in range(2)]
    checks = [json.loads(r['stdout']) if r['exit_code'] == 0 else {} for r in results]
    report = {'checks': checks, 'executions': results, 'environment_hash': canonical_hash(built), 'image_id': built['image_digest'],
              'passed': all(v and all(v.values()) for v in checks) and all(r['exit_code']==0 and not r['timed_out'] and not r['truncated'] for r in results),
              'scope': 'Observed isolation checks; not a security certification'}
    write_json(root / 'private/isolation-report.json', report)
    return report
