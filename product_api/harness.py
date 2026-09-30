"""Explicit tool profiles for demo inference; bash stays in a disposable sandbox."""
from dataclasses import asdict
import base64
import zlib
import hashlib
import json
from agent_harness.contracts import ToolSpec, ToolObservation
from agent_harness.bash_tool import command

class BashTool:
    spec = ToolSpec(name='bash', version='bash-only-v1', type='execution',
        capabilities=('shell',), input_schema={'type':'object','properties':{'command':{'type':'string','minLength':1}},'required':['command'],'additionalProperties':False},
        timeout_seconds=65, max_output_bytes=3500,
        description='Run a bash command in the pinned repository at /workspace.', required_resources=('bash_sandbox',))
    def execute(self, arguments, context):
        result = context.resources['bash_sandbox'].execute(command(arguments))
        content = asdict(result)
        return ToolObservation('ok' if result.exit_code == 0 else 'error', content,
            error=None if result.exit_code == 0 else 'Shell command failed', truncated=content.get('truncated',False))

def create_bash_sandbox(repo, manifest, directory, episode_id, limits):
    from agent_harness.images import normalize_recipe, _hash, _identity
    from agent_harness.modal_backend import ModalSandboxBackend, SandboxLimits
    from agent_harness.sandbox import _SOURCE_VERIFIER
    files=repo.snapshot_hashes()
    digest=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()
    if (manifest.get('status')!='ready' or manifest.get('backend')!='modal'
            or manifest.get('commit')!=repo.commit or manifest.get('snapshot_sha256')!=digest):
        raise ValueError('Bash requires a verified pinned Modal image')
    # Verify original source before any model-authored shell commands run.
    payload = base64.b64encode(zlib.compress(json.dumps(files).encode())).decode()
    # Keep each argv entry below the OS per-argument limit for larger repositories.
    verification = 'import base64,zlib\n' + _SOURCE_VERIFIER.split('\nprint(_source_digest')[0]
    verification += "\nexpected = json.loads(zlib.decompress(base64.b64decode(''.join(sys.argv[1:]))))\n"
    verification += 'assert _source_digest(expected) == ' + repr(digest) + '\n'
    argv = ['python', '-I', '-S', '-c', verification]
    argv.extend(payload[offset:offset+32000] for offset in range(0,len(payload),32000))
    recipe=normalize_recipe({'base_image_id':manifest['image_digest'],'mode':'source_reading',
        'readiness_commands':[argv]})
    spec={'schema_version':1,'status':'ready','image_id':manifest['image_digest'],
        'app_name':'repository-qa-training','commit':repo.commit,'source_sha256':digest,
        'recipe':recipe,'workspace_path':'/workspace',
        'readiness':[{'command':recipe['readiness_commands'][0],'exit_code':0}]}
    spec['environment_id']=_hash(_identity(spec))
    return ModalSandboxBackend(spec,directory/'sandbox-events',limits=SandboxLimits(
        lifetime_seconds=max(300,int(limits['wall_time_seconds'])+120),max_tool_calls=limits['max_tool_calls'],max_output_bytes=3500)).create(episode_id)
