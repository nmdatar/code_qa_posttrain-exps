"""Bounded, stateless repository investigation tools.

Each operation uses a fresh isolated sandbox; /tmp does not persist. This is a
controller interface, not a simulated model or an automatic semantic grader.
Only public inputs are loaded. Token usage and monetary cost remain unknown.
"""
import argparse
import fcntl
from functools import wraps
from dataclasses import asdict
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import time

from .contracts import canonical_hash, validate_public_task
from .environment import DockerBackend, ModalBackend, RunLimits, _export_snapshot
from qa_eval.deterministic import snapshot


def _read(path):
    return json.loads(Path(path).read_text())


def _write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _safe(path):
    p = PurePosixPath(path)
    if p.is_absolute():
        if not str(p).startswith('/workspace/') and str(p) != '/workspace':
            raise ValueError('Path must be within /workspace')
        p = p.relative_to('/workspace')
    if '..' in p.parts or '\\' in str(p) or '\0' in str(p):
        raise ValueError('Unsafe repository path')
    return str(p)


def validated_snapshot_files(checkout, built):
    if built.get('source_inventory_mode') == 'git_objects_primary':
        if built.get('capability') != 'source_reading':
            raise ValueError('Git source projection cannot execute repository code')
        from .git_source_environment import inventory
        observed = inventory(checkout, built['commit'])
        for key in ('git_tree', 'symbolic_links', 'submodules'):
            if observed[key] != built.get(key):
                raise ValueError('Git source metadata mismatch')
        return observed['snapshot_files']
    snapshot(checkout, built['commit'])
    hashes, _ = _export_snapshot(checkout, built['commit'], None)
    return hashes


def start(bundle, task, output):
    root, output = Path(bundle), Path(output)
    manifest = _read(root / 'manifest.json')
    for relative in ('public/tasks.jsonl', 'public/environment.json'):
        if _sha((root / relative).read_bytes()) != manifest['artifacts'].get(relative):
            raise ValueError('Public artifact hash mismatch: ' + relative)
    public = [json.loads(x) for x in (root / 'public/tasks.jsonl').read_text().splitlines() if x]
    matches = [t for t in public if t['id'] == task]
    if len(matches) != 1:
        raise ValueError('Expected exactly one matching public task')
    task = matches[0]
    validate_public_task(task)
    env = _read(root / 'public/environment.json')
    built = _read(root / 'environment-build/result.json')
    if (built.get('status') != 'ready' or built.get('commit') != task['repository']['commit']
            or task['repository'] != env['repository'] or task['environment_id'] != env['id']
            or built.get('source_environment_sha256') != canonical_hash(env)):
        raise ValueError('Environment is not ready or is bound to another task/recipe')
    readiness = built.get('readiness', {})
    if readiness.get('exit_code') != 0 or readiness.get('timed_out') or readiness.get('truncated'):
        raise ValueError('Readiness was not successful')
    checkout = Path(env['snapshot_path'])
    hashes = validated_snapshot_files(checkout, built)
    if hashes != built.get('snapshot_files') or built.get('snapshot_sha256') != _sha(json.dumps(hashes, sort_keys=True).encode()):
        raise ValueError('Built source snapshot mismatch')
    image = built.get('image_id', built.get('image_digest', ''))
    if built.get('backend') == 'modal':
        valid = re.fullmatch(r'im-[A-Za-z0-9]+', image)
    else:
        valid = built.get('backend') == 'docker' and re.fullmatch(r'sha256:[a-f0-9]{64}', image)
    if not valid:
        raise ValueError('Immutable image reference required')
    if output.exists() and any(output.iterdir()):
        raise ValueError('Attempt output must be empty')
    output.mkdir(parents=True, exist_ok=True)
    record = {'schema_version': '1', 'task': task, 'image_id': image, 'backend': built['backend'],
              'snapshot_files': hashes, 'environment_hash': canonical_hash(built),
              'started_at': time.time(), 'tool_semantics': 'fresh stateless sandbox per operation; temporary files do not persist',
              'token_usage': None, 'cost_usd': None}
    if built.get('source_inventory_mode') == 'git_objects_primary':
        if 'python_probe' in task['permitted_tools']:
            raise ValueError('Source-only projection cannot grant execution')
        record['source_metadata'] = {'symbolic_links': built['symbolic_links'], 'unavailable_submodules': built['submodules']}
    _write(output / 'attempt.json', record)
    (output / 'trajectory.jsonl').touch()
    return record


# This trusted program runs ONLY inside the selected backend sandbox.
_INSPECT = r"""
import hashlib,json,pathlib,sys
args=json.loads(sys.stdin.readline()); root=pathlib.Path('/workspace').resolve()
def safe(value):
    p=(root/value).resolve()
    if not p.is_relative_to(root): raise ValueError('Path escaped repository')
    return p
name=args['name']; a=args['args']; expected=args['hashes']
def read(rel):
    p=safe(rel)
    data=p.read_bytes()
    if hashlib.sha256(data).hexdigest()!=expected[rel]: raise ValueError('Source hash mismatch')
    return data.decode('utf-8',errors='replace').splitlines()
if name=='list_files':
    prefix=a.get('path','.')
    safe(prefix)
    files=[p for p in sorted(expected) if prefix=='.' or p==prefix or p.startswith(prefix.rstrip('/')+'/')]
    result={'files':files[:a['limit']], 'truncated':len(files)>a['limit']}
    if args.get('source_metadata'):
        result['source_metadata']={k:{p:v for p,v in entries.items() if prefix=='.' or p==prefix or p.startswith(prefix.rstrip('/')+'/')} for k,entries in args['source_metadata'].items()}
    print(json.dumps(result))
elif name=='read_file':
    lines=read(a['path']); start=a['start_line']; end=min(a['end_line'],len(lines))
    print(json.dumps({'path':a['path'],'total_lines':len(lines),'lines':[{'line':i+1,'text':lines[i]} for i in range(start-1,end)]}))
else:
    results=[]; truncated=False
    for rel in sorted(expected):
        prefix=a.get('path','.')
        if prefix!='.' and rel!=prefix and not rel.startswith(prefix.rstrip('/')+'/'): continue
        for i,line in enumerate(read(rel),1):
            if a['query'] in line:
                if len(results)>=a['limit']: truncated=True; break
                results.append({'path':rel,'line':i,'text':line[:2000]})
        if truncated: break
    print(json.dumps({'matches':results,'truncated':truncated}))
"""


def _command(record, name, args):
    if name not in record['task']['permitted_tools']:
        raise ValueError('Tool not permitted')
    a = dict(args)
    if name == 'python_probe':
        if set(a) != {'code'} or not isinstance(a['code'], str) or len(a['code'].encode()) > 32000:
            raise ValueError('Probe requires code up to 32000 bytes')
        return ['python', '-'], a['code'], a
    allowed = {'list_files': {'path', 'limit'}, 'search_code': {'path', 'query', 'limit'},
               'read_file': {'path', 'start_line', 'end_line'}}
    if name not in allowed or set(a) - allowed[name]:
        raise ValueError('Unknown tool or arguments')
    a['path'] = _safe(a.get('path', '.'))
    if name == 'read_file':
        if a['path'] not in record['snapshot_files']:
            raise ValueError('Path not a tracked snapshot file')
        a.setdefault('start_line', 1); a.setdefault('end_line', a['start_line'] + 199)
        if not all(type(a[k]) is int for k in ('start_line', 'end_line')) or not 1 <= a['start_line'] <= a['end_line'] or a['end_line'] - a['start_line'] >= 400:
            raise ValueError('Invalid line range (maximum 400 lines)')
    else:
        a.setdefault('limit', 100)
        if type(a['limit']) is not int or not 1 <= a['limit'] <= 200:
            raise ValueError('Limit must be 1..200')
        if name == 'search_code' and (not isinstance(a.get('query'), str) or not 1 <= len(a['query']) <= 500):
            raise ValueError('Search requires a nonempty literal query up to 500 characters')
    return ['python', '-I', '-c', _INSPECT], json.dumps({'name': name, 'args': a, 'hashes': record['snapshot_files'], 'source_metadata': record.get('source_metadata')}), a


def _events(attempt):
    events = [json.loads(x) for x in (Path(attempt) / 'trajectory.jsonl').read_text().splitlines() if x]
    previous = None
    for event in events:
        digest = event['record_sha256']
        body = {k:v for k,v in event.items() if k != 'record_sha256'}
        if body['previous_sha256'] != previous or canonical_hash(body) != digest:
            raise ValueError('Trajectory hash chain mismatch')
        previous = digest
    return events


def _locked(function):
    @wraps(function)
    def wrapped(attempt, *args, **kwargs):
        with (Path(attempt) / '.lock').open('a') as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            return function(attempt, *args, **kwargs)
    return wrapped


@_locked
def tool(attempt, name, args, backend=None):
    attempt = Path(attempt); record = _read(attempt / 'attempt.json'); events = _events(attempt)
    event = {'sequence': len(events), 'name': name, 'args': args, 'started_at': time.time(),
             'previous_sha256': events[-1]['record_sha256'] if events else None,
             'attempt_sha256': canonical_hash(record), 'token_usage': None, 'cost_usd': None}
    started = time.monotonic()
    try:
        if any(e['attempt_sha256'] != canonical_hash(record) for e in events):
            raise ValueError('Attempt binding changed')
        if (attempt / 'answer.json').exists():
            raise ValueError('Attempt already finished')
        budgets = record['task']['budgets']
        if len(events) >= budgets['max_tool_calls']:
            raise ValueError('Tool call budget exhausted')
        remaining = budgets['latency_seconds'] - (time.time() - record['started_at'])
        if remaining <= 0:
            raise ValueError('Attempt deadline exceeded')
        command, stdin, normalized = _command(record, name, args)
        limits = RunLimits(timeout_seconds=min(30, remaining), memory_mb=512, cpus=1, pids=64, output_bytes=65536)
        backend = backend or (ModalBackend() if record['backend'] == 'modal' else DockerBackend())
        event['normalized_args'] = normalized; event['limits'] = asdict(limits)
        result = backend.run(record['image_id'], command, limits, stdin=stdin)
        event['observation'] = result
        event['status'] = ('ok' if result['exit_code'] == 0 and not result['timed_out'] and not result['truncated'] else 'tool_error')
    except Exception as exc:
        event['status'] = 'error'; event['error'] = f'{type(exc).__name__}: {exc}'
    event['duration_seconds'] = time.monotonic() - started
    event['record_sha256'] = canonical_hash(event)
    with (attempt / 'trajectory.jsonl').open('a') as handle:
        handle.write(json.dumps(event, sort_keys=True) + '\n')
    return event


@_locked
def finish(attempt, answer_file):
    attempt = Path(attempt); record = _read(attempt / 'attempt.json'); events = _events(attempt)
    if (attempt / 'answer.json').exists():
        raise ValueError('Attempt already finished')
    if any(e['attempt_sha256'] != canonical_hash(record) for e in events):
        raise ValueError('Attempt binding changed')
    if time.time() - record['started_at'] > record['task']['budgets']['latency_seconds']:
        raise ValueError('Attempt deadline exceeded')
    raw = Path(answer_file).read_bytes()
    if len(raw) > record['task']['budgets']['max_submission_bytes']:
        raise ValueError('Submission exceeds byte budget')
    answer = raw.decode()
    if not answer.strip():
        raise ValueError('Submission must contain a nonempty answer')
    result = {'task_id': record['task']['id'], 'answer': answer, 'answer_sha256': _sha(raw),
              'attempt_sha256': canonical_hash(record), 'trajectory_tail': events[-1]['record_sha256'] if events else None,
              'finished_at': time.time(), 'tool_calls': len(events), 'token_usage': None, 'cost_usd': None,
              'output_token_limit_enforced': False, 'output_token_measurement': 'unknown; submission byte limit enforced only',
              'citation_validation': 'pending independent review', 'answer_quality_evaluated': False,
              'training_eligible': False}
    _write(attempt / 'answer.json', result)
    return result


@_locked
def replay(attempt, output, backend=None):
    attempt = Path(attempt); record = _read(attempt / 'attempt.json'); events = _events(attempt)
    backend = backend or (ModalBackend() if record['backend'] == 'modal' else DockerBackend())
    checks = []
    for event in events:
        if event['attempt_sha256'] != canonical_hash(record):
            raise ValueError('Attempt binding changed')
        if 'observation' not in event:
            continue
        command, stdin, _ = _command(record, event['name'], event['args'])
        try:
            actual = backend.run(record['image_id'], command, RunLimits(**event['limits']), stdin=stdin)
            keys = ('stdout', 'stderr', 'exit_code', 'timed_out', 'truncated')
            matched = all(actual.get(k) == event['observation'].get(k) for k in keys)
            checks.append({'sequence': event['sequence'], 'matched': matched, 'observation': actual})
        except Exception as exc:
            checks.append({'sequence': event['sequence'], 'matched': False, 'error': str(exc)})
    result = {'status': 'passed' if checks and all(x['matched'] for x in checks) else 'not_passed',
              'attempt_sha256': canonical_hash(record), 'checks': checks,
              'note': 'Exact observable output replay; nondeterminism is a mismatch. No semantic answer grading.'}
    _write(output, result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__); sub = p.add_subparsers(dest='action', required=True)
    q = sub.add_parser('start'); q.add_argument('--bundle', required=True); q.add_argument('--task', required=True); q.add_argument('--output', required=True)
    q = sub.add_parser('tool'); q.add_argument('--attempt', required=True); q.add_argument('--name', required=True); q.add_argument('--args-json', required=True)
    q = sub.add_parser('finish'); q.add_argument('--attempt', required=True); q.add_argument('--answer-file', required=True)
    q = sub.add_parser('replay'); q.add_argument('--attempt', required=True); q.add_argument('--output', required=True)
    a = p.parse_args()
    if a.action == 'start': result = start(a.bundle, a.task, a.output)
    elif a.action == 'finish': result = finish(a.attempt, a.answer_file)
    elif a.action == 'replay': result = replay(a.attempt, a.output)
    else:
        raw = Path(a.args_json[1:]).read_text() if a.args_json.startswith('@') else a.args_json
        result = tool(a.attempt, a.name, json.loads(raw))
    displayed = result
    if a.action == 'start':
        displayed = {k:v for k,v in result.items() if k != 'snapshot_files'}
        displayed['snapshot_file_count'] = len(result['snapshot_files'])
        displayed['attempt_directory'] = str(Path(a.output).resolve())
    print(json.dumps(displayed, indent=2))
    if result.get('status') in ('error', 'tool_error', 'not_passed'): raise SystemExit(1)


if __name__ == '__main__':
    main()
