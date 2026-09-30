"""Build and check collection source environments with independent resumable workers.

The source runtime canary exercises tools only. It does not generate an answer,
collect a teacher investigation, or validate an upstream reference answer.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing
from pathlib import Path
import re
import time
import uuid

from . import attest, build, investigate, isolation
from .contracts import canonical_hash
from .environment import _export_snapshot
from qa_eval.deterministic import snapshot


def _read(path):
    return json.loads(Path(path).read_text())


def _save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    temp.replace(path)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _bound(report, built):
    return (report.get('environment_hash') == canonical_hash(built)
            and report.get('image_id') == built.get('image_digest'))


def _ready(built, environment):
    readiness = built.get('readiness', {})
    return (built.get('status') == 'ready' and built.get('backend') == 'modal'
            and re.fullmatch(r'im-[A-Za-z0-9]+', built.get('image_digest', ''))
            and built.get('commit') == environment['repository']['commit']
            and built.get('source_environment_sha256') == canonical_hash(environment)
            and readiness.get('exit_code') == 0 and not readiness.get('timed_out')
            and not readiness.get('truncated') and bool(built.get('snapshot_files')))


def _load_optional(path):
    try:
        return _read(path)
    except (OSError, ValueError):
        return {}


def _select_text(environment, built):
    root = Path(environment['snapshot_path']).resolve()
    candidates = sorted(built['snapshot_files'], key=lambda p: (not p.lower().startswith('readme'), len(p), p))
    for relative in candidates:
        path = root / relative
        if not path.resolve().is_relative_to(root) or not path.is_file() or path.stat().st_size > 1000000:
            continue
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != built['snapshot_files'][relative] or b'\0' in raw:
            continue
        try:
            lines = raw.decode('utf-8').splitlines()
        except UnicodeDecodeError:
            continue
        for number, line in enumerate(lines[:100], 1):
            if len(line.strip()) >= 8 and len(line) < 2000:
                return relative, number, line.strip()[:80]
    raise ValueError('No verified nonempty text file available for runtime canary')


def _runtime_reusable(report, built, bundle):
    if not (_bound(report, built) and report.get('status') == 'passed'
            and report.get('public_manifest_sha256') == _sha(bundle / 'manifest.json')
            and report.get('tool_calls') == 3 and report.get('replayed_operations') == 3):
        return False
    artifacts = report.get('artifacts', {})
    if not artifacts or not all((bundle / p).is_file() and _sha(bundle / p) == h for p, h in artifacts.items()):
        return False
    return True


def check_source_runtime(bundle, built, environment):
    bundle = Path(bundle)
    report_path = bundle / 'private/source-runtime-check.json'
    cached = _load_optional(report_path)
    if _runtime_reusable(cached, built, bundle):
        return cached
    public = build.read_jsonl(bundle / 'public/tasks.jsonl')
    if not public:
        raise ValueError('No public task for source runtime check')
    attempt = bundle / 'private/source-runtime-attempts' / uuid.uuid4().hex
    relative, number, query = _select_text(environment, built)
    record = investigate.start(bundle, public[0]['id'], attempt)
    events = []
    calls = [('list_files', {'path': relative, 'limit': 5}, 'files'),
             ('search_code', {'path': relative, 'query': query, 'limit': 5}, 'matches'),
             ('read_file', {'path': relative, 'start_line': number, 'end_line': number}, 'lines')]
    for name, args, field in calls:
        event = investigate.tool(attempt, name, args)
        events.append(event)
        if event['status'] != 'ok':
            raise ValueError('Source runtime operation failed: ' + name)
        observed = json.loads(event['observation']['stdout'])
        if not observed.get(field):
            raise ValueError('Empty source runtime output: ' + name)
    replay_path = attempt / 'replay.json'
    replay = investigate.replay(attempt, replay_path)
    if replay['status'] != 'passed' or len(replay['checks']) != 3:
        raise ValueError('Source runtime exact replay failed')
    files = [attempt / name for name in ('attempt.json', 'trajectory.jsonl', 'replay.json')]
    report = {'status': 'passed', 'environment_hash': canonical_hash(built),
              'image_id': built['image_digest'], 'public_manifest_sha256': _sha(bundle / 'manifest.json'),
              'task_id': record['task']['id'], 'attempt_path': str(attempt.relative_to(bundle)),
              'tool_calls': len(events), 'replayed_operations': len(replay['checks']),
              'operations': [{'name': e['name'], 'status': e['status'], 'record_sha256': e['record_sha256'],
                              'sandbox_id': e['observation'].get('sandbox_id')} for e in events],
              'replay_status': replay['status'], 'replay_sha256': _sha(replay_path),
              'artifacts': {str(p.relative_to(bundle)): _sha(p) for p in files},
              'answer_generated': False, 'teacher_investigation': False,
              'scope': 'Actual nonempty source tool observations and exact clean-sandbox replay; no semantic grading.'}
    _save(report_path, report)
    return report


def process_bundle(bundle):
    bundle = Path(bundle).resolve()
    status_path = bundle / 'private/collection-environment-status.json'
    status = {'bundle': str(bundle), 'status': 'running', 'phase': 'validate', 'started_at': time.time()}
    def progress(phase):
        status['phase'] = phase
        _save(status_path, status)
        print(json.dumps({'bundle': bundle.name, 'phase': phase}), flush=True)
    try:
        progress('validate')
        manifest = _read(bundle / 'manifest.json')
        for path in ('public/tasks.jsonl', 'public/environment.json'):
            if _sha(bundle / path) != manifest['artifacts'].get(path):
                raise ValueError('Public manifest hash mismatch: ' + path)
        environment = _read(bundle / 'public/environment.json')
        built = _load_optional(bundle / 'environment-build/result.json')
        if not _ready(built, environment):
            progress('build')
            built = build.build_environment_bundle(bundle, backend='modal')
        if not _ready(built, environment):
            raise ValueError('Environment did not become ready with immutable bound image')
        checkout = Path(environment['snapshot_path'])
        hashes = investigate.validated_snapshot_files(checkout, built)
        if hashes != built['snapshot_files']:
            raise ValueError('Cached image source differs from pinned repository snapshot')
        status.update(image_id=built['image_digest'], environment_hash=canonical_hash(built))
        progress('isolation')
        observed = _load_optional(bundle / 'private/isolation-report.json')
        if not (_bound(observed, built) and observed.get('passed') and len(observed.get('checks', [])) == 2):
            observed = isolation.check_isolation(bundle)
        if not observed.get('passed') or not _bound(observed, built):
            raise ValueError('Isolation checks failed')
        progress('attestation')
        observed = _load_optional(bundle / 'private/image-attestation.json')
        if not (_bound(observed, built) and observed.get('status') == 'passed'
                and observed.get('observation', {}).get('source_files_checked') == len(built['snapshot_files'])):
            observed = attest.attest(bundle)
        if observed.get('status') != 'passed' or not _bound(observed, built):
            raise ValueError('Image source attestation failed')
        progress('source_runtime')
        runtime = check_source_runtime(bundle, built, environment)
        status.update(status='passed', phase='complete', task_count=len(build.read_jsonl(bundle / 'public/tasks.jsonl')),
                      runtime_report='private/source-runtime-check.json', replayed_operations=runtime['replayed_operations'])
    except Exception as exc:
        status.update(status='quarantined', error=f'{type(exc).__name__}: {exc}')
    status['finished_at'] = time.time()
    _save(status_path, status)
    print(json.dumps({'bundle': bundle.name, 'status': status['status'], 'error': status.get('error')}), flush=True)
    return status


def run(bundles_root, workers, output):
    if workers < 1:
        raise ValueError('workers must be positive')
    bundles = sorted(p.parent for p in Path(bundles_root).glob('*/manifest.json'))
    if not bundles:
        raise ValueError('No bundles found')
    report = {'schema_version': '1', 'status': 'running', 'bundles_root': str(Path(bundles_root).resolve()),
              'expected_environments': len(bundles), 'workers': workers, 'environments': [],
              'scope': 'Source-reading runtime readiness, isolation, attestation and replay; no teacher answers.'}
    _save(output, report)
    # Spawn rather than fork so each worker initializes its own Modal client state.
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        futures = {pool.submit(process_bundle, str(p)): p for p in bundles}
        for future in as_completed(futures):
            bundle = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {'bundle': str(bundle.resolve()), 'status': 'quarantined',
                          'error': f'Worker failure: {type(exc).__name__}: {exc}'}
                _save(bundle / 'private/collection-environment-status.json', result)
            report['environments'].append(result)
            report['environments'].sort(key=lambda r: r['bundle'])
            report['passed'] = sum(r['status'] == 'passed' for r in report['environments'])
            report['quarantined'] = sum(r['status'] == 'quarantined' for r in report['environments'])
            _save(output, report)
    report['status'] = 'passed' if report['passed'] == len(bundles) else 'completed_with_quarantine'
    _save(output, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundles-root', default='data/generated/collection-1000-v1', type=Path)
    parser.add_argument('--workers', default=3, type=int)
    parser.add_argument('--output', default='reports/task-generation-1000/environments.json', type=Path)
    args = parser.parse_args()
    result = run(args.bundles_root, args.workers, args.output)
    if result['status'] != 'passed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
