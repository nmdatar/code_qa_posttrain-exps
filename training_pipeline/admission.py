"""Automated experimental rollout admission; never asserts human gold/calibration.

Collection exports preserve their native environment and reference contracts.
They are rollout inputs, not strict TaskSpec training releases.
"""
from collections import Counter
from functools import lru_cache
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tempfile

POLICY = 'automated-source-rollout-v1'
READ_TOOLS = {'list_files', 'search_code', 'read_file'}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def index(path, key):
    result = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row[key] in result:
            raise ValueError('Duplicate identity: ' + row[key])
        result[row[key]] = row
    return result


def verified_source(root):
    root = Path(root).resolve()
    manifest = read(root / 'manifest.json')
    artifacts = manifest['artifacts']
    for name, spec in artifacts.items():
        path = (root / name).resolve()
        expected = spec if isinstance(spec, str) else spec['sha256']
        if not path.is_relative_to(root) or sha(path.read_bytes()) != expected:
            raise ValueError('Source artifact mismatch: ' + name)
    return manifest


@lru_cache(maxsize=4096)
def blob(snapshot, commit, path):
    if PurePosixPath(path).is_absolute() or '..' in PurePosixPath(path).parts:
        raise ValueError('Unsafe evidence path')
    import os
    if os.environ.get('QA_MODAL_WORKER') == '1':
        mapping = read(os.environ['QA_SNAPSHOT_MAP'])
        if snapshot not in mapping:
            raise ValueError('Snapshot was not staged for remote execution')
        snapshot = mapping[snapshot]
    return subprocess.run(['git', '--no-replace-objects', '-C', snapshot, 'show', commit + ':' + path],
                          capture_output=True, check=True, timeout=30).stdout


def assess(task, quality, grading, environment, evidence):
    """Assess selected reference (which may correct a rejected original)."""
    reasons = []
    if not task.get('user_prompt', '').strip():
        reasons.append('empty_question')
    if not environment or environment.get('repository') != task['repository'] or not environment.get('runtime_checked'):
        reasons.append('environment_binding_or_runtime')
    if quality.get('runtime_status') != 'passed':
        reasons.append('task_runtime')
    if not grading or not quality.get('grading_reference_available'):
        reasons.append('missing_or_quarantined_reference')
    admission = quality.get('reference_admission')
    supported = ((admission == 'source_reviewed_original_draft' and quality.get('semantic_review') == 'supported') or
                 (admission == 'independently_reviewed_correction_draft' and quality.get('correction_review') == 'supported') or
                 (admission == 'previously_reviewed_local_draft' and quality.get('semantic_review') == 'supported_previous_independent_review'))
    if not supported:
        reasons.append('selected_reference_not_supported')
    if environment and environment.get('capability') == 'source_reading':
        if not set(task['permitted_tools']) <= READ_TOOLS or quality.get('execution_requirement_screen') != 'no_explicit_execution_request_detected':
            reasons.append('execution_requirement_in_source_only_environment')
    if evidence:
        result, isolation, attestation, source = (evidence[k] for k in ('result', 'isolation', 'attestation', 'source'))
        image = environment.get('image_id') if environment else None
        if (not image or isolation.get('passed') is not True or attestation.get('status') != 'passed' or
                isolation.get('image_id') != image or attestation.get('image_id') != image or
                result.get('image_digest') != image or result.get('commit') != task['repository']['commit'] or
                source.get('repository') != task['repository']):
            reasons.append('environment_evidence_binding')
        try:
            if grading and grading.get('kind') == 'source_reference_comparison':
                answer = grading['reference_answer']
                if sha(answer.encode()) != grading['reference_sha256'] or quality.get('selected_reference_sha256', quality.get('reference_sha256')) != grading['reference_sha256']:
                    reasons.append('selected_reference_hash')
                if quality.get('question_sha256') != sha(task['user_prompt'].encode()):
                    reasons.append('question_review_binding')
                refs = grading.get('verified_evidence', [])
            else:
                record = (grading or {}).get('record', {})
                if record.get('question') != task['user_prompt'] or record.get('repository') != task['repository']:
                    reasons.append('local_reference_binding')
                refs = [ref for claim in record.get('claims', []) for ref in claim['evidence']]
            if not refs:
                reasons.append('no_source_evidence')
            for ref in refs:
                data = blob(source['snapshot_path'], task['repository']['commit'], ref['path'])
                if (sha(data) != ref['file_sha256'] or result['snapshot_files'].get(ref['path']) != ref['file_sha256'] or
                        not 1 <= ref['start_line'] <= ref['end_line'] <= len(data.splitlines())):
                    reasons.append('source_evidence_mismatch')
                    break
        except (KeyError, ValueError, OSError, subprocess.SubprocessError):
            reasons.append('source_evidence_unverifiable')
    else:
        reasons.append('environment_evidence_missing')
    return sorted(set(reasons))


def export_rollouts(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    manifest = verified_source(source)
    if output.exists():
        raise ValueError('Output already exists; use a new release path')
    tasks = index(source / 'public/tasks.jsonl', 'id')
    qualities = index(source / 'private/quality.jsonl', 'task_id')
    grading = index(source / 'private/grading.jsonl', 'task_id')
    envs = index(source / 'public/environments.jsonl', 'environment_id')
    evidence = {}
    for eid in envs:
        base = source / 'private/environment-evidence' / eid
        names = {'result': 'result.json', 'isolation': 'isolation-report.json',
                 'attestation': 'image-attestation.json', 'source': 'source-environment.json'}
        if all((base / n).exists() and str((base / n).relative_to(source)) in manifest['artifacts'] for n in names.values()):
            evidence[eid] = {k: read(base / n) for k, n in names.items()}
    decisions, accepted = [], []
    seen = set()
    for ident, task in sorted(tasks.items()):
        reasons = assess(task, qualities.get(ident, {}), grading.get(ident), envs.get(task['environment_id']), evidence.get(task['environment_id']))
        fingerprint = (task['repository']['family_id'].casefold(), ' '.join(task['user_prompt'].split()).casefold())
        if fingerprint in seen:
            reasons.append('duplicate_question')
        if not reasons:
            seen.add(fingerprint)
            accepted.append(task)
        decisions.append({'task_id': ident, 'rollout_ready': not reasons, 'reasons': reasons,
                          'selected_reference_admission': qualities.get(ident, {}).get('reference_admission')})
    if not accepted:
        raise ValueError('No tasks passed automated admission')
    families = sorted({t['repository']['family_id'].casefold() for t in accepted}, key=lambda f: sha((POLICY + f).encode()))
    # Previously exercised local families remain development-only.
    heldout = {f for f in families if f in {'pydantic/pydantic', 'tanstack/query', 'sqlalchemy/sqlalchemy'}}
    for family in families:
        if len(heldout) >= max(1, round(len(families) * .2)):
            break
        heldout.add(family)
    public = [{**t, 'split': 'development' if t['repository']['family_id'].casefold() in heldout else 'train'} for t in accepted]
    if not any(t['split'] == 'train' for t in public):
        raise ValueError('At least two families required for train/development separation')
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='.rollout-export-', dir=output.parent))
    def write(name, value, jsonl=False):
        dest = stage / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(''.join(json.dumps(r, sort_keys=True) + '\n' for r in value) if jsonl else json.dumps(value, indent=2, sort_keys=True) + '\n')
    try:
        write('public/tasks.jsonl', public, True)
        used = {t['environment_id'] for t in public}
        write('public/environments.jsonl', [envs[e] for e in sorted(used)], True)
        write('rl/train.jsonl', [t for t in public if t['split'] == 'train'], True)
        write('evaluation/development/tasks.jsonl', [t for t in public if t['split'] == 'development'], True)
        write('private/grading.jsonl', [grading[t['id']] for t in public], True)
        write('private/admission.jsonl', decisions, True)
        write('private/source-manifest.json', manifest)
        # Retain portable source attestations and license attribution with the export.
        for name in manifest['artifacts']:
            if name.startswith('attribution/') or name.startswith('private/environment-evidence/') and name.split('/')[2] in used:
                dest = stage / name
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source / name, dest)
        counts = dict(Counter(t['split'] for t in public))
        report = {'policy': POLICY, 'source_tasks': len(tasks), 'admitted': len(public), 'excluded': len(tasks)-len(public),
                  'splits': counts, 'train_families': len(families)-len(heldout), 'development_families': len(heldout),
                  'exclusion_reasons': dict(Counter(r for d in decisions for r in d['reasons'])),
                  'human_reviewed': False, 'reward_calibrated': False, 'new_live_environment_check': False,
                  'strict_training_pipeline_compatible': False,
                  'next_integration': 'Collection-native rollout adapter and provisional reference-comparison reward; strict human-gold adapter does not consume this schema.'}
        write('admission-report.json', report)
        write('manifest.json', {'schema_version': 'experimental-rollout-release-1.0', 'policy': POLICY,
              'source_manifest_sha256': sha((source / 'manifest.json').read_bytes()),
              'source_release': manifest.get('release_id'), 'human_reviewed': False,
              'rollout_eligible': True, 'training_eligible': False, 'experimental_rl_task_eligible': True,
              'reward_status': 'automatically_reviewed_reference_uncalibrated',
              'benchmark_repurposed': True, 'final_test_claim': False, 'counts': counts,
              'artifacts': {str(p.relative_to(stage)): sha(p.read_bytes()) for p in sorted(stage.rglob('*')) if p.is_file()}})
        stage.rename(output)
    except BaseException:
        shutil.rmtree(stage)
        raise
    return report
