"""Reuse a verified immutable environment, never task-specific verification."""
import argparse
import hashlib
import json
from pathlib import Path
import re

from .build import write_json
from .contracts import canonical_hash
from .environment import _export_snapshot
from qa_eval.deterministic import snapshot


def reuse_environment(source, target):
    source, target = Path(source), Path(target)
    def load(path):
        return json.loads(path.read_text())
    for root in (source, target):
        manifest = load(root / 'manifest.json')
        for relative, digest in manifest['artifacts'].items():
            path = root / relative
            if not path.resolve().is_relative_to(root.resolve()):
                raise ValueError('Artifact escapes bundle')
            if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise ValueError('Artifact changed: ' + relative)
    environment = load(target / 'public/environment.json')
    if environment != load(source / 'public/environment.json'):
        raise ValueError('Repository, recipe, snapshot and environment must match exactly')
    built = load(source / 'environment-build/result.json')
    if built.get('status') != 'ready' or built.get('source_environment_sha256') != canonical_hash(environment):
        raise ValueError('Environment is unready or unbound')
    image = built.get('image_id', built.get('image_digest', ''))
    if not (built.get('backend') == 'modal' and re.fullmatch(r'im-[A-Za-z0-9]+', image)
            or built.get('backend') == 'docker' and re.fullmatch(r'sha256:[a-f0-9]{64}', image)):
        raise ValueError('Immutable image reference required')
    readiness = built.get('readiness', {})
    if readiness.get('exit_code') != 0 or readiness.get('timed_out') or readiness.get('truncated'):
        raise ValueError('Readiness must pass')
    checkout = Path(environment['snapshot_path'])
    snapshot(checkout, built['commit'])
    files, _ = _export_snapshot(checkout, built['commit'])
    if files != built.get('snapshot_files'):
        raise ValueError('Snapshot differs from image')
    reports = {}
    for name, key, expected in [('isolation-report.json', 'passed', True),
                                ('image-attestation.json', 'status', 'passed')]:
        report = load(source / 'private' / name)
        if report.get(key) != expected or report.get('environment_hash') != canonical_hash(built):
            raise ValueError('Environment evidence missing or stale: ' + name)
        reports[name] = report
    destination = target / 'environment-build'
    if destination.exists():
        raise ValueError('Target already has a build; do not overwrite')
    # Copy only environment evidence, never task assertions or their pass reports.
    for name in ('result.json', 'environment.json', 'recipe.json'):
        write_json(destination / name, load(source / 'environment-build' / name))
    for name, report in reports.items():
        write_json(target / 'private' / name, report)
    result = {'status': 'reused_immutable_environment', 'source_bundle': str(source.resolve()),
              'environment_hash': canonical_hash(built), 'image_id': image,
              'task_verification_reused': False,
              'environment_evidence': 'Reused checks bound to the same immutable image; not new executions'}
    write_json(target / 'private/environment-reuse.json', result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--target', type=Path, required=True)
    args = p.parse_args()
    print(json.dumps(reuse_environment(args.source, args.target), indent=2))


if __name__ == '__main__':
    main()
