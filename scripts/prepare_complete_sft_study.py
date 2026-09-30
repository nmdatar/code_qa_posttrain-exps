"""Audit and freeze current strictly verified complete training investigations offline.

Never launches providers. A small release is a provenance artifact, not evidence
that a reportable SFT study is ready. Never repeat examples to satisfy the gate.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil

from training_pipeline.collection import CollectionFactory, load_collection
from training_pipeline.investigation_sft import VERSION, candidate
from training_pipeline.sft_collection import validate_rendering
from training_pipeline.storage import atomic_json, digest


def audit(paths, tasks, reward_version, renderer):
    counts = Counter()
    rejected = Counter()
    choices = {}
    seen = set()
    for path in sorted(paths):
        raw = path.read_bytes()
        checksum = hashlib.sha256(raw).hexdigest()
        if checksum in seen:
            counts['duplicate_archive_files'] += 1
            continue
        seen.add(checksum)
        trace = json.loads(raw)
        if trace.get('split') != 'train' or trace.get('task_id') not in tasks:
            continue
        counts['unique_training_trajectories'] += 1
        verification = trace.get('verification') or {}
        diagnostics = verification.get('diagnostics') or {}
        if verification.get('version') != reward_version:
            counts['other_grading_identity'] += 1
            continue
        counts['matched_grading_identity'] += 1
        if verification.get('reward') == 1 or diagnostics.get('training_reward') == 1:
            counts['perfect_training_reward_not_admission'] += 1
        if diagnostics.get('strict_score') != 1:
            counts['not_strict_pass'] += 1
            continue
        counts['strict_pass_candidates'] += 1
        try:
            example = candidate(trace, tasks[trace['task_id']], reward_version)
            validate_rendering(renderer, [example])
        except (ValueError, KeyError, TypeError, IndexError) as exc:
            rejected[str(exc)] += 1
            continue
        counts['complete_verified_candidates'] += 1
        tokens = sum(len(proof['generation']['tokens']) for proof in example['proofs'])
        rank = (tokens, checksum, str(path))
        lineage = example['lineage_id']
        if lineage not in choices or rank < choices[lineage][0]:
            choices[lineage] = (rank, example, path)
    return choices, counts, rejected


def prepare(config_path, archive_root, destination, min_lineages=200, min_families=15, trajectory_paths=None):
    root = Path(destination)
    if root.exists():
        raise ValueError('Use a fresh audit/release directory; never overwrite frozen data')
    config = json.loads(Path(config_path).read_text())
    data = load_collection(config['environment'])
    tasks = {row['id']: row for row in data['tasks']}
    from transformers import AutoTokenizer
    from training_pipeline.rendering import ChatRenderer
    renderer = ChatRenderer(AutoTokenizer.from_pretrained(config['model']['base_model'], local_files_only=True), config['limits']['context_tokens'])
    reward_version = CollectionFactory(config, root, None).reward_version
    paths = Path(archive_root).glob('**/trajectories/*.json') if trajectory_paths is None else trajectory_paths
    choices, counts, rejected = audit(paths, tasks, reward_version, renderer)
    (root / 'sources').mkdir(parents=True)
    records = []
    family_counts = Counter()
    policies = Counter()
    for _, example, path in sorted(choices.values(), key=lambda item: item[1]['id']):
        target = root / 'sources' / (example['id'] + '.json')
        shutil.copyfile(path, target)
        family_counts[example['family_id']] += 1
        policies[example['source_policy_id']] += 1
        records.append({'source': str(target.relative_to(root)), 'sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
                        'example_hash': digest(example), 'task_id': example['task_id'], 'lineage_id': example['lineage_id'],
                        'family_id': example['family_id'], 'original_path': str(path), 'selected_turns': example['assistant_turns']})
    manifest = {'version': VERSION, 'data_identity': data['identity'], 'grading_version': reward_version,
                'human_reviewed': False, 'records': records,
                'scope': 'Complete strictly passing policy-generated investigations; not human/expert gold',
                'selection': 'Shortest native-token verified complete trajectory per training lineage; no evaluation trajectories'}
    atomic_json(root / 'manifest.json', manifest)
    report = {'version': 'complete-sft-study-audit-v1', 'config': str(config_path), 'data_identity': data['identity'],
              'grading_version': reward_version, 'counts': dict(counts), 'rejections': dict(rejected),
              'admitted_lineages': len(records), 'family_counts': dict(family_counts), 'source_policies': dict(policies),
              'target_tokens': sum(rank[0] for rank, _, _ in choices.values()),
              'readiness_gate': {'min_unique_training_lineages': min_lineages, 'min_families': min_families,
                                 'passed': len(records) >= min_lineages and len(family_counts) >= min_families},
              'manifest_sha256': hashlib.sha256((root / 'manifest.json').read_bytes()).hexdigest(),
              'new_teacher_calls': 0, 'confirmation_used': False,
              'limitations': ['Automated judge is uncalibrated; source/citation checks do not replace semantic audit.',
                             'Readiness gate is a data-diversity threshold, not a statistical-power guarantee.',
                             'Current training factual reward alone never qualifies a demonstration.']}
    atomic_json(root / 'audit.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/experiments/reinforce-v6/run.json')
    parser.add_argument('--archive-root', default='artifacts')
    parser.add_argument('--destination', default='data/sft/complete-investigations-expanded-v1')
    parser.add_argument('--min-lineages', type=int, default=200)
    parser.add_argument('--min-families', type=int, default=15)
    args = parser.parse_args()
    print(json.dumps(prepare(args.config, args.archive_root, args.destination, args.min_lineages, args.min_families), indent=2))
