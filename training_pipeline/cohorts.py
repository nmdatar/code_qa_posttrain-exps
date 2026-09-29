"""Deterministic, integrity-checked selection and confirmation manifests."""
import hashlib
from collections import defaultdict
from pathlib import Path

from .contracts import ConfigurationError
from .storage import digest, read

SEED = 'qwen4b-procedures-v1'


def build_cohorts(data, selection_size=32, seed=SEED):
    rows = data['development']
    if type(selection_size) is not int or not 0 < selection_size < len(rows):
        raise ConfigurationError('Selection size must leave a nonempty confirmation cohort')
    families = defaultdict(list)
    train_ids = {r['id'] for r in data['tasks']}
    train_families = {r['family_id'].casefold() for r in data['tasks']}
    seen = set()
    for row in rows:
        family = row['family_id'].casefold()
        if row['id'] in seen or row['id'] in train_ids or family in train_families or row['split'] != 'development':
            raise ConfigurationError('Cohort duplicate or training/development leakage')
        seen.add(row['id'])
        families[family].append(row['id'])
    n = len(rows)
    allocation = {f: len(ids)*selection_size//n for f, ids in families.items()}
    ranking = sorted(families, key=lambda f: (-(len(families[f])*selection_size % n), f))
    for family in ranking[:selection_size-sum(allocation.values())]:
        allocation[family] += 1
    selection, confirmation = [], []
    for family in sorted(families):
        ids = sorted(families[family], key=lambda ident: (
            hashlib.sha256(f'{seed}|{family}|{ident}'.encode()).hexdigest(), ident))
        selection.extend(ids[:allocation[family]])
        confirmation.extend(ids[allocation[family]:])
    manifest = {'schema_version': 1, 'data_identity': data['identity'], 'seed': seed,
                'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'selection': selection, 'confirmation': confirmation,
                'families': {f: {'total': len(families[f]), 'selection': allocation[f]} for f in sorted(families)}}
    return {**manifest, 'manifest_hash': digest(manifest)}


def load_cohorts(path, expected_hash, data):
    manifest = read(path)
    content = {k: v for k, v in manifest.items() if k != 'manifest_hash'}
    if manifest.get('manifest_hash') != expected_hash or digest(content) != expected_hash:
        raise ConfigurationError('Cohort manifest integrity failure')
    rebuilt = build_cohorts(data, len(manifest['selection']), manifest['seed'])
    for key in ('schema_version', 'data_identity', 'selection', 'confirmation', 'families'):
        if manifest[key] != rebuilt[key]:
            raise ConfigurationError('Cohort membership or dataset changed')
    return manifest


def select_tasks(data, evaluation, cohort=None):
    if 'cohort_manifest' not in evaluation:
        if cohort is not None:
            raise ConfigurationError('Explicit cohort requires a pinned cohort manifest')
        return data['development'][:evaluation['max_tasks']], None
    manifest = load_cohorts(evaluation['cohort_manifest'], evaluation['cohort_sha256'], data)
    name = cohort or 'selection'
    if name not in ('selection', 'confirmation'):
        raise ConfigurationError('Unknown evaluation cohort')
    indexed = {r['id']: r for r in data['development']}
    # Explicit cohorts are never silently truncated by the legacy prefix limit.
    return [indexed[ident] for ident in manifest[name]], {
        'name': name, 'manifest_hash': manifest['manifest_hash']}
