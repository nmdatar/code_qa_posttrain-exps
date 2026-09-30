"""Pinned training subset and validation subset of an existing frozen cohort."""
from .contracts import ConfigurationError
from .storage import read, digest


def apply_subset(config, data):
    from .config import verified_file
    from .cohorts import load_cohorts
    spec = config['task_subset']
    manifest = read(verified_file(spec['manifest'], spec['sha256']))
    if manifest.get('kind') != 'selected-task-pilot-v1' or manifest.get('data_identity') != data['identity']:
        raise ConfigurationError('Task subset data identity mismatch')
    pool = manifest['training_pool']
    if digest({k:v for k,v in pool.items() if k != 'manifest_hash'}) != pool.get('manifest_hash') or pool.get('data_identity') != data['identity']:
        raise ConfigurationError('Selected training pool changed')
    train, evaluation = manifest['training_ids'], manifest['evaluation_ids']
    if not train or not evaluation or len(set(train)) != len(train) or len(set(evaluation)) != len(evaluation):
        raise ConfigurationError('Empty or duplicate subset IDs')
    indexed = {t['id']:t for t in data['tasks']}
    cohorts = load_cohorts(config['evaluation']['cohort_manifest'], config['evaluation']['cohort_sha256'], data)
    if not set(train) <= set(pool['task_ids']) or not set(train) <= set(indexed):
        raise ConfigurationError('Training subset escapes selected training pool')
    if not set(evaluation) <= set(cohorts['selection']) or set(evaluation) & set(train):
        raise ConfigurationError('Evaluation subset escapes validation selection')
    if any(indexed[i]['split'] != 'train' for i in train):
        raise ConfigurationError('Training subset contains held-out tasks')
    result = dict(data)
    # Loading verifies the complete source release before filtering. The remote
    # bundle must retain its snapshots even when only a subset is trained.
    result['snapshot_tasks'] = data['tasks']
    result['tasks'] = [indexed[i] for i in train]
    result['evaluation_subset'] = {'task_ids': evaluation, 'sha256': spec['sha256']}
    return result
