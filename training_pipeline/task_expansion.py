"""Frozen, training-only complement of an already evaluated cohort."""
from .contracts import ConfigurationError
from .storage import digest

VERSION = 'training-task-expansion-v1'


def create(data, previous):
    from .benchmark import task_manifest
    if previous != task_manifest(data, len(previous['task_ids'])):
        raise ConfigurationError('Previous cohort identity mismatch')
    full = task_manifest(data, len(data['tasks']))
    prior = set(previous['task_ids'])
    value = {'kind': VERSION, 'data_identity': data['identity'],
             'previous_manifest': previous, 'full_manifest': full,
             'task_ids': [i for i in full['task_ids'] if i not in prior]}
    if not value['task_ids']:
        raise ConfigurationError('No remaining training tasks')
    return {**value, 'manifest_hash': digest(value)}


def validate(manifest, config, data):
    if manifest != create(data, manifest['previous_manifest']):
        raise ConfigurationError('Expansion cohort changed')
    if config['benchmark']['manifest_hash'] != manifest['manifest_hash']:
        raise ConfigurationError('Expansion hash mismatch')
    if config.get('execution', {}).get('operation') != 'benchmark':
        raise ConfigurationError('Expansion is sampling-only')
    indexed = {t['id']: t for t in data['tasks']}
    tasks = [indexed[i] for i in manifest['task_ids']]
    if any(t['split'] != 'train' for t in tasks):
        raise ConfigurationError('Expansion must be training-only')
    return tasks
