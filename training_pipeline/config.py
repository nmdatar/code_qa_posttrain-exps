"""Validate static settings and manifest-backed inputs before remote allocation."""
import hashlib
import math
from pathlib import Path
from .contracts import ConfigurationError
from .storage import read, digest


def exact(value, required, optional=()):
    if not isinstance(value, dict) or set(value) - set(required) - set(optional) or set(required) - set(value):
        raise ConfigurationError('Missing or unknown configuration fields: ' + str(required))


def positive(value, name, integer=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ConfigurationError(name + ' must be positive and finite')
    if integer and type(value) is not int:
        raise ConfigurationError(name + ' must be an integer')


def validate_config(c):
    exact(c, ['schema_version', 'run_id', 'output', 'model', 'seed', 'limits', 'stages',
              'environment', 'evaluation', 'tracking', 'checkpoint_every', 'spend'])
    if not isinstance(c['output'], str) or not c['output']:
        raise ConfigurationError('Output directory is required')
    if c['schema_version'] != '1.0' or not isinstance(c['run_id'], str) or not c['run_id']:
        raise ConfigurationError('Invalid schema or run ID')
    if type(c['seed']) is not int:
        raise ConfigurationError('Seed must be an integer')
    exact(c['model'], ['base_model', 'rank', 'checkpoint_ttl_seconds'])
    for key in ('rank', 'checkpoint_ttl_seconds'):
        positive(c['model'][key], key, True)
    if not c['model']['base_model']:
        raise ConfigurationError('Base model is required')
    keys = ['max_generations', 'max_tokens_per_call', 'max_output_tokens', 'context_tokens',
            'max_tool_calls', 'max_tool_output_bytes', 'latency_seconds', 'provider_timeout_seconds']
    exact(c['limits'], keys)
    for k in keys:
        positive(c['limits'][k], k, True)
    if c['limits']['max_tokens_per_call'] >= c['limits']['context_tokens']:
        raise ConfigurationError('Generation budget must leave room for context')
    positive(c['checkpoint_every'], 'checkpoint_every', True)
    if not isinstance(c['stages'], list) or not c['stages']:
        raise ConfigurationError('At least one training stage required')
    for s in c['stages']:
        exact(s, ['kind', 'max_updates', 'max_batches', 'batch_size', 'learning_rate'], ['group_size', 'temperature'])
        if s['kind'] not in {'sft', 'grpo'}:
            raise ConfigurationError('Unsupported training strategy')
        for k in ('max_updates', 'max_batches', 'batch_size'):
            positive(s[k], k, True)
        positive(s['learning_rate'], 'learning_rate')
        if s['kind'] == 'grpo':
            if type(s.get('group_size')) is not int or s['group_size'] < 2 or s.get('temperature') != 1:
                raise ConfigurationError('GRPO requires group_size >= 2 and temperature=1')
    exact(c['environment'], ['kind'], ['release', 'manifest_sha256', 'calibration', 'calibration_sha256', 'modal_prices'])
    if c['environment']['kind'] not in {'toy', 'repository', 'collection'}:
        raise ConfigurationError('Unsupported environment')
    if c['environment']['kind'] == 'repository' and not all(c['environment'].get(k) for k in
            ('release', 'manifest_sha256', 'calibration', 'calibration_sha256')):
        raise ConfigurationError('Repository environment requires pinned release and calibration')
    if c['environment']['kind'] == 'collection':
        for key in ('release', 'manifest_sha256', 'modal_prices'):
            if not c['environment'].get(key):
                raise ConfigurationError('Collection requires pinned release and Modal prices')
        prices = c['environment']['modal_prices']
        for key in ('cpu_core_second', 'gib_second'):
            positive(prices.get(key), key)
        if not prices.get('source') or not prices.get('checked_at'):
            raise ConfigurationError('Missing Modal price provenance')
        if any(s['kind'] != 'grpo' for s in c['stages']):
            raise ConfigurationError('Collection inputs support GRPO only')
    exact(c['evaluation'], ['every', 'max_tasks', 'temperature'])
    if type(c['evaluation']['every']) is not int or c['evaluation']['every'] < 0:
        raise ConfigurationError('Invalid evaluation cadence')
    positive(c['evaluation']['max_tasks'], 'max_tasks', True)
    if c['evaluation']['temperature'] not in (0, 1):
        raise ConfigurationError('Evaluation temperature must be 0 or 1')
    exact(c['tracking'], ['mode', 'project'])
    if c['tracking']['mode'] not in {'disabled', 'offline', 'online'}:
        raise ConfigurationError('Invalid tracking mode')
    exact(c['spend'], ['cap_usd', 'ledger', 'prices'])
    positive(c['spend']['cap_usd'], 'cap_usd')
    p = c['spend']['prices']
    if p is not None:
        for k in ('prefill', 'sample', 'train', 'storage_gb_month', 'params'):
            positive(p[k], k)
    return c


def verified_file(path, expected):
    path = Path(path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ConfigurationError('Artifact hash mismatch: ' + str(path))
    return path


def load_release(env):
    root = Path(env['release']).resolve()
    manifest = read(verified_file(root / 'manifest.json', env['manifest_sha256']))
    if manifest.get('training_eligible') is not True:
        raise ConfigurationError('Release is not training eligible')
    result = {}
    for kind, rel in [('sft', 'sft/train.jsonl'), ('tasks', 'rl/train.jsonl'), ('development', 'evaluation/development/tasks.jsonl')]:
        path = (root / rel).resolve()
        if not path.is_relative_to(root):
            raise ConfigurationError('Release path escapes root')
        verified_file(path, manifest['artifacts'][rel])
        import json
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if len({r['id'] for r in rows}) != len(rows):
            raise ConfigurationError('Duplicate example IDs')
        expected_split = 'development' if kind == 'development' else 'train'
        if any(r.get('split') != expected_split for r in rows):
            raise ConfigurationError('Release split mismatch')
        if kind == 'sft' and any(r.get('status') != 'accepted' for r in rows):
            raise ConfigurationError('Unaccepted SFT example')
        result[kind] = rows
    families, lineages = {}, {}
    for rows in result.values():
        for r in rows:
            for mapping, key in ((families, 'family_id'), (lineages, 'lineage_id')):
                ident = r[key]
                if ident in mapping and mapping[ident] != r['split']:
                    raise ConfigurationError('Repository family or lineage crosses splits')
                mapping[ident] = r['split']
    result['identity'] = digest(manifest)
    return result


def inputs(config):
    validate_config(config)
    if config['environment']['kind'] == 'toy':
        from .toy import toy_inputs
        data = toy_inputs()
    elif config['environment']['kind'] == 'collection':
        from .collection import load_collection
        data = load_collection(config['environment'])
    else:
        data = load_release(config['environment'])
        calibration = read(verified_file(config['environment']['calibration'], config['environment']['calibration_sha256']))
        if calibration.get('status') != 'passed' or calibration.get('reviewed_tasks', 0) < 1:
            raise ConfigurationError('Repository verifier lacks passed human calibration')
        from .repository import validate_repository_task
        for row in data['tasks'] + data['development']:
            validate_repository_task(row)
    for s in config['stages']:
        if not data['sft' if s['kind'] == 'sft' else 'tasks']:
            raise ConfigurationError('Training input is empty for ' + s['kind'])
    if not data['development']:
        raise ConfigurationError('Development evaluation input is empty')
    for example in data['sft']:
        messages = example.get('messages')
        if not isinstance(messages, list) or not messages:
            raise ConfigurationError('SFT requires visible messages')
        for message in messages:
            if not isinstance(message, dict) or set(message) != {'role', 'content'} or message['role'] not in {'system', 'user', 'assistant', 'tool'} or not isinstance(message['content'], str):
                raise ConfigurationError('Invalid SFT role/content message')
    return data
