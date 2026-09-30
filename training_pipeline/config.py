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
              'environment', 'evaluation', 'tracking', 'checkpoint_every', 'spend'], ['judge', 'group_retries', 'stopping', 'concurrency', 'benchmark', 'execution', 'best_checkpoint', 'training_reward', 'supervised', 'harness', 'task_subset', 'prompt_decomposition'])
    if 'prompt_decomposition' in c:
        from .prompt_decomposition import VERSION
        if (c['prompt_decomposition'] != {'version':VERSION}
                or c['environment']['kind'] != 'collection' or 'task_subset' not in c
                or c.get('judge', {}).get('base_model') != 'Qwen/Qwen3.5-397B-A17B'):
            raise ConfigurationError('Question decomposition requires pinned Qwen397 and task subset')
    if 'task_subset' in c:
        exact(c['task_subset'], ['manifest', 'sha256'])
        if c['environment']['kind'] != 'collection' or 'cohort_manifest' not in c['evaluation']:
            raise ConfigurationError('Task subset requires collection and a frozen evaluation cohort')
    if 'harness' in c:
        from .harness_variants import validate
        validate(c['harness'])
        if c['environment']['kind'] != 'collection':
            raise ConfigurationError('Experimental harness requires collection')
    if 'supervised' in c:
        exact(c['supervised'], ['manifest', 'sha256'], ['loss_scope'])
        if c['supervised'].get('loss_scope', 'all-admitted-turns') not in {'all-admitted-turns', 'final-answer-only'}:
            raise ConfigurationError('Unsupported supervised loss scope')
        if c['environment']['kind'] != 'collection':
            raise ConfigurationError('Experimental supervised adapter requires collection')
    if 'training_reward' in c:
        from .shaped_reward import VERSION as reward_version
        from .efficiency_reward import VERSIONS as tier_versions
        from .reward_alignment import VERSION as aligned_version
        exact(c['training_reward'], ['version'], ['efficiency_penalty'])
        if 'efficiency_penalty' in c['training_reward']:
            if (c['training_reward']['efficiency_penalty'] != 'output-token-fraction-v1'
                    or c['training_reward']['version'] != 'positive-coverage-v4'
                    or c['environment'].get('scoring_policy') != 'correctness-only-v1'):
                raise ConfigurationError('Token penalty requires correctness-only positive coverage')
        if c['training_reward']['version'] not in (reward_version, aligned_version, *tier_versions) or c['environment'].get('grading_version') not in ('all-claims-v6', 'all-claims-v7'):
            raise ConfigurationError('Shaped reward requires a versioned reliable grader')
        if c['training_reward']['version'] == aligned_version and c['environment'].get('grading_version') != 'all-claims-v7':
            raise ConfigurationError('Aligned reward requires the v7 independent factual coverage pass')
    if 'best_checkpoint' in c:
        exact(c['best_checkpoint'], ['retention_seconds'])
        positive(c['best_checkpoint']['retention_seconds'], 'best checkpoint retention', True)
        if c['best_checkpoint']['retention_seconds'] > 28*86400:
            raise ConfigurationError('Best checkpoint retention exceeds the bounded 28-day allowance')
    if 'execution' in c:
        from .remote import validate_execution
        validate_execution(c['execution'])
    if 'benchmark' in c:
        exact(c['benchmark'], ['task_manifest', 'manifest_hash', 'attempts'])
        positive(c['benchmark']['attempts'], 'benchmark.attempts', True)
        if c['environment']['kind'] != 'collection':
            raise ConfigurationError('Throughput benchmark requires collection environment')
    if 'concurrency' in c:
        exact(c['concurrency'], ['rollouts', 'judges'])
        for key, value in c['concurrency'].items():
            positive(value, 'concurrency.'+key, True)
            if value > 32:
                raise ConfigurationError('Concurrency is bounded to 32 per process')
        if c['environment']['kind'] == 'repository' and c['concurrency']['rollouts'] > 1:
            raise ConfigurationError('Concurrent strict-repository grading is not supported; use collection')
    if 'stopping' in c:
        exact(c['stopping'], [], ['initial_zero_batches', 'regression_delta', 'regression_checks'])
        for key, value in c['stopping'].items():
            positive(value, 'stopping.'+key, key != 'regression_delta')
    if not isinstance(c['output'], str) or not c['output']:
        raise ConfigurationError('Output directory is required')
    if c['schema_version'] != '1.0' or not isinstance(c['run_id'], str) or not c['run_id']:
        raise ConfigurationError('Invalid schema or run ID')
    if type(c['seed']) is not int:
        raise ConfigurationError('Seed must be an integer')
    exact(c['model'], ['base_model', 'rank', 'checkpoint_ttl_seconds'])
    for key in ('rank', 'checkpoint_ttl_seconds'):
        positive(c['model'][key], key, True)
    if not isinstance(c['model']['base_model'], str) or not c['model']['base_model'].strip():
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
        exact(s, ['kind', 'max_updates', 'max_batches', 'batch_size', 'learning_rate'], ['group_size', 'temperature', 'optimizer', 'baseline'])
        if s['kind'] not in {'sft', 'grpo', 'reinforce'}:
            raise ConfigurationError('Unsupported training strategy')
        for k in ('max_updates', 'max_batches', 'batch_size'):
            positive(s[k], k, True)
        positive(s['learning_rate'], 'learning_rate')
        if 'optimizer' in s:
            exact(s['optimizer'], [], ['beta1', 'beta2', 'eps', 'weight_decay', 'grad_clip_norm'])
            for key, value in s['optimizer'].items():
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                    raise ConfigurationError('Optimizer '+key+' must be finite')
                if key in {'beta1', 'beta2'} and not 0 <= value < 1:
                    raise ConfigurationError('Optimizer betas must be in [0, 1)')
                if key == 'eps' and value <= 0:
                    raise ConfigurationError('Optimizer eps must be positive')
                if key in {'weight_decay', 'grad_clip_norm'} and value < 0:
                    raise ConfigurationError('Optimizer '+key+' must be nonnegative')
        if s['kind'] in {'grpo', 'reinforce'}:
            if type(s.get('group_size')) is not int or s['group_size'] < 2 or s.get('temperature') != 1:
                raise ConfigurationError('Grouped RL requires group_size >= 2 and temperature=1')
        if s['kind'] == 'reinforce' and s.get('baseline') != 'prior-mean-v1':
            raise ConfigurationError('REINFORCE requires baseline=prior-mean-v1')
        if s['kind'] != 'reinforce' and 'baseline' in s:
            raise ConfigurationError('Baseline is only supported for REINFORCE')
    if type(c.get('group_retries', 1)) is not int or c.get('group_retries', 1) < 0:
        raise ConfigurationError('group_retries must be a nonnegative integer')
    exact(c['environment'], ['kind'], ['release', 'manifest_sha256', 'calibration', 'calibration_sha256', 'modal_prices', 'protocol_version', 'grading_version', 'tool_read_policy', 'tool_action_policy', 'judge_evidence_policy', 'solver_tools', 'scoring_policy'])
    for key, allowed in [('scoring_policy', ('strict-v1', 'correctness-only-v1')),
                         ('solver_tools', ('structured-v1', 'bash-only-v1')),
                         ('tool_action_policy', ('strict-v1', 'action-alias-v1')),
                         ('judge_evidence_policy', ('legacy-v1', 'definition-context-v1'))]:
        if c['environment'].get(key, allowed[0]) not in allowed:
            raise ConfigurationError('Unsupported ' + key)
    if c['environment'].get('tool_read_policy', 'strict-v1') not in ('strict-v1', 'paginate-v1'):
        raise ConfigurationError('Unknown tool read policy')
    if c['environment'].get('scoring_policy') == 'correctness-only-v1' and (c['environment'].get('grading_version') != 'all-claims-v7' or c.get('training_reward', {}).get('version') != 'positive-coverage-v4'):
        raise ConfigurationError('Correctness-only requires v7 positive coverage')
    if 'grading_version' in c['environment']:
        from .claim_grading import VERSION as claim_version
        from .strict_grading import VERSIONS
        if c['environment']['kind'] != 'collection' or c['environment']['grading_version'] not in (claim_version, *VERSIONS):
            raise ConfigurationError('Unsupported collection grading version; use a new versioned run')
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
        if any(s['kind'] == 'sft' for s in c['stages']) and 'supervised' not in c:
            raise ConfigurationError('Collection SFT requires a verified supervised manifest')
    if 'judge' in c:
        if c['environment']['kind'] != 'collection':
            raise ConfigurationError('Independent judge currently supports collection environments only')
        j = c['judge']
        exact(j, ['base_model', 'renderer', 'context_tokens', 'max_tokens', 'provider_timeout_seconds', 'prices'], ['temperature', 'repair_attempts'])
        for key in ('base_model', 'renderer'):
            if not isinstance(j[key], str) or not j[key]:
                raise ConfigurationError('Judge '+key+' is required')
        for key in ('context_tokens', 'max_tokens', 'provider_timeout_seconds'):
            positive(j[key], 'judge '+key, True)
        if j['max_tokens'] >= j['context_tokens']:
            raise ConfigurationError('Judge generation budget must leave room for context')
        if type(j.get('repair_attempts', 0)) is not int or not 0 <= j.get('repair_attempts', 0) <= 1:
            raise ConfigurationError('At most one mechanical judge repair is allowed')
        if j.get('repair_attempts', 0) and c['environment'].get('grading_version') not in ('all-claims-v4', 'all-claims-v5', 'all-claims-v6', 'all-claims-v7'):
            raise ConfigurationError('Judge repairs require a versioned reliable grader')
        temperature(j.get('temperature', 0), 'judge.temperature')
        if j['prices'] is not None:
            exact(j['prices'], ['model', 'prefill', 'sample', 'source', 'checked_at'])
            if j['prices']['model'] != j['base_model']:
                raise ConfigurationError('Judge prices must match judge model')
            for key in ('prefill', 'sample'):
                positive(j['prices'][key], 'judge '+key)
            if not j['prices']['source'] or not j['prices']['checked_at']:
                raise ConfigurationError('Missing judge price provenance')
    exact(c['evaluation'], ['every', 'max_tasks', 'temperature'], ['cohort_manifest', 'cohort_sha256'])
    if ('cohort_manifest' in c['evaluation']) != ('cohort_sha256' in c['evaluation']):
        raise ConfigurationError('Cohort manifest and hash must be supplied together')
    if type(c['evaluation']['every']) is not int or c['evaluation']['every'] < 0:
        raise ConfigurationError('Invalid evaluation cadence')
    positive(c['evaluation']['max_tasks'], 'max_tasks', True)
    temperature(c['evaluation']['temperature'], 'evaluation.temperature')
    exact(c['tracking'], ['mode', 'project'], ['entity', 'experiment_id', 'run_name', 'tags', 'notes', 'flush_every', 'upload_policy'])
    if c['tracking'].get('upload_policy', 'metrics-only') not in {'metrics-only', 'full'}:
        raise ConfigurationError('tracking.upload_policy must be metrics-only or full')
    if 'flush_every' in c['tracking'] and (type(c['tracking']['flush_every']) is not int or c['tracking']['flush_every'] < 0):
        raise ConfigurationError('tracking.flush_every must be a nonnegative integer; zero means final-only')
    if c['tracking']['mode'] not in {'disabled', 'offline', 'online'}:
        raise ConfigurationError('Invalid tracking mode')
    for key in ('project', 'entity', 'experiment_id', 'run_name', 'notes'):
        if key in c['tracking'] and (not isinstance(c['tracking'][key], str) or not c['tracking'][key].strip()):
            raise ConfigurationError('tracking.'+key+' must be a nonempty string')
    if 'tags' in c['tracking'] and (not isinstance(c['tracking']['tags'], list) or
            any(not isinstance(t, str) or not t.strip() for t in c['tracking']['tags'])):
        raise ConfigurationError('tracking.tags must be a list of nonempty strings')
    exact(c['spend'], ['cap_usd', 'ledger', 'prices'])
    if not isinstance(c['spend']['ledger'], str) or not c['spend']['ledger'].strip():
        raise ConfigurationError('Spend ledger path is required')
    positive(c['spend']['cap_usd'], 'cap_usd')
    p = c['spend']['prices']
    if p is not None:
        if 'model' in p and p['model'] != c['model']['base_model']:
            raise ConfigurationError('Solver prices must match base model; use prices: null to refresh')
        for k in ('prefill', 'sample', 'train', 'storage_gb_month', 'params'):
            positive(p[k], k)
    return c


def temperature(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ConfigurationError(name+' must be nonnegative and finite')


def resolve_run_config(config):
    """Resolve opt-in fresh run identities; checkpoints retain the concrete values."""
    import copy
    import re
    import uuid
    from datetime import datetime, timezone
    c = copy.deepcopy(validate_config(config))
    if c['environment']['kind'] == 'collection':
        from .claim_grading import VERSION as claim_version
        c['environment'].setdefault('grading_version', claim_version)
    if c['run_id'] == 'auto':
        prefix = c['tracking'].get('experiment_id', c['model']['base_model'].rsplit('/', 1)[-1])
        prefix = re.sub(r'[^A-Za-z0-9_-]+', '-', prefix).strip('-')[:60] or 'training'
        c['run_id'] = f"{prefix}-s{c['seed']}-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}"
    for obj, key in ((c, 'output'), (c['spend'], 'ledger')):
        if '{run_id}' in obj[key]:
            if '/' in c['run_id'] or '\\' in c['run_id'] or c['run_id'] in {'.', '..'}:
                raise ConfigurationError('Path template requires a path-safe run_id')
            obj[key] = obj[key].replace('{run_id}', c['run_id'])
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
    if 'task_subset' in config:
        from .task_subset import apply_subset
        data = apply_subset(config, data)
    if 'supervised' in config:
        from .sft_collection import load_supervised
        data['sft'] = load_supervised(config['supervised'], data)
        from .investigation_sft import VERSION as investigation_version
        supervised_manifest = read(config['supervised']['manifest'])
        if supervised_manifest['version'] == investigation_version:
            from .collection import CollectionFactory
            expected = CollectionFactory(config, Path(config['output']), None).reward_version
            if supervised_manifest['grading_version'] != expected:
                raise ConfigurationError('Investigation SFT judge/reward identity changed')
    for s in config['stages']:
        if not data['sft' if s['kind'] == 'sft' else 'tasks']:
            raise ConfigurationError('Training input is empty for ' + s['kind'])
    if not data['development']:
        raise ConfigurationError('Development evaluation input is empty')
    if 'harness' in config:
        from .harness_variants import validate_sources
        validate_sources(config['harness'], data)
    if 'cohort_manifest' in config['evaluation']:
        from .cohorts import select_tasks
        selected, _ = select_tasks(data, config['evaluation'])
        if len(selected) != config['evaluation']['max_tasks']:
            raise ConfigurationError('max_tasks must match the complete selection cohort')
    for example in data['sft']:
        messages = example.get('messages')
        if not isinstance(messages, list) or not messages:
            raise ConfigurationError('SFT requires visible messages')
        for message in messages:
            if not isinstance(message, dict) or set(message) != {'role', 'content'} or message['role'] not in {'system', 'user', 'assistant', 'tool'} or not isinstance(message['content'], str):
                raise ConfigurationError('Invalid SFT role/content message')
    return data
