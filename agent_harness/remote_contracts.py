"""JSON-only remote boundaries. Private grading data never belongs in a job."""
from __future__ import annotations
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import tempfile

from .contracts import RunLimits

JOB_FIELDS = {'schema_version', 'run_id', 'episode_id', 'group_id', 'task_id', 'task_hash',
              'question', 'repository', 'permitted_tools', 'limits', 'max_submission_bytes',
              'model', 'environment_manifest'}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()


def job_hash(job):
    return hashlib.sha256(canonical(job)).hexdigest()


def identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,127}', value):
        raise ValueError('Invalid remote identifier')
    return value


def safe_join(root, relative):
    if not isinstance(relative, str) or not relative or '\\' in relative or '\0' in relative:
        raise ValueError('Invalid artifact path')
    parts = relative.split('/')
    if PurePosixPath(relative).is_absolute() or any(p in ('', '.', '..') for p in parts):
        raise ValueError('Artifact path must be normalized and relative')
    root = Path(root).resolve()
    path = root.joinpath(*parts)
    current = root
    for part in parts:
        current = current / part
        if current.is_symlink():
            raise ValueError('Symlink artifact path is forbidden')
    if not path.resolve().is_relative_to(root):
        raise ValueError('Artifact path escapes root')
    return path


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = canonical(value) + b'\n'
    fd, name = tempfile.mkstemp(prefix='.pending-', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def validate_job(job):
    if not isinstance(job, dict) or set(job) != JOB_FIELDS or type(job['schema_version']) is not int or job['schema_version'] != 1:
        raise ValueError('Invalid job fields; private task data is not permitted')
    for field in ('run_id', 'episode_id', 'group_id', 'task_id'):
        identifier(job[field])
    if not isinstance(job['task_hash'], str) or not re.fullmatch(r'[0-9a-f]{64}', job['task_hash']):
        raise ValueError('Invalid task hash')
    if not isinstance(job['question'], str) or not job['question'].strip():
        raise ValueError('Question is required')
    repository = job['repository']
    if not isinstance(repository, dict) or set(repository) != {'bundle', 'bundle_sha256', 'commit'}:
        raise ValueError('Invalid repository reference')
    safe_join('/tmp/harness-path-validation', repository['bundle'])
    if not isinstance(repository['commit'], str) or not re.fullmatch(r'[0-9a-f]{40}', repository['commit']):
        raise ValueError('Full commit required')
    if not isinstance(repository['bundle_sha256'], str) or not re.fullmatch(r'[0-9a-f]{64}', repository['bundle_sha256']):
        raise ValueError('Bundle SHA256 required')
    tools = job['permitted_tools']
    if not isinstance(tools, list) or any(not isinstance(t, str) or not t for t in tools) or len(set(tools)) != len(tools):
        raise ValueError('Invalid permitted tools')
    if not isinstance(job['limits'], dict) or set(job['limits']) != set(asdict(RunLimits())):
        raise ValueError('Explicit limits required')
    limits = RunLimits(**job['limits'])
    for field in ('max_steps', 'max_tool_calls', 'max_output_tokens', 'max_context_chars'):
        value = getattr(limits, field)
        if type(value) is not int or value < (0 if field == 'max_tool_calls' else 1):
            raise ValueError('Invalid limit: ' + field)
    if type(limits.wall_time_seconds) not in (float, int) or not 0 < limits.wall_time_seconds <= 3600:
        raise ValueError('Episode wall time must be in (0, 3600] seconds')
    if type(job['max_submission_bytes']) is not int or not 0 < job['max_submission_bytes'] <= 4_000_000:
        raise ValueError('Invalid submission limit')
    validate_model(job['model'])
    if job['environment_manifest'] is not None:
        validate_environment(job['environment_manifest'], repository['commit'])
    canonical(job)  # rejects NaN and non-JSON resources
    return job


def identity(job):
    return {key: job[key] for key in ('run_id', 'episode_id', 'task_id', 'group_id')}


_ENV_FIELDS = {'backend', 'commit', 'capability', 'image_digest', 'snapshot_files',
               'snapshot_sha256', 'readiness', 'status', 'runtime_policy'}
_POLICY_FIELDS = {'network', 'host_mounts', 'read_only', 'source_read_only', 'secrets',
                  'fresh_sandbox_per_operation'}


def public_environment(manifest):
    """Project build manifests without build commands, logs, or arbitrary fields."""
    if not isinstance(manifest, dict):
        raise ValueError('Environment manifest must be an object')
    result = {key: manifest[key] for key in _ENV_FIELDS if key in manifest}
    for key, allowed in (('readiness', {'exit_code', 'timed_out', 'truncated'}),
                         ('runtime_policy', _POLICY_FIELDS)):
        value = manifest.get(key)
        if not isinstance(value, dict):
            raise ValueError('Invalid environment ' + key)
        result[key] = {field: value[field] for field in allowed if field in value}
    return result


def validate_environment(env, commit):
    if not isinstance(env, dict) or set(env) != _ENV_FIELDS:
        raise ValueError('Environment must contain only public execution fields')
    if env['backend'] != 'modal' or env['status'] != 'ready' or env['capability'] != 'execution' or env['commit'] != commit:
        raise ValueError('Environment must be ready Modal execution for the pinned commit')
    if not isinstance(env['image_digest'], str) or not re.fullmatch(r'im-[A-Za-z0-9_-]+', env['image_digest']):
        raise ValueError('Immutable Modal image identifier required')
    files = env['snapshot_files']
    if not isinstance(files, dict):
        raise ValueError('Invalid snapshot file map')
    for name, value in files.items():
        safe_join('/tmp/harness-path-validation', name)
        if not isinstance(value, str) or not re.fullmatch(r'[0-9a-f]{64}', value):
            raise ValueError('Invalid snapshot file digest')
    if env['snapshot_sha256'] != hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest():
        raise ValueError('Snapshot manifest digest mismatch')
    readiness = env['readiness']
    if (not isinstance(readiness, dict) or set(readiness) != {'exit_code', 'timed_out', 'truncated'}
            or type(readiness['exit_code']) is not int or readiness['exit_code'] != 0
            or readiness['timed_out'] is not False or readiness['truncated'] is not False):
        raise ValueError('Environment readiness must be successful and complete')
    policy = env['runtime_policy']
    if (not isinstance(policy, dict) or set(policy) - _POLICY_FIELDS
            or policy.get('network') != 'blocked' or policy.get('host_mounts') is not False
            or policy.get('secrets') is not False or policy.get('source_read_only') is not True
            or policy.get('fresh_sandbox_per_operation') is not True
            or any(type(value) is not bool for key, value in policy.items() if key != 'network')):
        raise ValueError('Environment must declare complete isolated runtime policy')


def validate_model(model):
    if not isinstance(model, dict) or model.get('kind') not in ('http', 'tinker', 'scripted'):
        raise ValueError('Unknown model adapter')
    allowed = {
        'http': {'kind', 'model', 'base_url', 'timeout_seconds', 'max_response_bytes', 'input_price_per_million', 'output_price_per_million'},
        'tinker': {'kind', 'model', 'base_model', 'model_path', 'renderer_name', 'temperature', 'seed', 'input_price_per_million', 'output_price_per_million'},
        'scripted': {'kind', 'actions'},
    }[model['kind']]
    if set(model) - allowed:
        raise ValueError('Model config has unknown or credential-bearing fields')
    try:
        canonical(model)
        for key in ('input_price_per_million', 'output_price_per_million'):
            value = model.get(key)
            if value is not None and (type(value) not in (int, float) or not math.isfinite(value) or value < 0):
                raise ValueError('Invalid model token price')
        if model['kind'] == 'http':
            if not isinstance(model.get('model'), str) or not model['model'].strip() or not isinstance(model.get('base_url'), str):
                raise ValueError('HTTP model and base_url are required')
            timeout = model.get('timeout_seconds', 120)
            if type(timeout) not in (int, float) or not math.isfinite(timeout) or not 0 < timeout <= 3600:
                raise ValueError('Invalid model timeout')
            cap = model.get('max_response_bytes', 4 * 1024 * 1024)
            if type(cap) is not int or not 1 <= cap <= 16 * 1024 * 1024:
                raise ValueError('Invalid response byte limit')
            from .models import ChatCompletionsModel
            ChatCompletionsModel(**{k: v for k, v in model.items() if k != 'kind'})
        elif model['kind'] == 'tinker':
            from .tinker_model import TinkerModel
            TinkerModel(**{k: v for k, v in model.items() if k != 'kind'})
        else:
            actions = model.get('actions')
            if not isinstance(actions, list) or not actions:
                raise ValueError('Scripted model requires actions')
            for action in actions:
                if not isinstance(action, dict):
                    raise ValueError('Scripted action must be an object')
                if action.get('type') == 'tool_call':
                    if (not {'type', 'name', 'arguments'} <= set(action)
                            or set(action) - {'type', 'name', 'arguments', 'call_id'}
                            or not isinstance(action['name'], str) or not action['name']
                            or not isinstance(action['arguments'], dict)
                            or not isinstance(action.get('call_id', ''), str)):
                        raise ValueError('Invalid scripted tool call')
                elif action.get('type') == 'final_answer':
                    if set(action) != {'type', 'value'}:
                        raise ValueError('Invalid scripted final answer')
                else:
                    raise ValueError('Unknown scripted action')
    except (TypeError, OverflowError) as exc:
        raise ValueError('Invalid model configuration types') from None
    return model
