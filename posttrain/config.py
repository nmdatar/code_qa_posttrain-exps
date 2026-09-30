"""Frozen experiment configuration. No credentials belong in a RunSpec."""
from pathlib import Path
import copy
import math
import re
from .storage import digest, read

DEFAULTS = dict(schema_version='1', backend='fake', model='fake', diagnostic=True,
    seed=42, stages=[{'algorithm':'grpo','max_updates':3,'learning_rate':1e-5}],
    group_size=4, batch_groups=1, max_concurrency=4, judge_concurrency=2,
    checkpoint_every=1, eval_every=1, max_attempted_batches=30,
    max_tool_calls=8, max_turns=12, max_output_tokens=2048, max_context_tokens=16384,
    max_episode_seconds=300, max_observation_chars=12000, temperature=0.8,
    budget_cap_usd=20.0, budget_reserve_usd=2.0,
    budget_ledger='artifacts/posttrain/spending.json', tracking_mode='disabled',
    tracking_project='repo-qa-posttrain', artifacts_root='artifacts/posttrain',
    training_release=None, evaluation_release=None, calibration_report=None, review_policy='automated', rubric_reviews={},
    renderer_name=None, lora_rank=16, policy_version='initial', reward_version='qa-eval-v1',
    full_group_retries=1, tool_semantics='stateless', train_tasks=[], eval_tasks=[],
    sft_examples=[], environment_bundles={}, repository_roots={}, strict_tasks={},
    verifier_config=None, judge_catalog_mode='reference_and_observed', paid_call_bounds={}, pricing_snapshot=None)


def load_config(path_or_dict):
    supplied = read(path_or_dict) if isinstance(path_or_dict, (str, Path)) else copy.deepcopy(path_or_dict)
    unknown = set(supplied) - set(DEFAULTS) - {'run_id'}
    if unknown: raise ValueError('Unknown run settings: ' + ', '.join(sorted(unknown)))
    c = copy.deepcopy(DEFAULTS); c.update(supplied)
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,90}', c.get('run_id','')):
        raise ValueError('run_id must be a short safe identifier')
    if c['backend'] not in {'fake','tinker'}: raise ValueError('Unsupported backend')
    if c['review_policy'] not in {'automated','human'}: raise ValueError('Unsupported review policy')
    if c['tool_semantics'] != 'stateless': raise ValueError('Persistent episodes not implemented')
    if c['tracking_mode'] not in {'disabled','offline','online'}: raise ValueError('Invalid tracking mode')
    for key in ('group_size','batch_groups','max_concurrency','judge_concurrency','checkpoint_every','eval_every',
                'max_attempted_batches','max_tool_calls','max_turns','max_output_tokens','max_context_tokens',
                'max_episode_seconds','max_observation_chars','lora_rank'):
        if type(c[key]) is not int or c[key] < 1: raise ValueError('Positive integer required: ' + key)
    if c['group_size'] < 2 or c['full_group_retries'] not in (0,1): raise ValueError('Invalid group settings')
    if not 0 <= c['budget_reserve_usd'] < c['budget_cap_usd'] <= 20: raise ValueError('Budget cap cannot exceed $20')
    if c['judge_catalog_mode'] not in {'full','reference_and_observed'}: raise ValueError('Unsupported judge catalog scope')
    if not c['stages']: raise ValueError('At least one stage is required')
    for stage in c['stages']:
        if set(stage) != {'algorithm','max_updates','learning_rate'}: raise ValueError('Stage needs algorithm, max_updates, learning_rate')
        if stage['algorithm'] not in ('grpo','sft'): raise ValueError('Only SFT and GRPO supported')
        if type(stage['max_updates']) is not int or stage['max_updates'] < 1: raise ValueError('Invalid update count')
        if not math.isfinite(stage['learning_rate']) or stage['learning_rate'] <= 0: raise ValueError('Invalid learning rate')
    if c['diagnostic'] and (sum(s['max_updates'] for s in c['stages']) > 3 or c['max_concurrency'] > 4):
        raise ValueError('Diagnostic mode permits at most 3 updates and 4 concurrent episodes')
    def secret_keys(value):
        if isinstance(value, dict):
            for key,item in value.items():
                if re.search(r'(api_key|password|secret|access_token)$',key,re.I): raise ValueError('Credentials must not be serialized')
                secret_keys(item)
        elif isinstance(value,list):
            for item in value: secret_keys(item)
    secret_keys(c)
    return c


def semantics(config):
    return {k:v for k,v in config.items() if k not in {'run_id','artifacts_root','tracking_mode','tracking_project','budget_ledger'}}


def fingerprint(config): return digest(semantics(config))
