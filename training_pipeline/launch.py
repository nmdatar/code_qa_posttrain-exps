"""Worst-case operation estimate for bounded experimental collection runs."""
from .collection import modal_reservation
from .contracts import ConfigurationError


def estimate(config, prices, evaluation_only=False):
    limits = config['limits']
    stages = config['stages']
    if config['environment']['kind'] != 'collection':
        raise ConfigurationError('Launch estimate requires collection configuration')
    rollouts = 0 if evaluation_only else sum(s['max_batches']*s['batch_size']*s['group_size']*(1+config.get('group_retries', 1)) for s in stages if s['kind'] in {'grpo', 'reinforce'})
    updates = sum(min(s['max_updates'],s['max_batches']) for s in stages)
    # Training evaluates selection only. Bound scheduled checks by successful
    # updates, plus each stage end (possibly overlapping), not every update.
    max_tasks = config['evaluation']['max_tasks']
    if evaluation_only and 'cohort_manifest' in config['evaluation']:
        # The public estimate also serves explicit confirmation invocations.
        from .storage import read
        cohorts = read(config['evaluation']['cohort_manifest'])
        execution = config.get('execution', {})
        if execution.get('operation') == 'baseline' and execution.get('cohort') in ('selection', 'confirmation'):
            max_tasks = len(cohorts[execution['cohort']])
        else:
            max_tasks = max(len(cohorts['selection']), len(cohorts['confirmation']))
    every = config['evaluation']['every']
    checks = 1 + len(stages) + (updates // every if every else 0)
    if 'initial_zero_batches' in config.get('stopping', {}):
        checks += 1  # Separate terminal no-signal evaluation.
    evaluations = max_tasks if evaluation_only else checks*max_tasks
    episodes = rollouts+evaluations
    sample_calls = episodes*limits['max_generations']
    sample = sample_calls*(limits['context_tokens']*prices['prefill']+limits['max_tokens_per_call']*prices['sample'])/1e6
    judge = episodes*(limits['context_tokens']*prices['prefill']+512*prices['sample'])/1e6
    if 'judge' in config:
        j = config['judge']
        judge = episodes*(j['context_tokens']*j['prices']['prefill']+j['max_tokens']*j['prices']['sample'])/1e6
        if config['environment'].get('grading_version') in ('all-claims-v3', 'all-claims-v4', 'all-claims-v5', 'all-claims-v6', 'all-claims-v7'):
            judge *= (3 if config['environment'].get('grading_version') == 'all-claims-v7' else 2) * (1+j.get('repair_attempts', 0))
    if 'judge' in config and config['environment'].get('grading_version') == 'all-claims-v7':
        # TinkerBackend.sample rejects prompt + max_tokens > context before any
        # reservation. Every judge stage, including repairs, uses max_tokens.
        prompt_bound = j['context_tokens'] - j['max_tokens']
        per_call = (prompt_bound*j['prices']['prefill'] + j['max_tokens']*j['prices']['sample'])/1e6
        # Independent coverage is training-only; selection/confirmation uses
        # extract + assess. Benchmark inputs are training tasks, so retain 3.
        evaluation_stages = 3 if (config.get('execution', {}).get('operation') == 'benchmark' or config['environment'].get('scoring_policy') == 'correctness-only-v1') else 2
        judge = (3*rollouts + evaluation_stages*evaluations)*(1+j.get('repair_attempts', 0))*per_call
    training_tokens = sum(s['max_batches']*s['batch_size']*(s['group_size'] if s['kind'] in {'grpo', 'reinforce'} else 1) for s in stages)*limits['max_generations']*limits['context_tokens']
    train = 0 if evaluation_only else training_tokens*prices['train']/1e6
    checkpoints = 0 if evaluation_only else 1+updates+len(stages)
    if not evaluation_only and 'stopping' in config:
        # Final no-signal boundary plus persisted baseline/regression decisions.
        checkpoints += 1
        if 'regression_delta' in config['stopping']:
            checkpoints += 1+updates+len(stages)
    storage = checkpoints*prices['params']*32/1e9*prices['storage_gb_month']*config['model']['checkpoint_ttl_seconds']/(28*86400)
    if not evaluation_only and 'best_checkpoint' in config:
        storage += checks*prices['params']*32/1e9*prices['storage_gb_month']*config['best_checkpoint']['retention_seconds']/(28*86400)
    modal = episodes*modal_reservation(config)
    components = {'policy_sampling':sample,'reference_grading':judge,'training':train,'checkpoint_storage':storage,'modal_sandboxes':modal}
    if 'prompt_decomposition' in config:
        from .prompt_decomposition import teacher_config
        from .storage import read
        subset = read(config['task_subset']['manifest'])
        count = len(set(subset['training_ids'] + subset['evaluation_ids']))
        teacher = teacher_config(config)
        components['question_decomposition'] = count * (
            (teacher['context_tokens']-teacher['max_tokens'])*teacher['prices']['prefill']
            + teacher['max_tokens']*teacher['prices']['sample'])/1e6
    total = sum(components.values())
    return {'status':'within_ceiling' if total <= config['spend']['cap_usd'] else 'over_ceiling',
        'upper_estimate_usd':total,'cap_usd':config['spend']['cap_usd'],'components_usd':components,
        'evaluation_episodes_upper_bound':evaluations,
        'episodes_including_retries':episodes,'checkpoint_pairs':checkpoints,'prices':prices,
        'judge':config.get('judge', {'base_model':config['model']['base_model'],'legacy':True}),
        'modal_prices':config['environment']['modal_prices'],'actual_billing_usd':None,
        'note':'Conservative reservations, not a provider-enforced billing cap. Existing images only; no image builds.'}
