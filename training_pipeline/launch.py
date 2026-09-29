"""Worst-case operation estimate for bounded experimental collection runs."""
from .collection import modal_reservation
from .contracts import ConfigurationError


def estimate(config, prices, evaluation_only=False):
    limits = config['limits']
    stages = config['stages']
    if config['environment']['kind'] != 'collection':
        raise ConfigurationError('Launch estimate requires collection configuration')
    rollouts = 0 if evaluation_only else sum(s['max_batches']*s['batch_size']*s['group_size']*(1+config.get('group_retries', 1)) for s in stages)
    updates = sum(min(s['max_updates'],s['max_batches']) for s in stages)
    # Baseline, stage ends, and every possible scheduled evaluation; double-counting is conservative.
    max_tasks = config['evaluation']['max_tasks']
    if 'cohort_manifest' in config['evaluation']:
        from .storage import read
        cohorts = read(config['evaluation']['cohort_manifest'])
        max_tasks = max(len(cohorts['selection']), len(cohorts['confirmation']))
    evaluations = max_tasks if evaluation_only else (1+len(stages)+updates)*max_tasks
    episodes = rollouts+evaluations
    sample_calls = episodes*limits['max_generations']
    sample = sample_calls*(limits['context_tokens']*prices['prefill']+limits['max_tokens_per_call']*prices['sample'])/1e6
    judge = episodes*(limits['context_tokens']*prices['prefill']+512*prices['sample'])/1e6
    if 'judge' in config:
        j = config['judge']
        judge = episodes*(j['context_tokens']*j['prices']['prefill']+j['max_tokens']*j['prices']['sample'])/1e6
    training_tokens = sum(s['max_batches']*s['batch_size']*s['group_size'] for s in stages)*limits['max_generations']*limits['context_tokens']
    train = 0 if evaluation_only else training_tokens*prices['train']/1e6
    checkpoints = 0 if evaluation_only else 1+updates+len(stages)
    if not evaluation_only and 'stopping' in config:
        # Final no-signal boundary plus persisted baseline/regression decisions.
        checkpoints += 1
        if 'regression_delta' in config['stopping']:
            checkpoints += 1+updates+len(stages)
    storage = checkpoints*prices['params']*32/1e9*prices['storage_gb_month']*config['model']['checkpoint_ttl_seconds']/(28*86400)
    modal = episodes*modal_reservation(config)
    components = {'policy_sampling':sample,'reference_grading':judge,'training':train,'checkpoint_storage':storage,'modal_sandboxes':modal}
    total = sum(components.values())
    return {'status':'within_ceiling' if total <= config['spend']['cap_usd'] else 'over_ceiling',
        'upper_estimate_usd':total,'cap_usd':config['spend']['cap_usd'],'components_usd':components,
        'episodes_including_retries':episodes,'checkpoint_pairs':checkpoints,'prices':prices,
        'judge':config.get('judge', {'base_model':config['model']['base_model'],'legacy':True}),
        'modal_prices':config['environment']['modal_prices'],'actual_billing_usd':None,
        'note':'Conservative reservations, not a provider-enforced billing cap. Existing images only; no image builds.'}
