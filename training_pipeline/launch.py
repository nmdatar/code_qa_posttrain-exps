"""Worst-case operation estimate for bounded experimental collection runs."""
from .collection import modal_reservation
from .contracts import ConfigurationError


def estimate(config, prices):
    limits = config['limits']
    stages = config['stages']
    if config['environment']['kind'] != 'collection':
        raise ConfigurationError('Launch estimate requires collection configuration')
    rollouts = sum(s['max_batches']*s['batch_size']*s['group_size']*2 for s in stages)
    updates = sum(min(s['max_updates'],s['max_batches']) for s in stages)
    # Baseline, stage ends, and every possible scheduled evaluation; double-counting is conservative.
    evaluations = (1+len(stages)+updates)*config['evaluation']['max_tasks']
    episodes = rollouts+evaluations
    sample_calls = episodes*limits['max_generations']
    sample = sample_calls*(limits['context_tokens']*prices['prefill']+limits['max_tokens_per_call']*prices['sample'])/1e6
    judge = episodes*(limits['context_tokens']*prices['prefill']+512*prices['sample'])/1e6
    training_tokens = sum(s['max_batches']*s['batch_size']*s['group_size'] for s in stages)*limits['max_generations']*limits['context_tokens']
    train = training_tokens*prices['train']/1e6
    checkpoints = 1+updates+len(stages)
    storage = checkpoints*prices['params']*32/1e9*prices['storage_gb_month']*config['model']['checkpoint_ttl_seconds']/(28*86400)
    modal = episodes*modal_reservation(config)
    components = {'policy_sampling':sample,'reference_grading':judge,'training':train,'checkpoint_storage':storage,'modal_sandboxes':modal}
    total = sum(components.values())
    return {'status':'within_ceiling' if total <= config['spend']['cap_usd'] else 'over_ceiling',
        'upper_estimate_usd':total,'cap_usd':config['spend']['cap_usd'],'components_usd':components,
        'episodes_including_retries':episodes,'checkpoint_pairs':checkpoints,'prices':prices,
        'modal_prices':config['environment']['modal_prices'],'actual_billing_usd':None,
        'note':'Conservative reservations, not a provider-enforced billing cap. Existing images only; no image builds.'}
