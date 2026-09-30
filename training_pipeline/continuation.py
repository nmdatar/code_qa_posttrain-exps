"""Resume immutable optimizer state into a new audit directory and fresh trainer.

Runtime inputs must be mounted at the original pinned paths. No checkpoint
manifests are rewritten, and no scientific or spending setting is overridden.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path

from .config import inputs
from .contracts import ConfigurationError
from .storage import atomic_json, load_checkpoint, semantic_hash


def prepare_continuation(checkpoint, output, run_id):
    manifest = load_checkpoint(checkpoint)
    config = copy.deepcopy(manifest['config'])
    if not run_id or run_id == config['run_id'] or Path(output).resolve() == Path(config['output']).resolve():
        raise ConfigurationError('Continuation requires a new run ID and output directory')
    if Path(output).exists():
        raise ConfigurationError('Continuation output already exists')
    if manifest['state'].get('stop_reason'):
        raise ConfigurationError('Safety-stopped checkpoints require a separately reviewed plan')
    config['run_id'] = run_id
    config['output'] = str(output)
    config['tracking'].pop('run_name', None)
    config['tracking']['notes'] = ('Continuation from immutable acknowledged checkpoint ' + manifest['id'] +
                                   '; fresh trainer restores optimizer, baseline, cursor and RNG; prior failed run retained.')
    if semantic_hash(config) != manifest['config_hash']:
        raise ConfigurationError('Continuation changed scientific or spending configuration')
    data = inputs(config)
    if data['identity'] != manifest['state']['data_identity']:
        raise ConfigurationError('Continuation data identity mismatch')
    state = manifest['state']
    receipt = {'checkpoint': str(Path(checkpoint).resolve()), 'checkpoint_id': manifest['id'],
               'checkpoint_manifest_hash': manifest['manifest_hash'],
               'source_run_id': manifest['config']['run_id'], 'run_id': run_id,
               'output': str(output), 'purpose': 'resume', 'fresh_trainer_required': True,
               'scientific_config_hash': manifest['config_hash'],
               'data_identity': data['identity'], 'optimizer_step': state['optimizer_step'],
               'attempted_batches': state.get('attempted_batches'), 'cursor': state['cursor'],
               'reinforce_baselines': copy.deepcopy(state.get('reinforce_baselines', {})),
               'spend': copy.deepcopy(config['spend']),
               'artifact_paths': copy.deepcopy(manifest['artifacts'])}
    return config, data, receipt


def remaining_estimate(config, state, prices):
    """Budget only unfinished work; never change the config used for training.

    The standard estimator includes an initial evaluation that resume skips.
    That spare check also bounds one evaluation-cadence crossing due to the
    restored global optimizer-step offset.
    """
    from .launch import estimate
    stage_index = state['stage']
    if not 0 <= stage_index < len(config['stages']):
        raise ConfigurationError('Checkpoint has no unfinished training stage')
    bounded = copy.deepcopy(config)
    bounded['stages'] = bounded['stages'][stage_index:]
    active = bounded['stages'][0]
    active['max_updates'] -= state['stage_updates']
    active['max_batches'] -= state['stage_batches']
    if active['max_updates'] <= 0 or active['max_batches'] <= 0:
        raise ConfigurationError('Checkpoint stage completion is inconsistent')
    result = estimate(bounded, prices)
    result['resume_optimizer_step'] = state['optimizer_step']
    result['remaining_stage_max_batches'] = [stage['max_batches'] for stage in bounded['stages']]
    result['remaining_stage_max_updates'] = [stage['max_updates'] for stage in bounded['stages']]
    result['scope'] = 'unfinished_work_only_original_execution_config_unchanged'
    return result


def execute_continuation(checkpoint, output, run_id, backend=None):
    # Pipeline.run(resume) creates a new training client and calls
    # load_state_with_optimizer before restoring the full recorded local state.
    from .orchestrator import Pipeline
    config, data, receipt = prepare_continuation(checkpoint, output, run_id)
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    source_hashes = {}
    for package in ('training_pipeline', 'agent_harness', 'qa_eval'):
        for source in (Path(__file__).resolve().parent.parent / package).glob('*.py'):
            relative = source.relative_to(Path(__file__).resolve().parent.parent)
            target = root / 'source' / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            contents = source.read_bytes()
            target.write_bytes(contents)
            source_hashes[str(relative)] = hashlib.sha256(contents).hexdigest()
    atomic_json(root / 'source' / 'manifest.json', source_hashes)
    atomic_json(root / 'config.json', config)
    atomic_json(root / 'continuation.json', receipt)
    return Pipeline(config, data=data, backend=backend).run(checkpoint=checkpoint, purpose='resume')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--execute', action='store_true', help='Launch only after diagnostic/fix verification')
    args = parser.parse_args()
    if args.execute:
        print(json.dumps({'checkpoint': str(execute_continuation(args.checkpoint, args.output, args.run_id))}))
    else:
        _, _, receipt = prepare_continuation(args.checkpoint, args.output, args.run_id)
        print(json.dumps({'status': 'preflight_only_no_remote_calls', **receipt}, indent=2))


if __name__ == '__main__':
    main()
