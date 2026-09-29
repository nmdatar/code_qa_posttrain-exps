"""Paid, bounded machinery validation using explicitly synthetic local tools."""
from datetime import datetime, timezone
import uuid
from pathlib import Path
from .budget import current_prices, SpendLedger, BudgetLimit
from .storage import atomic_json, read, load_checkpoint
from .orchestrator import Pipeline, make_backend, evaluate_checkpoint


def smoke_config(root, prices, ledger_path):
    return {'schema_version': '1.0', 'run_id': 'toy-' + uuid.uuid4().hex[:12], 'output': str(Path(root)/'run'),
        'model': {'base_model': 'Qwen/Qwen3.5-4B', 'rank': 8, 'checkpoint_ttl_seconds': 86400},
        'seed': 42,
        'limits': {'max_generations': 2, 'max_tokens_per_call': 64, 'max_output_tokens': 128,
                   'context_tokens': 512, 'max_tool_calls': 1, 'max_tool_output_bytes': 2048,
                   'latency_seconds': 120, 'provider_timeout_seconds': 120},
        'stages': [
            {'kind': 'sft', 'max_updates': 1, 'max_batches': 1, 'batch_size': 8, 'learning_rate': 1e-4},
            {'kind': 'grpo', 'max_updates': 2, 'max_batches': 4, 'batch_size': 2,
             'group_size': 4, 'learning_rate': 1e-5, 'temperature': 1}],
        'environment': {'kind': 'toy'}, 'evaluation': {'every': 1, 'max_tasks': 2, 'temperature': 0},
        'checkpoint_every': 1, 'tracking': {'mode': 'disabled', 'project': 'repository-qa-training'},
        'spend': {'cap_usd': 5.0, 'ledger': str(ledger_path), 'prices': prices}}


def run_smoke(base):
    base = Path(base).resolve()
    base.mkdir(parents=True, exist_ok=True)
    root = base / ('smoke-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid.uuid4().hex[:6])
    root.mkdir()
    report = {'kind': 'synthetic_training_machinery_smoke', 'repository_quality_claim': False,
              'status': 'preflight', 'started_at': datetime.now(timezone.utc).isoformat()}
    report_path = root / 'smoke-report.json'
    atomic_json(report_path, report)
    try:
        fresh_prices = current_prices('Qwen/Qwen3.5-4B')
        ledger_path = base / 'spend-ledger.json'
        prices = read(ledger_path)['prices'] if ledger_path.exists() else fresh_prices
        for key in ('prefill', 'sample', 'train', 'storage_gb_month', 'params'):
            if fresh_prices[key] > prices[key]:
                raise BudgetLimit('Price increase requires a new reviewed spending estimate')
        config = smoke_config(root, prices, ledger_path)
        ledger = SpendLedger(ledger_path, 5.0, prices, config['model']['checkpoint_ttl_seconds'])
        # Four RL batches including one whole-group infrastructure retry; eight
        # checkpoint pairs cover initial, stages, fork verification, and recovery.
        estimate = (160 * ledger.estimate('sample', input_tokens=512, output_tokens=64)
                    + 5 * ledger.estimate('train', input_tokens=16*512)
                    + 8 * ledger.estimate('checkpoint'))
        if ledger.state['reserved_usd'] + estimate > 5:
            raise BudgetLimit('Full validation reservation cannot fit remaining $5 budget')
        report.update(config=config, preflight_upper_estimate_usd=estimate,
                      prior_reservations_usd=ledger.state['reserved_usd'], pricing_verified=fresh_prices)
        atomic_json(root / 'config.json', config)
        atomic_json(report_path, report)
        print('Smoke artifacts:', root, flush=True)
        print('Conservative full-run estimate: $' + format(estimate, '.4f'), flush=True)
        first = Pipeline(config)
        checkpoint = first.run(stop_after_updates=2)
        report['interruption_checkpoint'] = str(checkpoint)
        manifest = load_checkpoint(checkpoint)
        # Restore a distinct client with optimizer, cursor and RNG from disk.
        resumed = Pipeline(config)
        checkpoint = resumed.run(checkpoint=checkpoint, purpose='resume')
        report['final_checkpoint'] = str(checkpoint)
        final = load_checkpoint(checkpoint)
        report['resume_verified'] = final['state']['optimizer_step'] >= manifest['state']['optimizer_step']
        report['standalone_evaluation'] = evaluate_checkpoint(checkpoint, output=root/'standalone-evaluation')
        fork = make_backend(config)
        fork.create_trainer(config['seed'])
        fork.load(final['artifacts'], 'fork')
        sample = fork.sample([{'role': 'user', 'content': 'Reply with a single word: ready'}], 16, 0)
        fork_artifacts = fork.save('fork-check-' + uuid.uuid4().hex)
        fork.verify_artifacts(fork_artifacts)
        report['fork'] = {'fresh_optimizer': True, 'parent': final['id'], 'artifacts': fork_artifacts,
                          'sample': sample.text}
        events = [__import__('json').loads(line) for line in (Path(config['output'])/'events.jsonl').read_text().splitlines()]
        updates = [e for e in events if e['event'] == 'update']
        report['updates'] = updates
        report['sft_updates'] = sum(e['loss'] == 'cross_entropy' for e in updates)
        report['rl_updates'] = sum(e['loss'] == 'importance_sampling' for e in updates)
        report['status'] = 'passed' if report['sft_updates'] == 1 and report['rl_updates'] >= 1 else 'incomplete_no_contributing_rl_update'
        # Independently check the built-in sum reduction using forward-only calls.
        from .rendering import sft_batch
        from .toy import toy_inputs
        report['provider_loss_checks'] = [fork.inspect_loss(sft_batch(fork.renderer, toy_inputs()['sft']), 'cross_entropy')]
        if report['rl_updates']:
            from .contracts import Trajectory, Generation, VerificationResult
            from .strategies import grpo_batch
            last_update = max(i for i, e in enumerate(events) if e['event'] == 'update')
            paths = [e['artifact'] for e in events[:last_update] if e['event'] == 'trajectory'][-8:]
            groups = {}
            for path in paths:
                value = read(path)
                if value['split'] != 'train':
                    continue
                value['generations'] = [Generation(**g) for g in value['generations']]
                value['verification'] = VerificationResult(**value['verification'])
                trajectory = Trajectory(**value)
                groups.setdefault(trajectory.group_id, []).append(trajectory)
            loss_rows, _ = grpo_batch(list(groups.values()))
            report['provider_loss_checks'].append(fork.inspect_loss(loss_rows, 'importance_sampling'))
        try:
            from .billing import collect_billing
            artifact_paths = []
            for file in (Path(config['output']) / 'checkpoints').glob('ckpt-*.json'):
                saved = read(file)
                if 'artifacts' in saved:
                    artifact_paths.extend([saved['artifacts']['training'], saved['artifacts']['sampler']])
            artifact_paths.extend([fork_artifacts['training'], fork_artifacts['sampler']])
            report['provider_billing'] = collect_billing(fork.service, artifact_paths, report['started_at'])
        except Exception as exc:
            report['provider_billing'] = {'actual_billing_usd': None, 'unavailable': type(exc).__name__}
        fork.close()
        report['spending'] = read(ledger_path)
        # Billing events may lag; do not assign unrelated account charges to this run.
        report['billing_note'] = 'Reservations include uncached token upper bounds and conservative 24-hour storage; not actual charges.'
    except Exception as exc:
        report['status'] = 'blocked_or_failed'
        report['error_type'] = type(exc).__name__
        if isinstance(exc, (ValueError, BudgetLimit)):
            report['reason'] = str(exc)[:500]
        else:
            report['reason'] = 'See locally recorded events; provider exception bodies are not persisted.'
        ledger_path = base / 'spend-ledger.json'
        if ledger_path.exists():
            report['spending'] = read(ledger_path)
        raise
    finally:
        report['finished_at'] = datetime.now(timezone.utc).isoformat()
        atomic_json(report_path, report)
        print('Smoke report:', report_path, flush=True)
    return report_path, report
