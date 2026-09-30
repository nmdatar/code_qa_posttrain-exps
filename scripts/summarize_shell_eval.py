"""Summarize downloaded shell-only-v1 reports without hiding unresolved grades."""
import argparse
import copy
import json
from pathlib import Path
from statistics import mean, median
from training_pipeline.comparison import compare
from training_pipeline.storage import atomic_json


def summarize(root, configs):
    reports = {}; traces = {}; specs = {}
    for arm in ['control', 'shell']:
        for repeat in [1, 2]:
            name = f"{arm}-r{repeat}"
            spec = json.loads((configs/(name+'.json')).read_text()); specs[name] = spec
            folder = root/spec['output']
            paths = list((folder/'evaluations').glob('*.json'))
            if len(paths) != 1: raise ValueError(f"Expected one complete evaluation for {name}, got {len(paths)}")
            report = json.loads(paths[0].read_text()); reports[name] = report
            traces[name] = [json.loads((folder/'trajectories'/(row['episode_id']+'.json')).read_text()) for row in report['results']]
    base_spec = specs['control-r1']; base_report = reports['control-r1']
    for name, spec in specs.items():
        for key in ['model', 'environment', 'limits', 'evaluation', 'judge', 'concurrency', 'seed', 'training_reward']:
            if spec[key] != base_spec[key]: raise ValueError('Unmatched condition: '+key)
        if {k:v for k,v in spec['harness'].items() if k != 'interface'} != {k:v for k,v in base_spec['harness'].items() if k != 'interface'}:
            raise ValueError('Harness differs beyond interface')
        if spec['harness']['interface'] != ('source-shell-v1' if name.startswith('shell') else 'structured-v1'):
            raise ValueError('Wrong interface')
        report = reports[name]
        for key in ['data_identity', 'reward_version', 'cohort', 'policy_id']:
            if report[key] != base_report[key]: raise ValueError('Unmatched report: '+key)
        if report['optimizer_step'] != 0: raise ValueError('Expected frozen base')
    # The environment identity intentionally differs with the tool interface.
    # Validate every other experimental condition above before this explicit
    # comparison-only normalization; raw reports retain their real identities.
    normalized = {name:copy.deepcopy(report) for name,report in reports.items()}
    for report in normalized.values(): report['environment'] = base_report['environment']
    paired = [compare(normalized[f'control-r{i}'], [normalized[f'shell-r{i}']]) for i in [1,2]]
    arms = {}
    for name, report in reports.items():
        ts = traces[name]; latencies = sorted(t['usage']['latency_seconds'] for t in ts)
        arms[name] = {k:report[k] for k in ['expected','resolved','demonstrated_quality','mean_reward','scoring_coverage','completion_rate','reserved_cost_usd']}
        arms[name].update(input_tokens=sum(t['usage']['input_tokens'] for t in ts),
            output_tokens=sum(t['usage']['output_tokens'] for t in ts),
            tool_calls=sum(t['usage']['tool_calls'] for t in ts),
            invalid_actions=sum(1 for t in ts for e in t['events'] if e['kind']=='observation' and isinstance(e['value'],dict) and str(e['value'].get('error','')).startswith('Invalid action')),
            latency_p50=median(latencies), latency_p95=latencies[int(.95*(len(latencies)-1))])
    return {'status':'complete','arms':arms,'paired_repeats':paired,
        'mean_quality_difference':mean(p['quality_difference'] for p in paired),
        'environment_identities':{k:r['environment'] for k,r in reports.items()},
        'note':'Selection screen only. Reported quality treats unresolved as unproven zero, with sensitivity bounds. Same Qwen judge family; no independent human calibration. Repeats are inference repetitions, not training seeds. Reserved costs are not invoices. No automatic promotion.'}


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--configs',type=Path,default=Path('configs/experiments/shell-only-v1'));p.add_argument('--output',type=Path,default=Path('reports/shell-only-v1/summary.json'));a=p.parse_args()
    result=summarize(a.root,a.configs);atomic_json(a.output,result);print(json.dumps(result,indent=2))
