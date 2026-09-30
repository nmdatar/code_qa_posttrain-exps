"""Freeze decomposition and matched distillation configs offline, without provider calls."""
import argparse
import copy
import json
import math
from pathlib import Path

from training_pipeline.config import inputs
from training_pipeline.remote import operation_estimate, controller_reservation
from training_pipeline.storage import atomic_json


def prepare(destination, source_directory='configs/experiments/current-v8'):
    root = Path(destination)
    if root.exists():
        raise ValueError('Use a fresh output directory; never overwrite an existing campaign')
    if not root.name.replace('-', '').replace('_', '').isalnum():
        raise ValueError('Use an alphanumeric campaign name')
    source = Path(source_directory)
    configs = {}
    def add(name, template, dependencies=(), **changes):
        c = json.loads((source / (template + '.json')).read_text())
        c.update(changes)
        c['run_id'] = name + '-' + root.name + '-seed42'
        c['output'] = 'artifacts/experiments/' + c['run_id']
        c['spend']['ledger'] = 'artifacts/project-budget/' + root.name + '/' + name + '.json'
        c['tracking'].update(notes='Offline research extension; see procedures 07/08. No launch implied.', tags=[root.name, name])
        cost = operation_estimate(c)['upper_estimate_usd'] + controller_reservation(c)
        c['spend']['cap_usd'] = math.ceil(cost * 1.10 / 10) * 10
        configs[name] = (c, list(dependencies), cost)
        return c
    for cohort in ['selection', 'confirmation']:
        for repeat in [1, 2]:
            for label in ['control', 'decompose']:
                name = f'07-{label}-{cohort}-r{repeat}'
                c = add(name, '06-control-r1')
                c['execution']['cohort'] = cohort
                c['harness']['planning'] = 'question-checklist-v1' if label == 'decompose' else 'none'
    for label, scope in [('investigation', 'all-admitted-turns'), ('answer-only', 'final-answer-only')]:
        name = '08-' + label + '-sft'
        c = add(name, '04-investigation-sft')
        c['supervised']['loss_scope'] = scope
        fork = add('08-' + label + '-then-grpo', '04-sft-then-grpo', [name])
        fork['execution']['checkpoint'] = c['output'] + '/checkpoints/best.json'
    add('08-direct-grpo', '02-direct-grpo')
    # Use the cached native tokenizer to verify archived prompt/target alignment.
    from transformers import AutoTokenizer
    from training_pipeline.rendering import ChatRenderer
    from training_pipeline.sft_collection import validate_rendering
    template = configs['08-investigation-sft'][0]
    renderer = ChatRenderer(AutoTokenizer.from_pretrained(template['model']['base_model'], local_files_only=True),
                            template['limits']['context_tokens'])
    entries = []
    for name, (c, deps, _) in configs.items():
        data = inputs(copy.deepcopy(c))
        if 'supervised' in c:
            validate_rendering(renderer, data['sft'])
        # Recompute after setting cohort and loss scope.
        cost = operation_estimate(c)['upper_estimate_usd'] + controller_reservation(c)
        c['spend']['cap_usd'] = math.ceil(cost * 1.10 / 10) * 10
        entries.append({'id':name, 'config':str(root / (name + '.json')), 'dependencies':deps,
                        'validation':'passed', 'sft_examples':len(data['sft']),
                        'sft_selected_turns':sum(len(e['assistant_turns']) for e in data['sft']),
                        'estimated_upper_usd_including_controller':cost, 'cap_usd':c['spend']['cap_usd'],
                        'readiness':'smoke-only; one admitted lineage' if name.startswith('08-') and name != '08-direct-grpo' else
                                    'finalist-only' if 'confirmation' in name else 'offline-validated',
                        'submitted':False})
    root.mkdir(parents=True)
    for name, (c, _, _) in configs.items():
        atomic_json(root / (name + '.json'), c)
    prior = json.loads((source / 'budget-plan.json').read_text())
    prior['project_ceiling_usd'] += sum(c['spend']['cap_usd'] for c, _, _ in configs.values())
    prior['ledger_caps'].update({c['spend']['ledger']:c['spend']['cap_usd'] for c, _, _ in configs.values()})
    prior['authorization'] = 'Proposed caps for offline preparation only; no new spending authorized or submitted.'
    prior['note'] = 'Preserves current-v8 allocations; alternatives and finalist controls are not an instruction to run all arms. Reconcile actual spend and price snapshots at dispatch.'
    atomic_json(root / 'budget-plan.json', prior)
    atomic_json(root / 'study-plan.json', {'version':root.name, 'new_launches':0, 'experiments':entries,
        'limitations':['Decomposition is a prompt-only intervention; no emitted or enforced checklist.',
                        'Answer-only SFT retains identical teacher investigation context and changes only loss selection.',
                        'Existing demonstrations are policy-generated, not a stronger teacher release; one lineage supports smoke checks only.']})
    print(json.dumps({'configs':len(configs), 'validation':'passed', 'new_launches':0}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='configs/experiments/research-extensions-v1')
    parser.add_argument('--source', default='configs/experiments/current-v8')
    args = parser.parse_args()
    prepare(args.output, args.source)
