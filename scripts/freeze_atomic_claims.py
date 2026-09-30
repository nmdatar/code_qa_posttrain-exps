"""Reproduce the reviewed release without additional model calls.

Run from the repository root with PYTHONPATH=. and the original release present.
The destination must not exist. Model decisions and assistant corrections remain
separate in the audit; neither is labeled human gold.
"""
import argparse
import json
from pathlib import Path
from training_pipeline.atomic_claim_experiment import originals, validate_split, apply_split, rewrite_release


def freeze(source, target, model_report, resolutions):
    rows = [json.loads(line) for line in (Path(source) / 'private/grading.jsonl').read_text().splitlines()]
    decisions = {item['task_id']: item for item in json.loads(Path(model_report).read_text())['splits']}
    manual = json.loads(Path(resolutions).read_text())
    assert set(decisions) == {row['task_id'] for row in rows}
    assert set(manual) <= set(decisions)
    output, audit = [], []
    for row in rows:
        ident = row['task_id']
        parents = originals(row)
        decision = decisions[ident]
        if ident in manual:
            facts = manual[ident]['facts'] or [[c['text']] for c in parents]
            if len(facts) != len(parents):
                raise ValueError('Resolution parent count differs: ' + ident)
            mapping = validate_split({'claims': [
                {'id': c['id'], 'facts': [{'text': text, 'quote': c['text']} for text in children]}
                for c, children in zip(parents, facts)]}, parents)
        else:
            if decision['status'] != 'approved':
                raise ValueError('Unresolved model review: ' + ident)
            mapping = decision['mapping']
        revised = apply_split(row, mapping)
        output.append(revised)
        audit.append({'task_id': ident, 'model_audit': decision,
                      'assistant_resolution': manual.get(ident), 'final_mapping': mapping,
                      'status': 'resolved', 'before': len(parents), 'after': len(originals(revised))})
    rewrite_release(source, target, output, audit)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default='data/releases/repo-qa-automated-rollouts-v1')
    parser.add_argument('--target', default='data/releases/repo-qa-atomic-claims-v1')
    parser.add_argument('--model-report', default='reports/atomic-claims-judge-comparison/live-progress.json')
    parser.add_argument('--resolutions', default='reports/atomic-claims-judge-comparison/manual-resolutions.json')
    args = parser.parse_args()
    freeze(args.source, args.target, args.model_report, args.resolutions)
