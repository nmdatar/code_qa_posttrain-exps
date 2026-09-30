"""Freeze assistant-decomposed existing references on TRAINING only, without model calls.

Facts are reviewed for equivalence against previously source-reviewed reference prose.
Pinned source hashes/ranges are verified, not newly certified as human gold. The
original reference text, all evaluation records and public tasks remain unchanged.
"""
import argparse
import copy
import json
from pathlib import Path

from training_pipeline.admission import sha, verified_source
from training_pipeline.atomic_claim_experiment import rewrite_release
from training_pipeline.collection import load_collection
from training_pipeline.storage import atomic_json

PLACEHOLDER = 'Corrected central explanation and coverage of the original question'


def revised_reference(row, draft, facts):
    reference = row['reference']
    if row['split'] != 'train':
        raise ValueError('Only training references may change')
    if (draft['task_id'] != row['id'] or draft['reference_sha256'] != reference['reference_sha256']
            or sha(draft['reference_text_for_decomposition'].encode()) != reference['reference_sha256']):
        raise ValueError('Reference identity mismatch')
    if [c['claim'] for c in reference['reviewed_claims']] != [PLACEHOLDER]:
        raise ValueError('Expected sole placeholder claim')
    if not facts or len(facts) != len(set(facts)) or any(not isinstance(f,str) or not f.strip() or f == PLACEHOLDER for f in facts):
        raise ValueError('Invalid explicit facts')
    result = copy.deepcopy(reference)
    refs = [{k:r[k] for k in ('path','start_line','end_line')} for r in reference['verified_evidence']]
    result['reviewed_claims'] = [{'claim':f, 'verdict':'supported', 'evidence':refs[0],
        'supporting_evidence':refs[1:], 'reason':'Assistant decomposition of the existing source-reviewed reference; not independent human review.'} for f in facts]
    result['claim_revision'] = 'training-placeholder-facts-v1'
    result['claim_review'] = {'method':'assistant_reference_equivalence',
        'source_review':'inherited_from_selected_reference', 'human_reviewed':False,
        'reference_sha256':reference['reference_sha256']}
    return result


def freeze(config, drafts, facts_path, target):
    config = json.loads(Path(config).read_text())
    source = Path(config['environment']['release'])
    if Path(target).exists():
        raise ValueError('Destination exists; never overwrite a release')
    verified_source(source)
    # Verifies each selected reference, source hash and evidence range binding.
    rows = {r['id']:r for r in load_collection(config['environment'])['tasks']}
    drafts = json.loads(Path(drafts).read_text())['tasks']
    decisions = {}
    for line in Path(facts_path).read_text().splitlines():
        number, *facts = line.split('|')
        number = int(number)
        if number in decisions:
            raise ValueError('Duplicate decomposition index')
        decisions[number] = facts
    if set(decisions) != set(range(len(drafts))):
        raise ValueError('Decomposition must cover each staged training reference')
    expected = {r['id'] for r in rows.values() if any(c['claim'] == PLACEHOLDER for c in r['reference']['reviewed_claims'])}
    if {d['task_id'] for d in drafts} != expected:
        raise ValueError('Training placeholder inventory changed')
    replacements, audit = {}, []
    for i, draft in enumerate(drafts):
        row = rows[draft['task_id']]
        replacements[row['id']] = revised_reference(row, draft, decisions[i])
        audit.append({'task_id':row['id'], 'split':'train', 'original':row['reference'],
                      'facts':decisions[i], 'review':'assistant_reference_equivalence',
                      'pinned_evidence_verified':True, 'human_reviewed':False})
    originals = [json.loads(s) for s in (source/'private/grading.jsonl').read_text().splitlines()]
    records = [replacements.get(r['task_id'],r) for r in originals]
    rewrite_release(source, target, records, audit)
    target = Path(target)
    manifest = json.loads((target/'manifest.json').read_text())
    manifest['claim_revision'] = 'training-placeholder-facts-v1'
    manifest['claim_revision_note'] = '78 training placeholder rubrics expanded from existing reviewed prose. Assistant equivalence review, not human gold. Evaluation records unchanged.'
    atomic_json(target/'manifest.json', manifest)
    summary = {'release':str(target.resolve()), 'manifest_sha256':sha((target/'manifest.json').read_bytes()),
               'changed_training_tasks':len(replacements), 'explicit_claims':sum(len(v['reviewed_claims']) for v in replacements.values()),
               'unchanged_other_grading_records':sum(r['task_id'] not in replacements for r in originals),
               'human_reviewed':False, 'new_model_calls':0}
    for path in source.rglob('*'):
        if path.is_file() and path.relative_to(source).as_posix() not in {'manifest.json','private/grading.jsonl','private/atomic-claims-audit.json'}:
            if path.read_bytes() != (target/path.relative_to(source)).read_bytes():
                raise ValueError('Unrelated release artifact changed')
    verified_source(target)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/experiments/grpo-autoresearch/train-pagination-v2.json')
    parser.add_argument('--drafts', default='reports/grpo-autoresearch/placeholder-repair-inputs.json')
    parser.add_argument('--facts', default='reports/grpo-autoresearch/fixes-v1/claim-facts.txt')
    parser.add_argument('--target', default='data/releases/repo-qa-training-claims-v5')
    args = parser.parse_args()
    print(json.dumps(freeze(args.config, args.drafts, args.facts, args.target), indent=2))
