"""Bounded live calibration of saved answers; never updates a policy."""
import copy
import hashlib
import time
from pathlib import Path
from .storage import read, atomic_json, digest


def prepare(config, output):
    from .remote import prepare as prepare_remote
    from .config import inputs
    c = read(config)
    data = inputs(c)
    rows = {r['id']: r for r in data['tasks'] + data['development']}
    baseline = Path('artifacts/experiment-01-v4-results/artifacts/experiments/01-baseline-qwen4b-qwen397-seed42-all-claims-v3-modal-protocol-v4/trajectories')
    recent = Path('artifacts/reward-shaping-v3-results/artifacts/experiments/reward-shaping-v3-grader-v5-four-attempt-diagnostic/trajectories')
    cases = []
    def add(name, path, expected, text=None, uncited=False):
        t = read(path); row = copy.deepcopy(rows[t['task_id']])
        original_split = row['split']; row['split'] = 'train'
        answer = copy.deepcopy(t['submission'])
        if text is not None: answer['text'] = text
        if uncited: answer['citations'] = []
        cases.append({'name': name, 'source_trajectory': str(path), 'original_split': original_split,
            'controlled_variant': text is not None or uncited, 'expected_reward': expected,
            'request': {'episode_id': name, 'question': row['public']['user_prompt'],
                'answer': answer, 'rubric': row['rubric'], 'source_row': row,
                'observed_files': {x['path']:x['file_sha256'] for x in answer['citations']},
                'answer_evidence': answer['citations']}})
    correct = baseline/'ep-a5d5b26c9e0f41038e772f592b93f91b.json'
    add('saved-correct', correct, 1.)
    add('correct-uncited', correct, 1., uncited=True)
    add('partial-uncited', correct, .5,
        text='The IndexVariable name setter raises AttributeError, preventing in-place name assignment.', uncited=True)
    add('mixed-uncited', correct, .5,
        text='The IndexVariable name setter raises AttributeError, preventing in-place name assignment. However, get_level_variable modifies the original IndexVariable in place and returns that same object.', uncited=True)
    add('wrong-uncited', correct, 0.,
        text='The IndexVariable name setter permits in-place assignment, and get_level_variable modifies and returns the original object.', uncited=True)
    add('saved-partial', baseline/'ep-96db7c335b404e05a06157d6dd07a14c.json', .5)
    add('saved-no-match', recent/'ep-713844946c1d4ca0a056495d2d762308.json', 0.)
    fixture = {'cases': cases, 'purpose': 'Training-grader calibration only; selection answers are not training samples or new evaluation results. No confirmation data used.', 'optimizer_updates': 0}
    fixture['fixture_hash'] = digest(fixture)
    result = prepare_remote([config], output)
    root = Path(output)
    atomic_json(root/'reward-validation.json', fixture)
    m = read(root/'bundle.json');m.pop('bundle_id')
    m['reward_validation'] = {'fixture': 'reward-validation.json', 'max_reserved_usd': 5.}
    m['files']['reward-validation.json'] = hashlib.sha256((root/'reward-validation.json').read_bytes()).hexdigest()
    m['bundle_id'] = digest(m);atomic_json(root/'bundle.json', m)
    result['bundle_id'] = m['bundle_id'];result['validation_upper_bound_usd'] = 5.
    return result


def run(config, fixture, root):
    from .budget import SpendLedger
    from .collection import CollectionFactory
    from .remote import require_remote_worker
    require_remote_worker(config)
    expected = fixture['fixture_hash']
    if digest({k:v for k,v in fixture.items() if k!='fixture_hash'}) != expected:
        raise ValueError('Calibration fixture changed')
    if len(fixture['cases']) > 8: raise ValueError('Calibration case limit exceeded')
    root=Path(root);root.mkdir(parents=True,exist_ok=False)
    ledger=SpendLedger(config['spend']['ledger'],config['spend']['cap_usd'],config['spend']['prices'],config['model']['checkpoint_ttl_seconds'])
    before=read(ledger.path)['reserved_usd']
    factory=CollectionFactory(config,root,ledger)
    report={'fixture_hash':expected,'status':'running','cases':[],'optimizer_updates':0,'actual_billing_usd':None}
    started=time.monotonic()
    def save():
        report['reserved_usd']=read(ledger.path)['reserved_usd']-before
        report['elapsed_seconds']=time.monotonic()-started
        atomic_json(root/'results.json',report)
    try:
        for case in fixture['cases']:
            # Bound all four possible calls (extraction/assessment with one repair).
            j=config['judge'];p=j['prices']
            bound=4*(j['context_tokens']*p['prefill']+j['max_tokens']*p['sample'])/1e6
            if read(ledger.path)['reserved_usd']-before+bound>5.:
                raise ValueError('Calibration reservation limit reached')
            entry={k:v for k,v in case.items() if k!='request'}
            try:
                g=factory.grade(case['request']);entry['grade']=g
                entry['passed']=g['status']=='resolved' and g['training_feedback']['reward']==case['expected_reward']
                if case['name']=='saved-no-match':
                    entry['passed']=entry['passed'] and not g['semantic']['extracted_claims']
            except (ValueError, KeyError, TypeError) as exc:
                entry.update(passed=False,error_type=type(exc).__name__,error=str(exc)[:500])
            report['cases'].append(entry);save()
            print(case['name'], 'passed='+str(entry['passed']), flush=True)
        report['status']='passed' if all(c['passed'] for c in report['cases']) else 'failed'
        save();return report['status']=='passed'
    finally:
        factory.close();save()
