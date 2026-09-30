"""Live, frozen, training-only grading controls; no policy or optimizer calls."""
import copy
from dataclasses import asdict
import math
from pathlib import Path
from types import SimpleNamespace
from .storage import read, atomic_json, digest
from .contracts import ConfigurationError

VERSION = 'aligned-live-controls-v1'
VARIANTS = ('correct', 'partial', 'wrong', 'mixed_contradiction', 'unsupported_extra', 'uncited', 'invalid_citation')


def judge_bound(config, cases):
    judge = config['judge']; prices = judge['prices']
    per_call = ((judge['context_tokens']-judge['max_tokens'])*prices['prefill'] +
                judge['max_tokens']*prices['sample'])/1e6
    return cases * 3 * (1+judge['repair_attempts']) * per_call


def validate_fixture(fixture, data):
    from qa_eval.schema import validate, SUBMISSION
    if fixture['version'] != VERSION or fixture['fixture_hash'] != digest({k:v for k,v in fixture.items() if k!='fixture_hash'}):
        raise ConfigurationError('Live control fixture identity mismatch')
    if fixture['data_identity'] != data['identity'] or len(fixture['cases']) != 56:
        raise ConfigurationError('Live controls require pinned data and56cases')
    train = {row['id']:row for row in data['tasks']}
    heldout = {row['id'] for row in data['development']}
    if len(fixture['sources']) != 4 or len({x['task_id'] for x in fixture['sources']}) != 4:
        raise ConfigurationError('Controls require four distinct training tasks')
    for source in fixture['sources']:
        task_id = source['task_id']
        if task_id not in train or task_id in heldout or train[task_id]['split'] != 'train':
            raise ConfigurationError('Control source is not admitted training data')
        row = train[task_id]
        if (source['rubric_hash'] != row['rubric']['rubric_hash'] or
                source['repository'] != row['public']['repository'] or source['evidence'] != row['excerpts']):
            raise ConfigurationError('Control source/rubric binding changed')
    expected = {(source['task_id'], variant, replicate)
                for source in fixture['sources'] for variant in VARIANTS for replicate in (1,2)}
    actual = [(case['task_id'], case['variant'], case['replicate']) for case in fixture['cases']]
    if set(actual) != expected or len(actual) != len(set(actual)):
        raise ConfigurationError('Missing or duplicate frozen control case')
    if len({case['case_id'] for case in fixture['cases']}) != 56:
        raise ConfigurationError('Control case IDs must be unique')
    for case in fixture['cases']:
        validate(case['answer'], SUBMISSION)
        if case['answer']['task_id'] != case['task_id']:
            raise ConfigurationError('Control answer/task binding mismatch')
    return train


def evaluate_gate(records, fixture):
    failures = []
    eligible = []
    for record in records:
        if record.get('eligible') is not True:
            continue
        values = [record.get('reward'), record.get('coverage')]
        components = record.get('components') or {}
        values += [components.get(key) for key in ('contradicted', 'unsupported', 'uncited', 'bad_citation')]
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) for value in values):
            failures.append('nonfinite_or_missing_eligible_numeric_value')
            continue
        if not 0 <= record['coverage'] <= 1 or any(value < 0 for value in values[2:]):
            failures.append('invalid_eligible_numeric_range')
            continue
        eligible.append(record)
    if len(records) != 56 or len(eligible) < 54:
        failures.append('eligible_coverage_below_54_of_56')
    tasks = {}
    for source in fixture['sources']:
        tid = source['task_id']; groups = {v:[r for r in eligible if r['task_id']==tid and r['variant']==v] for v in VARIANTS}
        if any(not values for values in groups.values()):
            failures.append(tid+':missing_eligible_variant'); continue
        means = {key:sum(r['reward'] for r in values)/len(values) for key,values in groups.items()}
        task_failures = []
        if not (means['correct'] >= .9 and .1 <= means['partial'] <= .8 and
                means['wrong'] <= .05 and means['correct'] > means['partial'] > means['wrong']):
            task_failures.append('correct_partial_wrong_ranking')
        if any(r['coverage'] < .95 or r['components']['contradicted'] or r['components']['unsupported'] or r['components']['uncited'] or r['components']['bad_citation'] for r in groups['correct']):
            task_failures.append('false_penalty_or_incomplete_correct_coverage')
        if any(not .25 <= r['coverage'] <= .75 for r in groups['partial']):
            task_failures.append('partial_coverage_outside_prespecified_range')
        if not any(r['components']['contradicted'] >= 1 for r in groups['mixed_contradiction']):
            task_failures.append('source_contradiction_not_detected')
        if means['mixed_contradiction'] > means['correct']-.5:
            task_failures.append('source_contradiction_penalty_too_small')
        if not any(r['components']['unsupported'] >= 1 and r['components']['contradicted']==0 for r in groups['unsupported_extra']):
            task_failures.append('unsupported_not_distinguished_from_contradiction')
        if means['unsupported_extra'] > means['correct']-.1:
            task_failures.append('unsupported_penalty_too_small')
        for variant in ('uncited','invalid_citation'):
            if means[variant] > means['correct']-.05+1e-9:
                task_failures.append(variant+':penalty_missing')
        if any(r['components']['bad_citation'] < 1 for r in groups['invalid_citation']):
            task_failures.append('invalid_citation_component_missing')
        tasks[tid] = {'mean_reward':means, 'eligible_replicates':{k:len(v) for k,v in groups.items()}, 'failures':task_failures}
        failures.extend(tid+':'+value for value in task_failures)
    return {'passed':not failures, 'failures':failures, 'eligible':len(eligible), 'expected':56,
            'tasks':tasks, 'gate_version':VERSION, 'judge_accuracy_established':False}


class FixtureEnvelope:
    """Explicit non-rollout envelope; never eligible for SFT admission."""
    def __init__(self, case_id): self.case_id = case_id
    def finish(self, submission, key, termination):
        return {'kind':'grading_control_not_policy_trajectory', 'case_id':self.case_id,
                'submission':submission, 'termination':termination, 'training_admission_allowed':False}


def run(config, fixture, root):
    from .config import inputs
    from .remote import require_remote_worker
    from .budget import SpendLedger
    from .collection import CollectionFactory, CollectionEpisode
    require_remote_worker(config)
    if config['training_reward']['version'] != 'aligned-coverage-v1':
        raise ConfigurationError('Live gate requires the frozen aligned reward')
    data = inputs(config); train = validate_fixture(fixture, data)
    root = Path(root); root.mkdir(parents=True, exist_ok=False)
    ledger = SpendLedger(config['spend']['ledger'], config['spend']['cap_usd'],
                         config['spend']['prices'], config['model']['checkpoint_ttl_seconds'])
    before = read(ledger.path)['reserved_usd']
    if before + judge_bound(config, len(fixture['cases'])) > ledger.cap:
        raise ConfigurationError('All frozen controls cannot fit remaining ledger')
    factory = CollectionFactory(config, root, ledger)
    report = {'version':VERSION,'fixture_hash':fixture['fixture_hash'],'status':'running','cases':[],
              'policy_generation_calls':0,'optimizer_calls':0,'training_data_created':False,
              'selection_or_confirmation_used':False,'actual_billing_usd':None}
    def save():
        report['incremental_judge_reservations_usd'] = read(ledger.path)['reserved_usd']-before
        atomic_json(root/'results.json', report)
    try:
        for case in fixture['cases']:
            episode = CollectionEpisode.__new__(CollectionEpisode)
            episode.episode_id = case['case_id']; episode.row = train[case['task_id']]
            episode.factory = factory; episode.recorder = FixtureEnvelope(case['case_id'])
            episode.sandbox = SimpleNamespace(close=lambda:None)
            episode.observed_files = {ref['path']:ref['file_sha256'] for ref in episode.row['excerpts']}
            result = episode.verify(SimpleNamespace(submission=copy.deepcopy(case['answer']), termination='completed'))
            diagnostics = result.diagnostics; feedback = diagnostics.get('training_feedback') or {}
            coverage = (diagnostics.get('training_coverage') or {}).get('score')
            entry = {key:case[key] for key in ('case_id','task_id','variant','replicate')}
            entry.update(verification=asdict(result), eligible=result.status=='resolved' and feedback.get('eligible') is True,
                         reward=result.reward, coverage=coverage, components=feedback.get('components',{}))
            report['cases'].append(entry); save()
            print(case['case_id'], 'eligible='+str(entry['eligible']), flush=True)
        report['gate'] = evaluate_gate(report['cases'],fixture)
        report['status'] = 'passed' if report['gate']['passed'] else 'failed_gate'
        save()
        return report
    finally:
        factory.close(); save()
