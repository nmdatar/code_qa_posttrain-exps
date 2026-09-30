"""Version configs and cohort identities while preserving exact membership."""
from pathlib import Path
from training_pipeline.storage import read, atomic_json
from training_pipeline.config import inputs, validate_config
from training_pipeline.collection import load_collection
from training_pipeline.cohorts import build_cohorts
from training_pipeline.benchmark import task_manifest, selected_tasks

base = Path('configs/experiments/grpo-autoresearch')
release = read('reports/grpo-autoresearch/fixes-v1/release-validation.json')
for mode, source in [('benchmark','paginate-v1.json'),('training','train-pagination-v2.json')]:
    c = read(base/source)
    c['run_id'] = 'autoresearch-fixes-v1-'+mode+'-seed42'
    c['output'] = 'artifacts/experiments/'+c['run_id']
    c['environment'].update(release=release['release'],manifest_sha256=release['manifest_sha256'],
        tool_action_policy='action-alias-v1',judge_evidence_policy='definition-context-v1')
    data = load_collection(c['environment'])
    old = read(c['evaluation']['cohort_manifest'])
    new = build_cohorts(data,len(old['selection']),old['seed'])
    assert new['selection'] == old['selection'] and new['confirmation'] == old['confirmation']
    cp = base/'fixes-v1-cohorts.json';atomic_json(cp,new)
    c['evaluation'].update(cohort_manifest=str(cp),cohort_sha256=new['manifest_hash'])
    if 'benchmark' in c:
        old = read(c['benchmark']['task_manifest']);new = task_manifest(data,len(old['task_ids']))
        assert new['task_ids'] == old['task_ids']
        bp = base/'fixes-v1-training-eight.json';atomic_json(bp,new)
        c['benchmark'].update(task_manifest=str(bp),manifest_hash=new['manifest_hash'])
        selected_tasks(c,data)
    c['tracking']['notes']='Training-only explicit claims, strict definition context, bounded action alias. Fresh same-condition baseline required; old reward curves are not directly comparable.'
    validate_config(c);inputs(c)
    atomic_json(base/('fixes-v1-'+mode+'.json'),c)
print('Config and input validation passed; selection, confirmation and benchmark membership preserved.')
