"""Attach saved judge answers to an existing W&B run without resuming it."""
import argparse
import hashlib
from pathlib import Path
from types import SimpleNamespace

import wandb
from training_pipeline.judge_viewer import write_report, log_report
from training_pipeline.storage import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_root')
    parser.add_argument('--run', required=True, help='entity/project/run_id')
    parser.add_argument('--output', default='reports/judge-observability')
    args = parser.parse_args()
    entity, project, ident = args.run.split('/')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    path, records = write_report(args.run_root, output/'judge-viewer.html')
    captured = {}
    log_report(SimpleNamespace(log=captured.update), wandb, path, records)
    api = wandb.Api(timeout=60)
    original = api.run(args.run)
    before = original.state
    name = 'judge-inspection-'+hashlib.sha256(args.run.encode()).hexdigest()[:20]
    artifact = wandb.Artifact(name, type='judge-inspection',
        description='Saved judge answers, claim verdicts, strict/training scores and repair attempts. No new grading.',
        metadata={'source_run':args.run, 'episodes':len(records),
                  'attempts':sum(len(r['attempts']) for r in records), 'snapshot':True})
    for key, value in captured.items():
        artifact.add(value, key)
    artifact.add_file(str(path), name='judge-viewer.html')
    artifact.add_file(str(path.with_suffix('.json')), name='judge-viewer.json')
    with wandb.init(entity=entity, project=project, job_type='judge-viewer',
                    name='Judge inspection · '+ident, dir=str(output),
                    settings=wandb.Settings(disable_git=True)) as publisher:
        publisher.log(captured)
        uploaded = publisher.log_artifact(artifact)
        uploaded.wait()
        attached = original.log_artifact(api.artifact(uploaded.qualified_name))
        original.summary['judge_inspection_url'] = attached.url
        original.summary['judge_inspection_episodes'] = len(records)
        original.summary['judge_inspection_attempts'] = artifact.metadata['attempts']
        original.summary.update()
        receipt = {'run':args.run, 'artifact':attached.qualified_name, 'url':attached.url,
                   'viewer_run_url':publisher.url, 'original_state':before,
                   'episodes':len(records), 'attempts':artifact.metadata['attempts']}
        atomic_json(output/'wandb-upload.json', receipt)
    checked = wandb.Api(timeout=60).run(args.run)
    names = {a.qualified_name for a in checked.logged_artifacts() if a.type == 'judge-inspection'}
    assert receipt['artifact'] in names, 'Judge artifact not attached'
    assert checked.summary.get('judge_inspection_url') == receipt['url'], 'Missing summary link'
    entries = api.artifact(receipt['artifact']).manifest.entries
    assert 'judge_answers.table.json' in entries and 'judge-viewer.html' in entries
    atomic_json(output/'wandb-verification.json', {'attached':True, 'table_present':True,
                'summary_link_present':True, 'state':checked.state, 'original_state':before})
    print(receipt)


if __name__ == '__main__':
    main()
