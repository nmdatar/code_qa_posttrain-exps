"""Optional W&B logging; local results remain authoritative if upload fails."""
import json
from .run import DIMS


def log_run(directory, config, rows, summary, mode, project, entity=None):
    try:
        import wandb
    except ImportError:
        raise RuntimeError('Local report saved. Install W&B: python -m pip install -r requirements-eval.txt') from None
    columns = ['id', 'repo', 'question', 'answer', 'reference_answer', 'status',
               'error', 'judge_reason', 'latency_seconds', 'input_tokens', 'output_tokens',
               'judge_input_tokens', 'judge_output_tokens', 'tool_calls', 'total_score', *DIMS]
    values = []
    for row in rows:
        scores = row['scores'] or {}
        item = dict(row, total_score=sum(scores.values()) if scores else None, **scores)
        values.append([item.get(c) for c in columns])
    organization = config.get('organization', {})
    tags = list(dict.fromkeys([config['mode'], *organization.get('tags', [])]))
    with wandb.init(project=project, entity=entity, id=config['run_id'],
                    name=organization.get('run_name') or config['run_id'],
                    group=organization.get('experiment_id'), notes=organization.get('notes'),
                    job_type='evaluation', tags=tags, config=config,
                    mode=mode, dir=str(directory.resolve()), save_code=False,
                    settings=wandb.Settings(disable_git=True)) as run:
        run.summary.update({f'eval/{k}': v for k, v in summary.items() if v is not None})
        run.log({'examples': wandb.Table(columns=columns, data=values)})
        run.log({'report': wandb.Html(str(directory/'report.html'), inject=False)})
        artifact = wandb.Artifact('qa-eval-'+config['run_id'], type='evaluation')
        for filename in ('config.json', 'results.jsonl', 'summary.json', 'report.html'):
            artifact.add_file(str(directory/filename), name=filename)
        run.log_artifact(artifact)
        (directory/'wandb.json').write_text(json.dumps(
            dict(id=run.id, url=run.url if mode == 'online' else None, mode=mode), indent=2))
