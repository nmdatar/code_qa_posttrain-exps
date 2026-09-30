"""Inspect recorded judge outputs without rerunning the judge or exporting prompts."""
import argparse
import json
import re
from pathlib import Path

from .storage import atomic_json
from .trajectory_viewer import STYLE, block, esc, load_traces, pretty


ATTEMPT = re.compile(r'^(?P<episode>.+?)\.(?:(?P<stage>extract|assess)(?:\.repair-(?P<repair>\d+))?\.judge-raw|coverage-(?P<coverage>\d+)|(?P<legacy>judge-raw))\.json$')


def collect(root):
    root = Path(root)
    episodes = {}
    for trace, context in load_traces(root):
        verification = trace.get('verification') or {}
        diagnostics = verification.get('diagnostics') or {}
        episodes[trace['episode_id']] = {
            'episode_id': trace['episode_id'], 'task_id': trace.get('task_id'),
            'phase': context.get('phase', trace.get('split')),
            'optimizer_step': context.get('optimizer_step'),
            'answer': trace.get('submission'),
            'status': verification.get('status'), 'reward': verification.get('reward'),
            'strict_score': diagnostics.get('strict_score'),
            'strict_status': diagnostics.get('strict_status'),
            'training_reward': diagnostics.get('training_reward'),
            'reasons': verification.get('reasons'),
            'semantic': diagnostics.get('semantic'),
            'coverage': diagnostics.get('training_coverage'), 'attempts': [],
        }
    for path in sorted((root/'private').glob('*.json')):
        match = ATTEMPT.fullmatch(path.name)
        if not match:
            continue
        episode = match['episode']
        record = episodes.setdefault(episode, {'episode_id': episode, 'attempts': []})
        attempt = {'artifact': path.name,
                   'stage': match['stage'] or ('coverage' if match['coverage'] is not None else 'legacy'),
                   'attempt': int(match['repair'] or match['coverage'] or 0)}
        try:
            raw = json.loads(path.read_text())
            generation = raw.get('generation') or {}
            # Explicit fields only: generation.prompt and request evidence stay private.
            attempt.update(response=generation.get('text', ''),
                           stop_reason=generation.get('stop_reason'),
                           output_tokens=len(generation['tokens']) if isinstance(generation.get('tokens'), list) else None,
                           timing=raw.get('timing'), judge_identity=raw.get('judge_identity'),
                           validation_error=raw.get('validation_error'))
            payload = raw.get('payload') or raw.get('request') or {}
            record.setdefault('question', payload.get('question') or payload.get('untrusted', {}).get('question'))
            claims = payload.get('reference_claims') or payload.get('rubric', {}).get('claims')
            if claims:
                record['reference_claims'] = [{k: c.get(k) for k in ('id', 'text', 'weight')} for c in claims]
            candidate = payload.get('candidate_answer')
            if isinstance(candidate, dict):
                record.setdefault('answer', candidate)
        except (OSError, ValueError, TypeError, AttributeError) as exc:
            attempt['read_error'] = str(exc)
        record['attempts'].append(attempt)
    for record in episodes.values():
        record['attempts'].sort(key=lambda a: ({'extract': 0, 'assess': 1, 'coverage': 2, 'legacy': 3}[a['stage']], a['attempt']))
    return list(episodes.values())


def render(records, title='Judge answers'):
    parts = ['<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',
             '<title>'+esc(title)+'</title>', STYLE, '<body><h1>'+esc(title)+'</h1>',
             '<p>Recorded judge outputs; no new grading. Missing scores remain unknown. Raw responses are model output, not necessarily validated findings.</p>',
             '<input id="search" aria-label="Search judge answers" placeholder="Search episode, claim, verdict or reason">']
    for record in records:
        parts.append('<details class="episode"><summary>'+esc(record['episode_id'])+
                     ' · strict '+esc(record.get('strict_score'))+' · training '+esc(record.get('training_reward'))+'</summary>')
        parts.append(block('Scores and status', {k:record.get(k) for k in
                     ('task_id', 'phase', 'optimizer_step', 'status', 'reward', 'strict_status', 'strict_score', 'training_reward', 'reasons')}, True))
        for key, label in [('question', 'Question'), ('answer', 'Candidate answer'),
                           ('reference_claims', 'Required facts'), ('semantic', 'Validated strict assessment'),
                           ('coverage', 'Validated factual coverage')]:
            if record.get(key) is not None:
                parts.append(block(label, record[key]))
        if not record['attempts']:
            parts.append('<p>Raw judge responses are not available in this archive.</p>')
        for attempt in record['attempts']:
            parts.append(block(attempt['stage']+' · attempt '+str(attempt['attempt'])+' · metadata',
                               {k:v for k,v in attempt.items() if k != 'response'},
                               error=bool(attempt.get('validation_error') or attempt.get('read_error'))))
            parts.append('<details open><summary>Exact judge response</summary><pre>'+esc(attempt.get('response', 'Unavailable'))+'</pre></details>')
        parts.append('</details>')
    parts.append("<script>document.getElementById('search').addEventListener('input',function(){for(const e of document.querySelectorAll('.episode'))e.hidden=!e.textContent.toLowerCase().includes(this.value.toLowerCase())})</script></body></html>")
    return ''.join(parts)


def write_report(root, output=None):
    root = Path(root)
    records = collect(root)
    output = Path(output) if output else root/'judge-viewer.html'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render(records, root.name+' · judge answers'), encoding='utf-8')
    atomic_json(output.with_suffix('.json'), {'episodes': records})
    return output, records


def log_report(remote, wandb, path, records):
    columns = ['episode_id', 'task_id', 'phase', 'optimizer_step', 'strict_status', 'strict_score',
               'training_reward', 'stage', 'attempt', 'stop_reason', 'output_tokens',
               'validation_error', 'read_error', 'timing', 'judge_identity', 'response']
    rows = []
    for record in records:
        for attempt in record['attempts']:
            merged = {**record, **attempt}
            rows.append([pretty(merged.get(k)) if k in ('timing', 'judge_identity') else merged.get(k) for k in columns])
    remote.log({'judge_answers': wandb.Table(columns=columns, data=rows),
                'judge_viewer': wandb.Html(Path(path).read_text(), inject=False)})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_root')
    parser.add_argument('--output')
    args = parser.parse_args()
    path, records = write_report(args.run_root, args.output)
    print(f'{path}: {len(records)} episodes')
