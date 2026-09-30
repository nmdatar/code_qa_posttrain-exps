"""Portable, escaped HTML reports for recorded training/evaluation trajectories."""
import argparse
import html
import json
from pathlib import Path


def esc(value):
    return html.escape(str(value), quote=True)


def pretty(value):
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (ValueError, TypeError):
            return value
    return json.dumps(value, indent=2, ensure_ascii=False)


def block(title, value, opened=False, error=False):
    return (f'<details class="{"error" if error else ""}" {"open" if opened else ""}>'
            f'<summary>{esc(title)}</summary><pre>{esc(pretty(value))}</pre></details>')


def is_error(value):
    if not isinstance(value, dict):
        return False
    return bool(value.get('error') or value.get('stderr') or value.get('exit_code', 0))


def summarize(trace, context=None):
    verification = trace.get('verification') or {}
    events = trace.get('events', [])
    context = context or {}
    return {
        'episode_id': trace['episode_id'], 'task_id': trace.get('task_id', ''),
        'phase': context.get('phase', trace.get('split', 'unknown')),
        'optimizer_step': context.get('optimizer_step'),
        'score': (verification.get('reward') if verification.get('diagnostics', {}).get('reward_applied') == 'correctness' else verification.get('diagnostics', {}).get('strict_score', verification.get('reward'))),
        'grading_status': verification.get('status', 'unresolved'),
        'termination': trace.get('termination', 'unknown'),
        'tool_calls': trace.get('usage', {}).get('tool_calls', 0),
        'errors': sum(is_error(e.get('value')) for e in events if e.get('kind') == 'observation'),
    }


STYLE = '''<style>
:root{font:15px system-ui;color:#172b42;background:#f4f7fb}body{max-width:1150px;margin:32px auto;padding:0 20px}h1{font-size:30px}h2{font-size:20px}p{line-height:1.6}.muted{color:#52657d}.episode{background:white;border:1px solid #d7e0ec;border-radius:12px;margin:14px 0;padding:16px}summary{cursor:pointer;padding:8px;overflow-wrap:anywhere}details details{margin:8px 0;border-left:3px solid #a8bdd9;background:#f8fafc}pre{white-space:pre-wrap;overflow-wrap:anywhere;padding:12px;max-height:420px;overflow:auto;font:13px/1.55 ui-monospace,monospace}.error{border-color:#d54444!important;background:#fff0ef!important}.badge{display:inline-block;background:#eaf0f8;padding:5px 9px;margin:4px;border-radius:6px}input,select{padding:10px;border:1px solid #acbdd1;border-radius:6px;margin:5px}input{min-width:280px}a{color:#145db3}nav{position:sticky;top:0;background:#f4f7fb;padding:8px 0;z-index:2}.step{margin-top:20px}.step h3{font-size:16px;margin-bottom:5px}[hidden]{display:none!important}
</style>'''


def episode_html(trace, context=None):
    meta = summarize(trace, context)
    parts = [f'<h2>{esc(meta["task_id"])}</h2>', '<p class="muted">'+esc(meta['episode_id'])+'</p>']
    parts.append(''.join(f'<span class="badge">{esc(k)}: {esc(v if v is not None else "unresolved")}</span>' for k,v in meta.items() if k not in ('episode_id','task_id')))
    review = (context or {}).get('grading_review')
    if review is not None:
        parts.append('<p class="muted">Grader-only assertions, shown after the run. These were not supplied to the agent.</p>')
        if 'scorecard' in review:
            parts.append(block('Strict score vs training-reward diagnostic', review['scorecard'], True))
        parts.append(block('Required assertions and recorded verdicts', review['required_assertions'], True))
        parts.append(block('Additional answer assertions and citation checks', review.get('answer_checks', {})))
    step = 0
    for event in trace.get('events', []):
        kind = event.get('kind')
        if kind == 'initial':
            # The recorder can retain the growing message list; only initial system/user messages belong here.
            messages = []
            for message in event.get('messages', []):
                if message.get('role') == 'assistant':
                    break
                messages.append(message)
            for message in messages:
                parts.append(block('Task / prompt · '+message.get('role', ''), message.get('content', ''), message.get('role') == 'user'))
        elif kind == 'generation':
            step += 1
            parts.append(f'<div class="step"><h3>Step {step}</h3></div>')
            parts.append(block('Model output', event.get('text', ''), True))
        elif kind == 'parsed_action':
            action = event.get('value', {})
            if isinstance(action, dict):
                parts.append(block('Tool call · '+str(action['tool']) if 'tool' in action else 'Final answer submission', action.get('arguments', action), True))
            else:
                parts.append(block('Invalid parsed action', action, True, True))
        elif kind == 'observation':
            value = event.get('value')
            error = is_error(value)
            if isinstance(value, dict) and 'stdout' in value:
                parts.append(block('Tool response', value['stdout']))
                parts.append(block('Response metadata', {k:v for k,v in value.items() if k != 'stdout'}, error, error))
            else:
                parts.append(block('Error / observation' if error else ('Final submission recorded' if isinstance(value, dict) and 'task_id' in value and 'text' in value else 'Tool response / observation'), value, error, error))
    parts.append(block('Final answer and citations', trace.get('submission'), True))
    verification = trace.get('verification') or {}
    if verification.get('diagnostics', {}).get('reward_applied') == 'correctness':
        parts.append(block('Correctness reward and citation diagnostics (citations do not affect reward)', {k:verification['diagnostics'].get(k) for k in ('correctness_score','correctness_status','citation_score','citation_status','citation_errors','strict_score')}, True))
    # Match the existing public grading surface; do not export private grader diagnostics.
    parts.append(block('Grading', {k:verification.get(k) for k in ('status','reward','reasons')}, True))
    parts.append(block('Token usage and timing', trace.get('usage', {})))
    return ''.join(parts)


def load_traces(root):
    root = Path(root)
    contexts = {}
    answers = root/'answers.json'
    if answers.exists():
        table = json.loads(answers.read_text())
        for row in table['data']:
            value = dict(zip(table['columns'], row))
            contexts[value['episode_id']] = value
    traces = [json.loads(p.read_text()) for p in sorted((root/'trajectories').glob('*.json'))]
    return [(t, contexts.get(t['episode_id'], {})) for t in traces]


def render_report(records, title='Agent trajectories'):
    parts = ['<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',
             '<title>'+esc(title)+'</title>', STYLE, '<body><h1>'+esc(title)+'</h1>',
             f'<p>{len(records)} episodes · Expand an episode to inspect its actions, responses and grading. Long responses are collapsed. Scores remain unresolved when no grade exists.</p>',
             '<nav><input id="search" aria-label="Search trajectories" placeholder="Search task, episode, tool or text"><select id="filter" aria-label="Filter trajectories"><option value="all">All episodes</option><option value="failed">Resolved score below 1</option><option value="errors">Tool / action errors</option><option value="budget_exhausted">Budget exhausted</option><option value="unresolved">Unresolved grading</option></select><span id="count"></span></nav>']
    for trace, context in records:
        m = summarize(trace, context)
        flags = [m['termination']]
        if m['errors']: flags.append('errors')
        if m['grading_status'] != 'resolved': flags.append('unresolved')
        elif m['score'] is not None and m['score'] < 1: flags.append('failed')
        label = f"{m['task_id']} · {m['phase']} · step {m['optimizer_step']} · score {m['score']} · {m['tool_calls']} calls · {m['errors']} errors"
        scorecard = context.get('grading_review', {}).get('scorecard') if context else None
        if scorecard:
            diagnostic = scorecard.get('saved_training_reward_diagnostic')
            label += ' · training diagnostic ' + (str(diagnostic) if diagnostic is not None else 'unavailable')
        parts.append(f'<details class="episode" id="{esc(m["episode_id"])}" data-flags="{esc(" ".join(flags))}"><summary>{esc(label)}</summary>'+episode_html(trace, context)+'</details>')
    parts.append('''<script>
const episodes=[...document.querySelectorAll('.episode')];
const search=document.getElementById('search'), filter=document.getElementById('filter');
function update(){let n=0;for(const e of episodes){e.hidden=!(e.textContent.toLowerCase().includes(search.value.toLowerCase())&&(filter.value==='all'||e.dataset.flags.split(' ').includes(filter.value)));if(!e.hidden)n++;}document.getElementById('count').textContent=n+' episodes';}
search.addEventListener('input',update);filter.addEventListener('change',update);update();
if(location.hash){const e=document.getElementById(decodeURIComponent(location.hash.slice(1)));if(e){e.open=true;e.scrollIntoView();}}
</script></body></html>''')
    return ''.join(parts)


def write_report(root, output=None):
    root = Path(root)
    output = Path(output) if output else root/'trajectory-viewer.html'
    records = load_traces(root)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_report(records, root.name), encoding='utf-8')
    return output, records


def tool_call_rows(records):
    """One native W&B row per model turn, with tool inputs and outputs separated."""
    columns = ['episode_id', 'task_id', 'phase', 'optimizer_step', 'score',
               'step', 'tool', 'arguments', 'response', 'error', 'elapsed_seconds', 'model_output']
    rows = []
    for trace, context in records:
        meta = summarize(trace, context)
        turns = []
        current = None
        for event in trace.get('events', []):
            kind = event.get('kind')
            if kind == 'generation':
                current = {'text': event.get('text', ''), 'action': None, 'observation': None}
                turns.append(current)
                try:
                    action = json.loads(current['text'])
                    if isinstance(action, dict): current['action'] = action
                except (ValueError, TypeError): pass
            elif current is not None and kind == 'parsed_action':
                current['action'] = event.get('value')
            elif current is not None and kind == 'observation':
                current['observation'] = event.get('value')
        for step, turn in enumerate(turns, 1):
            action, observation = turn['action'], turn['observation']
            name = action.get('tool', 'final answer') if isinstance(action, dict) else 'invalid action'
            if not isinstance(name, str): name = 'invalid tool name: '+pretty(name)
            arguments = action.get('arguments', action) if isinstance(action, dict) else action
            error = ''
            elapsed = None
            response = observation
            if isinstance(observation, dict):
                elapsed = observation.get('elapsed_seconds')
                if is_error(observation): error = pretty(observation.get('error') or observation.get('stderr') or observation)
                if 'stdout' in observation: response = observation['stdout']
            rows.append([meta[k] for k in columns[:5]] +
                        [step, name, pretty(arguments), pretty(response), error, elapsed, turn['text']])
    return columns, rows


def log_report(remote, wandb, path, records):
    columns = ['episode_id','task_id','phase','optimizer_step','score','grading_status','termination','tool_calls','errors','trace']
    rows = []
    for trace, context in records:
        meta = summarize(trace, context)
        # Individual reports use native details elements and work without JavaScript.
        view = '<!doctype html><meta charset="utf-8">'+STYLE+'<body>'+episode_html(trace, context)+'</body>'
        rows.append([meta[k] for k in columns[:-1]]+[wandb.Html(view, inject=False)])
    tool_columns, tool_rows = tool_call_rows(records)
    remote.log({'tool_calls': wandb.Table(columns=tool_columns, data=tool_rows),
                'trajectories': wandb.Table(columns=columns, data=rows),
                'trajectory_viewer': wandb.Html(Path(path).read_text(), inject=False)})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_root')
    parser.add_argument('--output')
    args = parser.parse_args()
    path, records = write_report(args.run_root, args.output)
    print(f'{path}: {len(records)} episodes')
