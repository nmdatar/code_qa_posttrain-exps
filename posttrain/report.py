"""Self-contained, escaped run report with no network dependencies."""
from __future__ import annotations
import html
import json
import math
from pathlib import Path
from urllib.parse import quote
from .tracking import EventLog


def _numeric_metrics(event):
    result = {}
    def visit(value, prefix=''):
        for key, item in value.items():
            label = prefix + key
            if isinstance(item, (int, float)) and not isinstance(item, bool) and math.isfinite(item):
                result[label] = item
            elif isinstance(item, dict):
                visit(item, label + '/')
    visit(event)
    return result


def _plot(title, points):
    xmin, xmax = min(x for x, _ in points), max(x for x, _ in points)
    ymin, ymax = min(y for _, y in points), max(y for _, y in points)
    coords = [(45 + (x-xmin)/(xmax-xmin or 1)*510, 175-(y-ymin)/(ymax-ymin or 1)*145) for x,y in points]
    line = ' '.join(f'{x:.2f},{y:.2f}' for x,y in coords)
    circles = ''.join(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3"/>' for x,y in coords)
    return (f'<section><h2>{html.escape(title)}</h2><svg viewBox="0 0 600 220" role="img" aria-label="{html.escape(title, quote=True)}">'
            f'<path d="M45 20V175H565" stroke="#777" fill="none"/>'
            f'<polyline points="{line}" stroke="#3868b2" fill="none"/>{circles}'
            f'<text x="5" y="22">{ymax:.4g}</text><text x="5" y="175">{ymin:.4g}</text>'
            f'<text x="45" y="198">{xmin:g}</text><text x="535" y="198">{xmax:g}</text>'
            '<text x="210" y="218">Optimizer update (or event ID)</text></svg></section>')


def create_report(run_dir, output=None):
    directory = Path(run_dir).resolve()
    events = EventLog(directory).read()
    destination = Path(output) if output else directory / 'report.html'
    series = {}
    rows = []
    for event in events:
        metrics = _numeric_metrics(event)
        step = event.get('optimizer_step', event.get('optimizer_updates', event.get('step', event['event_id'])))
        for name, value in metrics.items():
            if any(word in name.lower() for word in ('loss', 'reward', 'accuracy', 'coverage', 'accepted', 'latency', 'cost', 'citation', 'error', 'completion')):
                series.setdefault(name, []).append((float(step), value))
        links = []
        for key, value in event.items():
            if isinstance(value, str) and (key.endswith('_path') or key in {'artifact', 'trajectory'}):
                candidate = Path(value)
                candidate = candidate if candidate.is_absolute() else candidate.resolve() if candidate.exists() else directory / candidate
                candidate = candidate.resolve()
                # Only link existing files contained in this run; no remote/javascript URLs.
                if candidate.is_relative_to(directory) and candidate.is_file():
                    links.append(f'<a href="{quote(str(candidate))}">{html.escape(key)}</a>')
        rows.append('<tr><td>'+str(event['event_id'])+'</td><td>'+html.escape(event['kind'])+'</td><td>'+
                    html.escape(str(event.get('status', event.get('termination', ''))))+'</td><td>'+
                    ' '.join(links)+'</td><td><details><summary>Fields</summary><pre>'+
                    html.escape(json.dumps(event, indent=2))+'</pre></details></td></tr>')
    page = ('<!doctype html><meta charset="utf-8"><title>Post-training run</title><style>'
            'body{font:15px system-ui;max-width:1200px;margin:30px auto;padding:20px;color:#172330}'
            '.charts{display:grid;grid-template-columns:repeat(auto-fit,minmax(380px,1fr));gap:20px}'
            'svg{width:100%;font-size:11px}table{border-collapse:collapse;width:100%}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:left}'
            'pre{white-space:pre-wrap;overflow-wrap:anywhere}section{border:1px solid #ddd;padding:12px;border-radius:8px}h2{font-size:16px}</style>'
            f'<h1>Post-training run: {html.escape(directory.name)}</h1><p>{len(events)} durable events. '
            'Policy loss is not accuracy. Unresolved evaluations must remain visible in scoring coverage. '
            'This local report may contain private run details; review before sharing.</p>'
            '<div class="charts">'+''.join(_plot(name, points) for name, points in sorted(series.items()))+'</div>'
            '<h2>Events and rollout artifacts</h2><table><tr><th>ID</th><th>Event</th><th>Status</th><th>Artifacts</th><th>Details</th></tr>'+
            ''.join(rows)+'</table>')
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(page)
    return destination


render_report = create_report
