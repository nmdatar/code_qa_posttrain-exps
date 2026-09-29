"""Dependency-free, offline evaluation report rendering."""

from html import escape
from pathlib import Path
import math


DIMENSIONS = ("correctness", "completeness", "relevance", "clarity", "reasoning")


def _text(value):
    return escape(str(value if value is not None else "—"), quote=True)


def _number(value, places=1):
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        return f"{value:,.{places}f}"
    return "—"


def _score(row):
    scores = row.get("scores") or {}
    values = [scores.get(key) for key in DIMENSIONS]
    if row.get("status") != "success" or not all(
        isinstance(v, (int, float)) and not isinstance(v, bool)
        and math.isfinite(v) and 1 <= v <= 10 for v in values
    ):
        return None
    return sum(values)


def write_report(run_dir: Path, config: dict, rows: list[dict], summary: dict):
    """Write an offline report; all dataset, model and judge text is untrusted."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    demo = config.get("mode") == "demo"
    banner = (
        '<aside class="banner"><strong>DEMO · Synthetic pipeline check</strong>'
        'These answers and scores demonstrate the reporting pipeline. '
        'They do not measure model quality and must not be compared with real evaluation runs.</aside>'
        if demo else ''
    )
    metrics = [
        ("Mean rubric score", _number(summary.get("mean_score")), "/ 50 · scored questions only"),
        ("Answers completed", f'{_number(summary.get("answered"), 0)} / {_number(summary.get("expected"), 0)}', "coverage of expected questions"),
        ("Questions scored", f'{_number(summary.get("scored"), 0)} / {_number(summary.get("expected"), 0)}', "inspect missing scores separately"),
        ("Failures", _number(summary.get("failed"), 0), "agent or judge errors"),
        ("Mean latency", _number(summary.get("mean_latency_seconds")), "seconds per measured question"),
        ("Total tokens", _number(summary.get("total_tokens"), 0), "— means usage unavailable"),
    ]
    cards = ''.join(f'<div class="metric"><span>{label}</span><strong>{value}</strong><small>{note}</small></div>' for label, value, note in metrics)
    metadata = ''.join(
        f'<div><dt>{label}</dt><dd>{_text(config.get(key))}</dd></div>'
        for label, key in (("Mode", "mode"), ("Answer model", "model"), ("Judge model", "judge_model"), ("Dataset revision", "dataset_revision"))
    )
    dimension_cards = []
    for dimension in DIMENSIONS:
        values = [(row.get("scores") or {}).get(dimension) for row in rows if _score(row) is not None]
        mean = sum(values) / len(values) if values else None
        dimension_cards.append(
            f'<div class="dimension"><span>{dimension.title()}</span><strong>{_number(mean)} <small>/ 10</small></strong>'
            f'<meter min="0" max="10" value="{mean or 0}" aria-label="{dimension.title()} mean"></meter></div>'
        )
    questions = []
    for index, row in enumerate(rows, start=1):
        total = _score(row)
        status = row.get("status") or "unknown"
        status_class = "success" if status == "success" else "error"
        status_label = {"success": "Scored" if total is not None else "Unscored", "agent_error": "Agent error", "judge_error": "Judge error"}.get(status, "Unknown status")
        error = f'<div class="error-message"><strong>Error:</strong> {_text(row.get("error"))}</div>' if row.get("error") else ''
        score_cells = ''.join(
            f'<div><dt>{dimension.title()}</dt><dd>{_number((row.get("scores") or {}).get(dimension))} / 10</dd></div>'
            for dimension in DIMENSIONS
        )
        questions.append(f'''
<article class="question" data-status="{_text(status)}" data-score="{total if total is not None else -1}" data-index="{index}">
  <details>
    <summary><span class="question-info"><span class="eyebrow">{index:02d} · {_text(row.get("repo"))}</span><span class="question-title">{_text(row.get("question"))}</span></span><span class="result"><span class="badge {status_class}">{status_label}</span><strong>{_number(total)} <small>/ 50</small></strong></span></summary>
    <div class="question-body"><p class="identifier">Question ID: {_text(row.get("id"))}</p>{error}
      <dl class="score-grid">{score_cells}</dl>
      <div class="judge"><h3>Judge explanation</h3><pre>{_text(row.get("judge_reason"))}</pre></div>
      <div class="answer-grid"><section><h3>Agent answer</h3><pre>{_text(row.get("answer"))}</pre></section><section><h3>Reference answer</h3><pre>{_text(row.get("reference_answer"))}</pre></section></div>
      <p class="usage">Latency: {_number(row.get("latency_seconds"))} s · Input tokens: {_number(row.get("input_tokens"), 0)} · Output tokens: {_number(row.get("output_tokens"), 0)} · Tool calls: {_number(row.get("tool_calls"), 0)}</p>
    </div>
  </details>
</article>''')
    completion = summary.get("completion_rate")
    scoring = summary.get("scoring_rate")
    completion_pct = _number(completion * 100) if isinstance(completion, (int, float)) else "—"
    scoring_pct = _number(scoring * 100) if isinstance(scoring, (int, float)) else "—"
    document = '''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Coding Q&amp;A · Evaluation report</title>
<style>
:root{color-scheme:light;--ink:#172b3a;--muted:#586b79;--line:#dce5e9;--accent:#08786d}*{box-sizing:border-box}body{margin:0;background:#f3f6f7;color:var(--ink);font:15px/1.55 system-ui,-apple-system,sans-serif}main{max-width:1240px;margin:auto;padding:40px 28px 64px}h1{font-size:clamp(28px,4vw,42px);letter-spacing:-1.5px;margin:8px 0}h2{font-size:22px;letter-spacing:-.5px;margin:32px 0 12px}h3{font-size:14px;margin:0 0 10px}p{margin:8px 0}.eyebrow{color:var(--muted);font-size:12px;letter-spacing:.6px;display:block}header .eyebrow{color:var(--accent);font-weight:700;letter-spacing:2px}.subtitle{color:var(--muted)}.banner{background:#fff0ca;border:1px solid #e4c87d;padding:16px 20px;border-radius:12px;margin:24px 0}.banner strong{display:block;color:#775411;margin-bottom:4px}.metadata{display:flex;gap:20px 40px;flex-wrap:wrap;margin:24px 0}.metadata div{min-width:100px}.metadata dt,small,.identifier,.usage{color:var(--muted);font-size:12px}.metadata dd{margin:2px 0 0;overflow-wrap:anywhere;max-width:600px}.metrics{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}.metric,.dimension{background:white;border:1px solid var(--line);padding:18px 20px;border-radius:12px}.metric span,.metric small{display:block}.metric span{color:var(--muted);font-size:13px}.metric strong{display:block;font-size:30px;letter-spacing:-1px;margin:6px 0}.dimensions{display:grid;grid-template-columns:repeat(5,1fr);gap:10px}.dimension{padding:14px}.dimension>span{font-size:12px;color:var(--muted);display:block}.dimension strong{display:block;font-size:22px;margin:6px 0}meter{width:100%;height:8px;accent-color:var(--accent)}.interpretation{border-left:3px solid var(--accent);padding:4px 18px;margin:24px 0;color:var(--muted)}.interpretation strong{color:var(--ink)}.toolbar{display:flex;gap:10px;flex-wrap:wrap;margin:16px 0}input,select,button{font:inherit;border:1px solid #bbcbd3;background:white;color:var(--ink);border-radius:8px;padding:10px 12px}input{flex:1;min-width:200px}button{cursor:pointer}button:hover{background:#e8f4f0}:focus-visible{outline:3px solid #2b998d;outline-offset:3px}label{display:flex;align-items:center;gap:8px;font-size:13px}.question{background:white;border:1px solid var(--line);border-radius:12px;margin:10px 0;overflow:hidden}.question[hidden]{display:none}summary{padding:20px;cursor:pointer;display:flex;gap:20px;align-items:center;justify-content:space-between}summary::before{content:'+';color:var(--accent);font-size:24px}details[open]>summary::before{content:'−'}.question-info{flex:1;min-width:0}.question-title{display:block;margin-top:5px;font-weight:600;overflow-wrap:anywhere}.result{display:flex;align-items:flex-end;flex-direction:column;gap:8px;flex-shrink:0}.result strong{font-size:22px}.badge{border-radius:20px;font-size:11px;font-weight:600;padding:3px 9px}.success{background:#e0f3ed;color:#16604c}.error{background:#fce8e5;color:#963f32}.question-body{border-top:1px solid var(--line);padding:20px}.identifier{overflow-wrap:anywhere;margin:0 0 16px}.score-grid{display:grid;grid-template-columns:repeat(5,1fr);gap:8px;background:#f3f6f7;border-radius:8px;padding:14px;margin:14px 0}.score-grid dt{font-size:12px;color:var(--muted)}.score-grid dd{margin:4px 0 0;font-weight:650}.answer-grid{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin:16px 0}.answer-grid section,.judge{border:1px solid var(--line);padding:16px;border-radius:8px;min-width:0}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:13px/1.7 ui-monospace,SFMono-Regular,Consolas,monospace;margin:0}.error-message{background:#fff0ed;padding:12px;border-radius:8px;white-space:pre-wrap;overflow-wrap:anywhere}.usage{margin-top:18px}#empty{padding:28px;text-align:center;color:var(--muted)}footer{margin-top:32px;font-size:12px;color:var(--muted)}@media(max-width:800px){.metrics{grid-template-columns:repeat(2,1fr)}.dimensions{grid-template-columns:repeat(3,1fr)}.answer-grid{grid-template-columns:1fr}.score-grid{grid-template-columns:repeat(3,1fr)}}@media(max-width:480px){main{padding:24px 14px}.dimensions{grid-template-columns:repeat(2,1fr)}summary{padding:14px;gap:10px}.metric{padding:14px}.metric strong{font-size:24px}}@media print{.toolbar{display:none}body{background:white}main{max-width:none;padding:0}.question{break-inside:avoid}}
</style></head><body><main>
<header><span class="eyebrow">CODING Q&amp;A / EVALUATION</span><h1>Inspect the answers behind the score.</h1><p class="subtitle">Run quality, coverage, and individual judgments in one place.</p></header>
''' + banner + '<dl class="metadata">' + metadata + '</dl><section class="metrics" aria-label="Run metrics">' + cards + '</section>' + f'''
<div class="interpretation"><p><strong>Read quality and coverage together.</strong> The rubric total adds five 1–10 dimensions (range 5–50); it is not a percent-correct score. Means include only fully scored questions. Agent and judge failures can make the mean look better by excluding difficult questions.</p><p>Answer completion: <strong>{completion_pct}%</strong>. Scoring coverage: <strong>{scoring_pct}%</strong>. Compare runs on the same question IDs, repository snapshot, rubric, and judge model. Review low scores and failures, then spot-check strong answers against the repository. Judge scores are estimates, not proof of correctness.</p></div>
<h2>Where answers succeed or struggle</h2><section class="dimensions" aria-label="Mean scores by dimension">{''.join(dimension_cards)}</section>
<h2>Question explorer</h2><p class="subtitle">Open a question to compare the answer with its reference and read the judge’s reasoning.</p>
<div class="toolbar"><input id="search" type="search" aria-label="Search questions, answers, repositories, and errors" placeholder="Search questions, answers, repositories…"><label>Status <select id="status"><option value="all">All questions</option><option value="success">Successful</option><option value="agent_error">Agent errors</option><option value="judge_error">Judge errors</option><option value="unscored">Unscored</option></select></label><label>Order <select id="sort"><option value="original">Dataset order</option><option value="low">Lowest score first</option><option value="high">Highest score first</option></select></label><button id="expand" type="button">Expand visible</button><button id="collapse" type="button">Collapse all</button></div>
<p id="count" class="subtitle" role="status" aria-live="polite">{len(rows)} questions</p><section id="questions">{''.join(questions)}</section><p id="empty" hidden>No questions match the current filters.</p>
<footer>Self-contained local report. Search and filters run in your browser; no external scripts or network requests. Reference answers are included in this file.</footer>
<noscript><p>JavaScript is disabled. All questions remain available; open them individually to inspect answers.</p></noscript>
</main><script>
const cards = Array.from(document.querySelectorAll('.question'));
const search = document.getElementById('search');
const status = document.getElementById('status');
const order = document.getElementById('sort');
function update() {{
  const term = search.value.toLocaleLowerCase().trim();
  let visible = 0;
  for (const card of cards) {{
    const matchesStatus = status.value === 'all' || (status.value === 'unscored' ? Number(card.dataset.score) < 0 : card.dataset.status === status.value);
    card.hidden = !matchesStatus || !card.textContent.toLocaleLowerCase().includes(term);
    if (!card.hidden) visible++;
  }}
  const sorted = [...cards].sort((a, b) => {{
    if (order.value === 'original') return Number(a.dataset.index) - Number(b.dataset.index);
    const av = Number(a.dataset.score), bv = Number(b.dataset.score);
    if (av < 0 || bv < 0) return av < 0 && bv < 0 ? 0 : av < 0 ? -1 : 1;
    return order.value === 'low' ? av - bv : bv - av;
  }});
  const container = document.getElementById('questions');
  for (const card of sorted) container.appendChild(card);
  document.getElementById('count').textContent = `${{visible}} of ${{cards.length}} questions shown · failures appear first when sorting by score`;
  document.getElementById('empty').hidden = visible !== 0;
}}
search.addEventListener('input', update);
status.addEventListener('change', update);
order.addEventListener('change', update);
document.getElementById('expand').addEventListener('click', () => cards.filter(c => !c.hidden).forEach(c => c.querySelector('details').open = true));
document.getElementById('collapse').addEventListener('click', () => cards.forEach(c => c.querySelector('details').open = false));
update();
</script></body></html>'''
    path = run_dir / "report.html"
    path.write_text(document, encoding="utf-8")
    return path
