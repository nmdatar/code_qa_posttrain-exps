"""Build historical trajectory reports from discovered local archives, without uploads."""
import hashlib
import json
from pathlib import Path
from training_pipeline.trajectory_viewer import load_traces, render_report, STYLE, esc

OUT=Path('reports/trajectory-history')
sources=json.loads((OUT/'sources.json').read_text())
remote=json.loads((OUT/'wandb-runs.json').read_text())
entries=[];seen={};errors=[]
for source in sources:
    root=Path(source['root'])
    try:
        records=load_traces(root)
    except Exception as e:
        errors.append({'root':str(root),'error':str(e)});continue
    fingerprint=hashlib.sha256(json.dumps(records,sort_keys=True).encode()).hexdigest()
    if fingerprint in seen:
        seen[fingerprint]['duplicate_sources'].append(str(root));continue
    ident=source['run_id']+'-'+fingerprint[:8]
    directory=OUT/ident;directory.mkdir(exist_ok=True)
    tracking=[]
    for path in root.glob('tracking*.json'):
        value=json.loads(path.read_text())
        if value.get('id'):tracking.append(value['id'])
    metadata={'run_id':source['run_id'],'episodes':len(records),'source':str(root),
              'duplicate_sources':[],'tracking_ids':tracking,'pages':[], 'fingerprint':fingerprint}
    expected=set()
    if (root/'answers.json').exists():
        table=json.loads((root/'answers.json').read_text())
        expected={dict(zip(table['columns'],row))['episode_id'] for row in table['data']}
    actual={t['episode_id'] for t,c in records}
    metadata['answer_rows_without_trace']=sorted(expected-actual)
    metadata['unique_episodes']=len(actual)
    metadata['snapshot_read_errors']=[]
    if 'trajectory-history-snapshots' in str(root):
        metadata['snapshot_read_errors']=[r for r in json.loads((OUT/'snapshot-download.json').read_text()) if r['status'] != 'saved']
    pages=(len(records)+199)//200
    for number,start in enumerate(range(0,len(records),200),1):
        file=f'page-{number}.html'
        navigation='<p><a href="../index.html">← All historical runs</a> · '+ ' · '.join(f'<a href="page-{p}.html">Page {p}</a>' for p in range(1,pages+1))+'</p>'
        content=render_report(records[start:start+200],source['run_id']+f' · page {number}/{pages}')
        content=content.replace('<body>','<body>'+navigation,1)
        (directory/file).write_text(content)
        metadata['pages'].append(f'{ident}/{file}')
    entries.append(metadata);seen[fingerprint]=metadata
    print(source['run_id'],len(records),flush=True)
matched={v for e in entries for v in e['tracking_ids']}|{e['run_id'] for e in entries}
missing=[r for r in remote if r['id'] not in matched]
manifest={'archives':entries,'missing_wandb_runs':missing,'errors':errors,'wandb_runs':len(remote),'matched_wandb_runs':sum(r['id'] in matched for r in remote)}
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
parts=['<!doctype html><html lang="en"><meta charset="utf-8"><title>Historical agent trajectories</title>',STYLE,'<body><h1>Historical agent trajectories</h1>',f'<p>{len(entries)} distinct archives · {len(set(e["run_id"] for e in entries))} run IDs · saved traces match {manifest["matched_wandb_runs"]}/{len(remote)} W&B runs.</p>', '<p>These are saved snapshots, not live feeds. Archive variants retain their original grades. Large archives use pages of at most 200 episodes; trajectory search applies to the current page.</p>', '<input id="search" aria-label="Find a run" placeholder="Find a run: shell, lr, grpo…">']
for e in sorted(entries,key=lambda e:e['run_id']):
    parts.append('<section class="episode"><h2>'+esc(e['run_id'])+'</h2><p>'+str(e['episodes'])+' episodes · '+ ' · '.join(f'<a href="{esc(p)}">Open page {i}</a>' for i,p in enumerate(e['pages'],1))+'</p>')
    matches=[r for r in remote if r['id'] in e['tracking_ids'] or r['id']==e['run_id']]
    for r in matches:parts.append(f'<p><a href="{esc(r["url"])}">W&B: {esc(r["id"])}</a> · {esc(r["state"])} at inventory time</p>')
    if e.get('snapshot_read_errors'):parts.append('<p class="error">Active-run snapshot: '+str(len(e['snapshot_read_errors']))+' trajectory files were incomplete or unreadable and are not displayed.</p>')
    if e['answer_rows_without_trace']:parts.append('<p class="error">Missing '+str(len(e['answer_rows_without_trace']))+' traces referenced in answers.</p>')
    parts.append('<details><summary>Archive provenance</summary><pre>'+esc(e['source'])+'</pre></details></section>')
parts.append('<h2>Runs without local trajectories</h2>')
for r in missing:parts.append(f'<p><a href="{esc(r["url"])}">{esc(r["id"])}</a> · {esc(r["state"])} · no local archive found</p>')
parts.append('''<script>document.getElementById('search').addEventListener('input',function(){for(const e of document.querySelectorAll('section.episode'))e.hidden=!e.textContent.toLowerCase().includes(this.value.toLowerCase())})</script></body></html>''')
(OUT/'index.html').write_text(''.join(parts))
print(json.dumps({'archives':len(entries),'run_ids':len(set(e['run_id'] for e in entries)),'matched_wandb_runs':manifest['matched_wandb_runs'],'missing':missing,'errors':errors,'missing_trace_counts':sum(len(e['answer_rows_without_trace']) for e in entries)},indent=2))
