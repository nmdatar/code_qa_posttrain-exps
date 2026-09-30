import { useState } from 'react';
import { api, type Repository, type Citation, type ToolStep } from './types';

export function ImportRepository({ onImported, disabled }: { onImported: (repo: Repository) => void; disabled: boolean }) {
  const [open, setOpen] = useState(false), [url, setUrl] = useState(''), [ref, setRef] = useState('');
  const [pending, setPending] = useState(false), [error, setError] = useState('');
  return <div className="repo-import"><button disabled={disabled || pending} onClick={() => setOpen(!open)}>+ Add GitHub repo</button>
    {open && <form onSubmit={async e => { e.preventDefault(); setPending(true); setError('');
      try { const repo = await api<Repository>('/repos', { method: 'POST', body: JSON.stringify({ url: url.trim(), ref: ref.trim() || 'HEAD' }) }); onImported(repo); setOpen(false); }
      catch (e) { setError((e as Error).message); } finally { setPending(false); }
    }}><label>Public GitHub URL<input required value={url} onChange={e => setUrl(e.target.value)} placeholder="https://github.com/owner/repo" disabled={pending} /></label>
      <label>Commit, branch, or tag (optional)<input value={ref} onChange={e => setRef(e.target.value)} placeholder="Commit SHA, main, or v1.0 — default: HEAD" disabled={pending} /></label>
      <p>Leave blank to use the default branch. The repository is pinned to the resolved commit for your investigation.</p>
      <p>Imported repositories support source reading. Private repositories and code execution are not supported.</p>
      <button disabled={pending}>{pending ? 'Importing source…' : 'Import repository'}</button>
      {error && <p role="alert">{error}</p>}
    </form>}
  </div>;
}

export function EvidenceAnswer({ text, citations, openFile }: { text: string; citations: Citation[]; openFile: (path: string, start: number, end: number) => void }) {
  const button = (c: Citation, i: number) => <button key={i} disabled={!c.verified} onClick={() => openFile(c.path,c.start_line,c.end_line)} title="Opens the exact source range; support is model-attributed, not independently verified">{c.path}:{c.start_line}–{c.end_line} · {c.verified ? 'Source read' : 'Not observed'}</button>;
  const paragraphs = text.split(/\n\s*\n/);
  const linked = new Set(citations.filter(c => c.claim && paragraphs.some(p => p.includes(c.claim!))));
  return <><p className="evidence-note">Source read means these lines were available to the agent. Claim support is attributed by the model, not independently verified.</p>
    {paragraphs.map((p,i) => <div key={i}><p>{p}</p><div className="citations">{citations.filter(c => linked.has(c) && p.includes(c.claim!)).map(button)}</div></div>)}
    {citations.some(c => !linked.has(c)) && <><small>Additional sources · no claim mapping provided</small><div className="citations">{citations.filter(c => !linked.has(c)).map(button)}</div></>}
  </>;
}

export function InvestigationMap({ steps, openFile }: { steps: ToolStep[]; openFile: (path: string, start?: number, end?: number) => void }) {
  const files = new Map<string, { actions: string[]; line: number }>();
  for (const step of steps) {
    const name = step.call.name || '', args = step.call.arguments || {};
    const targets: { path: string; line: number }[] = [];
    if (name === 'read_file' && typeof args.path === 'string') targets.push({ path: args.path, line: Number(args.start_line || 1) });
    if (name === 'search_code') for (const match of (step.result?.observation?.content?.matches || []) as {path: string; line: number}[]) targets.push(match);
    for (const target of targets) {
      const item = files.get(target.path) || { actions: [], line: target.line };
      const label = name === 'read_file' ? (step.result?.observation?.status === 'ok' ? 'Read' : 'Read attempted') : 'Search match';
      if (!item.actions.includes(label)) item.actions.push(label);
      files.set(target.path,item);
    }
  }
  return <section className="investigation-map" aria-label="Investigation map"><h3>Investigation map</h3><p>Files encountered during this run · select a file to inspect its source.</p>
    <div className="map-files">{[...files].map(([path,item],i) => <button key={path} onClick={() => openFile(path,item.line,item.line)}><small>{i+1} · {item.actions.join(' → ')}</small><strong>{path}</strong></button>)}</div>
    {!files.size && <p>No source files encountered yet.</p>}
  </section>;
}
