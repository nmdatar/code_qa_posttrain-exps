import HarnessSelect from './HarnessSelect';
import ModelSelect from './ModelSelect';
import { newestCheckpointsFirst } from './modelOrdering';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { ArrowUp, GitBranch, LoaderCircle, Play, RefreshCw, Square, X } from 'lucide-react';
import { api } from './types';
import type { Activity, AssertionGrade, Benchmark, Comparison, Model, Repository, Run, Selection, ToolStep } from './types';
import { StepCard } from './App';
import Inspector from './Inspector';
import { EvidenceAnswer, InvestigationMap, ImportRepository } from './ResearchFeatures';
import RunMetrics, { costLabel } from './RunMetrics';
import ComparisonScore, { scoreLabel } from './ComparisonScore';

type Evidence = { run: Run; selection: Selection; step?: ToolStep };
const status: Record<string, string> = { running: 'Investigating', completed: 'Complete', cancelled: 'Stopped', interrupted: 'Interrupted', infrastructure_error: 'Run failed', budget_exhausted: 'Budget reached', agent_error: 'Invalid model action' };

function RunColumn({ initial, previous, side, onUpdate, onInspect, onError, onActivity, grade, gradingStatus }: {
  initial: Run; previous: Run[]; side: number; onUpdate: (run: Run) => void;
  onInspect: (evidence: Evidence) => void; onError: (error: string) => void;
  onActivity: (run: Run, steps: ToolStep[]) => void; grade?: AssertionGrade; gradingStatus?: string;
}) {
  const [run, setRun] = useState(initial);
  const [events, setEvents] = useState<Activity[]>([]);
  const [connected, setConnected] = useState(true);
  const [stopping, setStopping] = useState(false);
  useEffect(() => {
    let disposed = false;
    const stream = new EventSource(`/api/runs/${initial.id}/events`);
    stream.onopen = () => setConnected(true);
    stream.onerror = () => setConnected(false);
    stream.onmessage = e => {
      if (disposed) return;
      const event: Activity = JSON.parse(e.data);
      setEvents(old => old.some(x => x.sequence === event.sequence) ? old : [...old, event].sort((a, b) => a.sequence - b.sequence));
    };
    stream.addEventListener('metrics', e => {
      if (!disposed) setRun(old => ({ ...old, metrics: JSON.parse((e as MessageEvent).data) }));
    });
    stream.addEventListener('terminal', e => {
      if (disposed) return;
      const value: Run = JSON.parse((e as MessageEvent).data);
      setRun(value); onUpdate(value); setConnected(true); stream.close();
    });
    return () => { disposed = true; stream.close(); };
  }, [initial.id, onUpdate]);
  const steps = useMemo(() => {
    const results = new Map(events.filter(e => e.kind === 'tool_observation').map(e => [e.call_id, e]));
    return events.filter(e => e.kind === 'tool_call').map(call => ({ call, result: results.get(call.call_id) }));
  }, [events]);
  useEffect(() => onActivity(run, steps), [run, steps, onActivity]);
  function openFile(path: string, start = 1, end = start) {
    onInspect({ run, selection: { type: 'file', path, start, end } });
  }
  const active = run.status === 'running';
  const last = events.at(-1);
  return <section className={`compare-column side-${side}`} aria-label={`Model ${side ? 'B' : 'A'} run`}>
    <header className="compare-column-header">
      <div><span className="model-letter">{side ? 'B' : 'A'}</span><h2>{run.model.name}</h2></div>
      <span className={active ? 'live-status' : ''}>{active && <LoaderCircle size={13} className="spin" />}{status[run.status] || run.status}</span>
      {active && <button className="stop-button" disabled={stopping} onClick={async () => {
        setStopping(true);
        try { await api(`/runs/${run.id}/cancel`, { method: 'POST' }); }
        catch (e) { onError((e as Error).message); setStopping(false); }
      }}><Square size={10} />{stopping ? 'Stopping…' : `Stop ${side ? 'B' : 'A'}`}</button>}
    </header>
    {gradingStatus && <div className="column-score"><span>Assertion score</span><strong>{scoreLabel(grade,gradingStatus)}</strong></div>}
    <RunMetrics metrics={run.metrics} />
    <div className="compare-column-scroll">
      {previous.map(prior => <details className="previous-turn" key={prior.id}>
        <summary>{prior.question}</summary>
        {prior.answer ? <EvidenceAnswer text={prior.answer.text} citations={prior.answer.citations} openFile={(path, start, end) => onInspect({ run: prior, selection: { type: 'file', path, start, end } })} /> : <p>No answer in this turn.</p>}
      </details>)}
      <div className="timeline-header"><span>Independent investigation</span><small>{run.repo.commit.slice(0, 7)}</small></div>
      <InvestigationMap steps={steps} openFile={openFile} />
      <div className="timeline">
        {events.filter(e => e.kind === 'submission_rejected' || e.kind === 'action_rejected').map(e => <div className="notice" key={e.sequence}><strong>Correcting response</strong><p>{e.feedback}</p></div>)}
        {steps.map(step => <StepCard key={step.call.call_id} step={step} selected={false} stopped={!active}
          onSelect={() => onInspect({ run, selection: { type: 'tool', id: step.call.call_id! }, step })} />)}
        {active && <div className="working" role="status"><LoaderCircle size={14} className="spin" />
          {!connected ? 'Reconnecting to activity…' : last?.kind === 'model_request' ? 'Waiting for model response…' : steps.at(-1) && !steps.at(-1)?.result ? 'Running tool…' : 'Preparing next action…'}
        </div>}
      </div>
      {run.answer && <section className="answer-card"><div className="answer-heading">{run.preview ? 'Scripted preview answer' : 'Answer'}</div><EvidenceAnswer text={run.answer.text} citations={run.answer.citations} openFile={openFile} /></section>}
      {!active && !run.answer && <div className="notice error"><strong>{status[run.status]}</strong><p>{run.error || 'No final answer. Completed activity is preserved.'}</p></div>}
    </div>
  </section>;
}

function ModelPicker({ label, value, models, disabled, onChange }: { label: string; value: string; models: Model[]; disabled: boolean; onChange: (value: string) => void }) {
  const model = models.find(m => m.id === value);
  return <div className="compare-picker"><span>{label}</span><ModelSelect label={label} value={value} models={models} disabled={disabled} onChange={onChange} /><small title={model?.reason || model?.base_model}>{model?.ready === false ? model.reason : model?.parameters ? `${(model.parameters / 1e9).toLocaleString()}B total${model.active_parameters && model.active_parameters !== model.parameters ? ` · ${(model.active_parameters / 1e9).toLocaleString()}B active` : ''}` : model?.base_model || 'Choose a hosted model or checkpoint'}</small></div>;
}

export default function Compare() {
  const [repos, setRepos] = useState<Repository[]>([]), [models, setModels] = useState<Model[]>([]);
  const [leftHarness, setLeftHarness] = useState('auto'), [rightHarness, setRightHarness] = useState('auto');
  const [repoId, setRepoId] = useState(''), [left, setLeft] = useState('base'), [right, setRight] = useState('');
  const [pair, setPair] = useState<Comparison | null>(null), [history, setHistory] = useState<Comparison[]>([]), [ancestors, setAncestors] = useState<Comparison[]>([]);
  const [benchmarks, setBenchmarks] = useState<Benchmark[]>([]), [benchmarkId, setBenchmarkId] = useState(''), [questionMode, setQuestionMode] = useState('free');
  const [question, setQuestion] = useState(''), [error, setError] = useState(''), [warning, setWarning] = useState('');
  const [busy, setBusy] = useState(false), [loadingModels, setLoadingModels] = useState(true), [historyOpen, setHistoryOpen] = useState(false);
  const navigationVersion = useRef(0);
  const [evidence, setEvidence] = useState<Evidence | null>(null);
  const active = (pair?.runs.some(r => r.status === 'running') ?? false) || ['pending','running'].includes(pair?.grading?.status || '');
  const selectedBenchmark = benchmarks.find(b => b.id === benchmarkId) || benchmarks.find(b => b.id === 'pydantic-frozen-copy-8960') || benchmarks[0];
  const choosingBenchmark = !pair && questionMode === 'benchmark';
  const inputQuestion = choosingBenchmark ? selectedBenchmark?.question || '' : question;
  const repo = repos.find(r => r.id === repoId);
  const selectPair = useCallback((value: Comparison) => {
    navigationVersion.current += 1;
    setPair(value); setRepoId(value.repo.id); setQuestion(''); setEvidence(null); setHistoryOpen(false);
    if (!value.preview) { setLeft(value.runs[0].model.id); setRight(value.runs[1].model.id); }
    window.history.replaceState(null, '', '?comparison=' + value.id);
  }, []);
  function newComparison() {
    navigationVersion.current += 1;
    setPair(null); setAncestors([]); setEvidence(null); setQuestion('');
    setQuestionMode('free'); setBenchmarkId(''); setError(''); setHistoryOpen(false);
    window.history.replaceState(null, '', '?mode=compare');
  }
  useEffect(() => {
    const version = navigationVersion.current;
    const controller = new AbortController();
    const options = { signal: controller.signal };
    const fail = (e: Error) => { if (e.name !== 'AbortError') setError(e.message); };
    api<Repository[]>('/repos', options).then(rows => { setRepos(rows); setRepoId(old => old || rows.find(r => r.ready)?.id || ''); }).catch(fail);
    api<{ models: Model[]; warning?: string }>('/models', options).then(value => {
      setModels(newestCheckpointsFirst(value.models)); setWarning(value.warning || '');
      setRight(old => old || value.models.find(m => m.base_model === 'Qwen/Qwen3.5-397B-A17B' && m.ready)?.id || value.models.find(m => m.id !== 'base' && m.ready)?.id || '');
    }).catch(fail).finally(() => { if (!controller.signal.aborted) setLoadingModels(false); });
    api<Comparison[]>('/comparisons', options).then(rows => {
      setHistory(rows);
      const id = new URLSearchParams(location.search).get('comparison');
      if (id) api<Comparison>('/comparisons/' + id, options).then(value => {
        if (!controller.signal.aborted && navigationVersion.current === version) selectPair(value);
      }).catch(fail);
    }).catch(fail);
    return () => controller.abort();
  }, [selectPair]);
  useEffect(() => {
    const controller = new AbortController();
    setAncestors([]);
    async function load() {
      const rows: Comparison[] = [], seen = new Set<string>();
      let id = pair?.parent_comparison_id;
      while (id && !seen.has(id)) {
        seen.add(id);
        const value = await api<Comparison>('/comparisons/' + id, { signal: controller.signal });
        rows.unshift(value); id = value.parent_comparison_id;
      }
      if (!controller.signal.aborted) setAncestors(rows);
    }
    load().catch(e => { if (e.name !== 'AbortError') setError(e.message); });
    return () => controller.abort();
  }, [pair?.id, pair?.parent_comparison_id]);
  useEffect(() => {
    if (!repoId) return;
    const controller = new AbortController();
    setBenchmarks([]);
    api<Benchmark[]>('/benchmarks?repo_id=' + encodeURIComponent(repoId), { signal: controller.signal })
      .then(setBenchmarks).catch(e => { if (e.name !== 'AbortError') setError(e.message); });
    return () => controller.abort();
  }, [repoId]);
  useEffect(() => {
    if (!pair?.benchmark || !['pending','running'].includes(pair.grading?.status || '')) return;
    const id = pair.id, controller = new AbortController();
    let timer: ReturnType<typeof setTimeout>;
    async function poll() {
      try {
        const value = await api<Comparison>('/comparisons/' + id, { signal: controller.signal });
        if (controller.signal.aborted) return;
        setPair(old => old?.id === id ? value : old);
        if (['pending','running'].includes(value.grading?.status || '')) timer = setTimeout(poll, 1500);
      } catch (e) {
        if (!controller.signal.aborted) { setError((e as Error).message); timer = setTimeout(poll, 3000); }
      }
    }
    poll();
    return () => { controller.abort(); clearTimeout(timer); };
  }, [pair?.id, pair?.benchmark?.id, pair?.grading?.status]);
  const updateRun = useCallback((run: Run) => {
    setPair(old => old && old.runs.some(r => r.id === run.id) ? { ...old, runs: old.runs.map(r => r.id === run.id ? run : r) } : old);
  }, []);
  const updateEvidence = useCallback((run: Run, steps: ToolStep[]) => {
    setEvidence(old => {
      if (!old || old.run.id !== run.id || old.selection?.type !== 'tool') return old;
      const id = old.selection.id;
      const step = steps.find(s => s.call.call_id === id);
      return old.step === step && old.run === run ? old : { ...old, step, run };
    });
  }, []);
  async function start(preview = false) {
    if (busy || active) return;
    setBusy(true); setError('');
    const synthetic = pair?.preview || preview;
    try {
      const value = await api<Comparison>('/comparisons', { method: 'POST', body: JSON.stringify({
        left_harness: pair?.runs[0].harness || leftHarness, right_harness: pair?.runs[1].harness || rightHarness,
        repo_id: repoId, left_model_id: synthetic ? 'preview-left' : left, right_model_id: synthetic ? 'preview-right' : right,
        question: synthetic ? question.trim() || repo?.example : inputQuestion.trim(), preview: synthetic, parent_comparison_id: pair?.id,
        benchmark_id: !synthetic && choosingBenchmark ? selectedBenchmark?.id : undefined,
      }) });
      selectPair(value); setHistory(old => [value, ...old]);
    } catch (e) { setError((e as Error).message); }
    finally { setBusy(false); }
  }
  async function refresh() {
    setLoadingModels(true); setError('');
    try { const value = await api<{ models: Model[]; warning?: string }>('/models?refresh=true'); setModels(newestCheckpointsFirst(value.models)); setWarning(value.warning || ''); }
    catch (e) { setError((e as Error).message); }
    finally { setLoadingModels(false); }
  }
  const ready = !pair?.benchmark && (!choosingBenchmark || !!selectedBenchmark) && repo?.ready && (pair?.preview || (left !== right && models.find(m => m.id === left)?.ready && models.find(m => m.id === right)?.ready));
  return <div className="compare-app">
    <header className="app-header"><a className="brand" href="/" aria-label="action-trace home"><span className="brand-mark"><GitBranch size={20} /></span>action-trace<span className="brand-divider" /><span className="brand-sub">Model comparison</span></a>
      <div className="header-actions"><button className="text-button" disabled={busy} onClick={newComparison}>New comparison</button><nav className="mode-switch" aria-label="Run mode"><a href="/">Single</a><span aria-current="page">Compare</span></nav>
      <button className="text-button" onClick={() => setHistoryOpen(!historyOpen)}>Recent comparisons</button></div>
      {historyOpen && <div className="history-menu"><strong>Recent comparisons</strong>{history.length ? history.map(h => <button key={h.id} onClick={() => selectPair(h)}><span>{h.question}</span><small>{h.runs.map(r => r.model.name).join(' vs ')}</small></button>) : <p>No comparisons yet.</p>}</div>}
    </header>
    <main className="compare-main">
      <div className="compare-intro"><div><div className="eyebrow">SAME QUESTION. INDEPENDENT INVESTIGATIONS.</div><h1>Compare how models investigate.</h1><p>Compare answers, tool calls, speed, and token cost on the same pinned repository.</p></div>
      </div>
      <div className="compare-controls">
        <label className="compare-picker"><span>Repository</span><select aria-label="Repository" disabled={!!pair || busy} value={repoId} onChange={e => setRepoId(e.target.value)}>{repos.map(r => <option key={r.id} value={r.id} disabled={!r.ready}>{r.name} · {r.commit.slice(0, 7)}</option>)}</select><small>Pinned source · {repo?.execution ? 'isolated execution' : 'source reading'}</small></label>
        <ModelPicker label="Model A" models={pair?.preview ? pair.runs.map(r => ({...r.model, ready: true})) : models} value={pair?.preview ? pair.runs[0].model.id : left} disabled={!!pair || loadingModels || busy} onChange={id=>{setLeft(id);setLeftHarness('auto')}} />
        <ModelPicker label="Model B" models={pair?.preview ? pair.runs.map(r => ({...r.model, ready: true})) : models} value={pair?.preview ? pair.runs[1].model.id : right} disabled={!!pair || loadingModels || busy} onChange={id=>{setRight(id);setRightHarness('auto')}} />
        <HarnessSelect label="Tools A" value={pair?.runs[0].harness || leftHarness} onChange={setLeftHarness} model={models.find(m=>m.id===left)} disabled={!!pair || busy} />
        <HarnessSelect label="Tools B" value={pair?.runs[1].harness || rightHarness} onChange={setRightHarness} model={models.find(m=>m.id===right)} disabled={!!pair || busy} />
        <button className="icon-button" aria-label="Refresh models" disabled={active || loadingModels || busy} onClick={refresh}><RefreshCw size={17} className={loadingModels ? 'spin' : ''} /></button>
      </div>
      <ImportRepository disabled={busy} onImported={r => {
        setRepos(old => [...old.filter(p => p.id !== r.id), r]);
        setRepoId(r.id); setPair(null); setEvidence(null); setQuestion('');
        setQuestionMode('free'); setBenchmarkId(''); setError('');
        window.history.replaceState(null, '', '?mode=compare');
      }} />
      {!pair && <section className="benchmark-picker" aria-label="Question selection">
        <label>Question type<select aria-label="Question type" value={questionMode} disabled={busy} onChange={e=>setQuestionMode(e.target.value)}><option value="benchmark">Known-answer benchmark · scored</option><option value="free">Free-form chat · unscored</option></select></label>
        {choosingBenchmark && <label>Benchmark question<select aria-label="Benchmark question" disabled={busy || !benchmarks.length} value={selectedBenchmark?.id || ''} onChange={e=>setBenchmarkId(e.target.value)}>{benchmarks.length ? benchmarks.map(b=><option key={b.id} value={b.id}>{b.id} · {b.claim_count} assertions</option>) : <option value="">No saved benchmark for this repository</option>}</select></label>}
        <p>{choosingBenchmark ? 'This benchmark question is fixed and cannot be edited. Choose Free-form chat above to type your own question. Both models receive the same question; the reference answer stays with the grader.' : 'Free-form questions have no rubric score. Choose a benchmark for a comparable assertion score.'}</p>
      </section>}
      {(error || warning) && <div className="notice error" role="alert">{error || warning}<button className="icon-button" aria-label="Dismiss notice" onClick={() => { setError(''); setWarning(''); }}><X size={14} /></button></div>}
      {pair?.preview && <div className="preview-banner">Scripted interface demo · no model calls or real code execution · not a model-quality comparison</div>}
      {pair ? <>
        <div className="compare-question"><span className="eyebrow">SHARED QUESTION</span><h2>{pair.question}</h2></div>
        <div className="compare-grid">{pair.runs.map((run, i) => <RunColumn grade={pair.grading?.scores[run.id]} gradingStatus={pair.benchmark ? pair.grading?.status : undefined} key={run.id} initial={run} previous={ancestors.map(a => a.runs[i])} side={i} onUpdate={updateRun} onInspect={setEvidence} onError={setError} onActivity={updateEvidence} />)}</div>
        <ComparisonScore pair={pair} openEvidence={(path,start,end)=>setEvidence({run:pair.runs[0],selection:{type:"file",path,start,end}})} />
        {!active && <section className="comparison-summary" aria-label="Comparison summary"><h2>At a glance</h2><table><thead><tr><th>Model</th><th>Result</th><th>Score</th><th>Time</th><th>Tools</th><th>Token cost</th></tr></thead><tbody>{pair.runs.map(r => <tr key={r.id}><th>{r.model.name}</th><td>{status[r.status]}</td><td>{scoreLabel(pair.grading?.scores[r.id],pair.grading?.status)}</td><td>{r.metrics.elapsed_seconds?.toFixed(1)}s</td><td>{r.metrics.tool_calls}</td><td>{costLabel(r.metrics)}</td></tr>)}</tbody></table></section>}
      </> : <div className="compare-empty"><div className="empty-pair"><span>A</span><i>vs</i><span>B</span></div><h2>One question. Two paths to an answer.</h2><p>Choose your models above, then ask a question below.<br />Each agent gets its own tools, context, and execution budget.</p><button className="preview-link" disabled={!repo?.ready || busy} onClick={() => start(true)}><Play size={13} />Try a scripted comparison <span>No model calls</span></button></div>}
      <form className="compare-composer" onSubmit={e => { e.preventDefault(); start(); }}>
        <textarea aria-label="Ask both models" aria-describedby="compare-composer-hint" placeholder={pair ? 'Ask both models a follow-up…' : repo?.example || 'Ask both models a repository question…'} value={inputQuestion} onChange={e => setQuestion(e.target.value)} readOnly={choosingBenchmark || !!pair?.benchmark} disabled={active || busy || !!pair?.benchmark} rows={choosingBenchmark ? 5 : 2} onKeyDown={e => {
          if (e.key === 'Tab' && !e.shiftKey && !e.ctrlKey && !e.metaKey && !e.altKey && !e.nativeEvent.isComposing && !question && !pair && !choosingBenchmark && repo?.example) {
            e.preventDefault();
            setQuestion(repo.example);
            return;
          }
          if (e.key === 'Enter' && (e.metaKey || e.ctrlKey) && inputQuestion.trim() && ready) { e.preventDefault(); start(); } }} />
        <div><small id="compare-composer-hint">{active ? pair?.grading?.status === 'running' ? 'Grading both answers against the frozen assertion rubric…' : 'Both runs proceed independently. You can inspect tools as they finish.' : !question && !pair && !choosingBenchmark && repo?.example ? 'Tab to use suggestion · same question for both models' : pair?.benchmark ? 'Start a new comparison to ask another scored question.' : choosingBenchmark ? 'Known answer · frozen rubric · automatic assertion grading' : 'Same question · separate context · equal budgets'}</small>
        {active ? <button type="button" className="stop-button" onClick={async () => { try { await api(`/comparisons/${pair!.id}/cancel`, { method: 'POST' }); } catch (e) { setError((e as Error).message); } }}><Square size={12} />{pair?.grading?.status === 'running' ? 'Stop grading' : 'Stop both'}</button> : <button type="submit" className="ask-button" disabled={busy || !ready || !inputQuestion.trim()}>{busy ? <LoaderCircle size={15} className="spin" /> : <>Compare <ArrowUp size={16} /></>}</button>}</div>
      </form>
    </main>
    {evidence && <aside className="compare-evidence" aria-label="Evidence inspector"><header><strong>{evidence.run.model.name} · evidence</strong><button className="icon-button" aria-label="Close inspector" onClick={() => setEvidence(null)}><X size={18} /></button></header><Inspector runId={evidence.run.id} repoId={evidence.run.repo.id} selection={evidence.selection} step={evidence.step} onFile={(path, start = 1, end = start) => setEvidence(old => old && ({ ...old, selection: { type: 'file', path, start, end }, step: undefined }))} /></aside>}
  </div>;
}
