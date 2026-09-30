import HarnessSelect from './HarnessSelect';
import ModelSelect from './ModelSelect';
import { newestCheckpointsFirst } from './modelOrdering';
import { ImportRepository, EvidenceAnswer, InvestigationMap } from "./ResearchFeatures";
import { useEffect, useMemo, useRef, useState } from "react";
import {
  ArrowUp,
  ArrowUpRight,
  Check,
  ChevronDown,
  Circle,
  Code2,
  Cpu,
  FileCode2,
  FolderGit2,
  GitBranch,
  LoaderCircle,
  Play,
  Radio,
  RefreshCw,
  Search,
  Square,
  Terminal,
  X,
  Zap,
  PanelLeft,
  Braces,
  Clock3,
} from "lucide-react";
import RunMetrics from "./RunMetrics";
import Explorer from "./Explorer";
import Inspector from "./Inspector";
import { api } from "./types";
import type {
  Activity,
  Model,
  Repository,
  Run,
  Selection,
  ToolStep,
} from "./types";
const labels: Record<string, string> = {
  list_files: "Explore repository",
  search_code: "Search code",
  read_file: "Read source",
  find_symbols: "Find symbols",
  python_probe: "Run Python probe",
  run_tests: "Run tests",
  read_artifact: "Read saved output",
};
const icons: Record<string, typeof Search> = {
  list_files: FolderGit2,
  search_code: Search,
  read_file: FileCode2,
  find_symbols: Code2,
  python_probe: Terminal,
  run_tests: Terminal,
};
const statusLabel: Record<string, string> = {
  completed: "Complete",
  cancelled: "Stopped",
  interrupted: "Interrupted",
  agent_error: "Invalid model action",
  budget_exhausted: "Budget reached",
  infrastructure_error: "Run failed",
  running: "Investigating",
};
function detail(step: ToolStep) {
  const a = step.call.arguments || {};
  return String(
    a.path ||
      a.query ||
      (a.paths as string[] | undefined)?.join(", ") ||
      (step.call.name === "python_probe"
        ? "Isolated Python environment"
        : a.glob || "Pinned repository"),
  );
}
function duration(step: ToolStep) {
  return step.result
    ? Math.max(
        0,
        step.result.elapsed_seconds - step.call.elapsed_seconds,
      ).toFixed(1) + "s"
    : "Running";
}
export function StepCard({
  step,
  selected,
  onSelect,
  stopped,
}: {
  step: ToolStep;
  selected: boolean;
  onSelect: () => void;
  stopped: boolean;
}) {
  const Icon = icons[step.call.name || ""] || Braces;
  const o = step.result?.observation;
  const failed =
    o &&
    (o.status !== "ok" ||
      (!!o.content &&
        (Number(o.content.exit_code || 0) !== 0 ||
          o.content.timed_out === true)));
  return (
    <button
      className={
        "step-card " + (selected ? "active " : "") + (failed ? "failed" : "")
      }
      onClick={onSelect}
    >
      <span className="step-icon">
        <Icon size={18} />
      </span>
      <span className="step-body">
        <strong>{labels[step.call.name || ""] || step.call.name}</strong>
        <span>{detail(step)}</span>
      </span>
      <span className="step-state">
        {step.result ? (
          failed ? (
            <X size={14} />
          ) : (
            <Check size={14} />
          )
        ) : stopped ? (
          <Square size={12} />
        ) : (
          <LoaderCircle className="spin" size={15} />
        )}
        <small>
          {step.result ? duration(step) : stopped ? "Stopped" : "Running"}
        </small>
      </span>
    </button>
  );
}
export default function App() {
  const [repos, setRepos] = useState<Repository[]>([]),
    [models, setModels] = useState<Model[]>([]),
    [repoId, setRepoId] = useState(""),
    [modelId, setModelId] = useState("base");
  const [harness, setHarness] = useState("auto");
  const [chatHistory, setChatHistory] = useState<Run[]>([]);
  const [question, setQuestion] = useState(""),
    [run, setRun] = useState<Run | null>(null),
    [events, setEvents] = useState<Activity[]>([]),
    [paths, setPaths] = useState<string[]>([]);
  const [selection, setSelection] = useState<Selection>(null),
    [follow, setFollow] = useState(true),
    [error, setError] = useState(""),
    [warning, setWarning] = useState(""),
    [busy, setBusy] = useState(false),
    [loading, setLoading] = useState(true),
    [tab, setTab] = useState("activity"),
    [connected, setConnected] = useState(true),
    [history, setHistory] = useState<Run[]>([]),
    [historyOpen, setHistoryOpen] = useState(false);
  const bottom = useRef<HTMLDivElement>(null),
    composer = useRef<HTMLTextAreaElement>(null);
  const repo = repos.find((r) => r.id === repoId),
    model = models.find((m) => m.id === modelId),
    active = run?.status === "running";
  useEffect(() => {
    const controller = new AbortController();
    Promise.all([
      api<Repository[]>("/repos", { signal: controller.signal }),
      api<{ models: Model[]; warning?: string }>("/models", {
        signal: controller.signal,
      }),
      api<Run[]>("/runs", { signal: controller.signal }),
    ])
      .then(([r, m, h]) => {
        setRepos(r);
        const requestedRepo = new URLSearchParams(location.search).get("repo");
        setRepoId(r.find(repo => repo.id === requestedRepo)?.id || r[0]?.id || "");
        setModels(newestCheckpointsFirst(m.models));
        setWarning(m.warning || "");
        setHistory(h);
        const id = new URLSearchParams(location.search).get("run");
        if (id)
          api<Run>("/runs/" + id, { signal: controller.signal })
            .then(selectRun)
            .catch((e) => { if (e.name !== "AbortError") setError(e.message); });

      })
      .catch((e) => {
        if (e.name !== "AbortError") setError(e.message);
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false);
      });
    return () => controller.abort();
  }, []);
  useEffect(() => {
    const controller = new AbortController();
    setChatHistory([]);
    async function loadConversation() {
      const previous: Run[] = [];
      const visited = new Set<string>();
      let id = run?.parent_run_id;
      while (id && !visited.has(id)) {
        visited.add(id);
        const parent = await api<Run>("/runs/" + id, { signal: controller.signal });
        previous.unshift(parent);
        id = parent.parent_run_id;
      }
      if (!controller.signal.aborted) setChatHistory(previous);
    }
    loadConversation().catch(e => { if (e.name !== "AbortError") setError(e.message); });
    return () => controller.abort();
  }, [run?.id, run?.parent_run_id]);
  function selectRun(value: Run) {
    if (value.comparison_id) {
      window.location.assign('?comparison=' + value.comparison_id);
      return;
    }
    setRun(value);
    setRepoId(value.repo.id);
    if (value.model.kind !== "preview") setModelId(value.model.id);
    setQuestion("");
    setEvents([]);
    setPaths([]);
    setSelection(null);
    setFollow(true);
    setHistoryOpen(false);
    window.history.replaceState(null, "", "?run=" + value.id);
  }
  function selectRepository(id: string) {
    setRepoId(id);
    setRun(null);
    setHistoryOpen(false);
    setChatHistory([]);
    setEvents([]);
    setPaths([]);
    setSelection(null);
    setQuestion("");
    setError("");
    setFollow(true);
    window.history.replaceState(null, "", "?repo=" + id);
  }
  useEffect(() => {
    if (!repoId) return;
    setPaths([]);
    const controller = new AbortController();
    const endpoint = run ? `/runs/${run.id}/files` : `/repos/${repoId}/files`;
    api<{ paths: string[] }>(endpoint, { signal: controller.signal })
      .then(value => setPaths(value.paths))
      .catch(e => { if (e.name !== "AbortError") setError(e.message); });
    return () => controller.abort();
  }, [repoId, run?.id]);
  useEffect(() => {
    if (!run?.id) return;
    const id = run.id,
      controller = new AbortController();
    let disposed = false;
    const stream = new EventSource(`/api/runs/${id}/events`);
    stream.onopen = () => setConnected(true);
    stream.onmessage = (e) => {
      if (disposed) return;
      const event: Activity = JSON.parse(e.data);
      setEvents((old) =>
        old.some((x) => x.sequence === event.sequence)
          ? old
          : [...old, event].sort((a, b) => a.sequence - b.sequence),
      );
    };
    stream.onerror = () => setConnected(false);
    stream.addEventListener('metrics', (e) => {
      if (!disposed) setRun(old => old?.id === id ? { ...old, metrics: JSON.parse((e as MessageEvent).data) } : old);
    });
    stream.addEventListener("terminal", (e) => {
      if (disposed) return;
      const value: Run = JSON.parse((e as MessageEvent).data);
      setRun(old => old?.id === id ? value : old);
      setConnected(true);
      stream.close();
      api<Run[]>("/runs")
        .then(setHistory)
        .catch(() => {});
    });
    return () => {
      disposed = true;
      controller.abort();
      stream.close();
    };
  }, [run?.id]);
  const steps = useMemo(() => {
    const results = new Map(
      events
        .filter((e) => e.kind === "tool_observation")
        .map((e) => [e.call_id, e]),
    );
    return events
      .filter((e) => e.kind === "tool_call")
      .map((call) => ({ call, result: results.get(call.call_id) }));
  }, [events]);
  const currentSelection: Selection =
    follow && steps.length
      ? { type: "tool", id: steps.at(-1)!.call.call_id! }
      : selection;
  const selectedStep =
    currentSelection?.type === "tool"
      ? steps.find((s) => s.call.call_id === currentSelection.id)
      : undefined;
  const marks = useMemo(() => {
    const m = new Map<string, string>();
    steps.forEach(({ call, result }) => {
      if (!result) return;
      const o = result.observation;
      if (o?.status !== "ok") return;
      if (call.name === "search_code") {
        o.evidence?.forEach((e) => {
          if (e.path && !m.has(e.path)) m.set(e.path, "match");
        });
      }
      if (call.name === "read_file" && call.arguments?.path)
        m.set(String(call.arguments.path), "read");
      if (
        call.name === "run_tests" &&
        Array.isArray(call.arguments?.paths) &&
        o.content?.synthetic !== true
      )
        call.arguments.paths.forEach((p) => m.set(String(p), "execute"));
    });
    return m;
  }, [steps]);
  useEffect(() => {
    if (run && follow)
      bottom.current?.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }, [events.length, follow, run?.answer]);
  function openFile(path: string, start = 1, end = start) {
    setFollow(false);
    setSelection({ type: "file", path, start, end });
    setTab("inspector");
  }
  async function start(preview = false) {
    if (!repoId || busy || active) return;
    setBusy(true);
    setError("");
    try {
      const value = await api<Run>("/runs", {
        method: "POST",
        body: JSON.stringify({
          repo_id: repoId,
          model_id: run?.model.id || modelId,
          harness: run?.harness || harness,
          question: question.trim() || repo?.example,
          preview: run?.preview || preview,
          parent_run_id: run?.id,
        }),
      });
      selectRun(value);
      setTab("activity");
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }
  async function refresh() {
    setBusy(true);
    try {
      const r = await api<{ models: Model[]; warning?: string }>(
        "/models?refresh=true",
      );
      setModels(newestCheckpointsFirst(r.models));
      setWarning(r.warning || "");
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }
  async function stop() {
    try {
      await api(`/runs/${run!.id}/cancel`, { method: "POST" });
    } catch (e) {
      setError((e as Error).message);
    }
  }
  const phase =
    events.at(-1)?.kind === "model_request"
      ? "Generating next action"
      : steps.length && !steps.at(-1)?.result
        ? "Running tool"
        : "Preparing environment";
  const availableTools = events.find((e) => e.kind === "started")?.tools || [];
  const selectedPath =
    currentSelection?.type === "file"
      ? currentSelection.path
      : selectedStep?.call.name === "read_file"
        ? String(selectedStep.call.arguments?.path)
        : undefined;
  return (
    <div className="app-shell">
      <header className="app-header">
<h1 className="sr-only">action-trace repository research</h1>
        <a className="brand" href="/" aria-label="action-trace home">
          <span className="brand-mark">
            <GitBranch size={20} />
          </span>
          action-trace
          <span className="brand-divider" />
          <span className="brand-sub">Repository research</span>
        </a>
        <div className="header-actions">
          <button className="text-button" disabled={busy || loading} onClick={() => { selectRepository(repoId); composer.current?.focus(); }}>New chat</button>
          <nav className="mode-switch" aria-label="Run mode"><span aria-current="page">Single</span><a href="?mode=compare">Compare</a></nav>
          <span className="local-badge">
            <i />
            LOCAL WORKSPACE
          </span>
          <button
            className="text-button"
            onClick={() => setHistoryOpen(!historyOpen)}
          >
            <Clock3 size={15} />
            Recent runs
            <ChevronDown size={13} />
          </button>
        </div>
        {historyOpen && (
          <div className="history-menu">
            <strong>Recent investigations</strong>
            {history.length ? (
              history.map((h) => (
                <button key={h.id} onClick={() => selectRun(h)}>
                  <span>{h.question}</span>
                  <small>
                    {h.preview ? "Preview" : h.model.name} ·{" "}
                    {statusLabel[h.status]}
                  </small>
                </button>
              ))
            ) : (
              <p>No runs yet.</p>
            )}
          </div>
        )}
      </header>
      <div className="config-bar" role="region" aria-label="Run configuration">
        <div className="config-field repo-select">
          <span className="config-icon">
            <FolderGit2 size={18} />
          </span>
          <label>
            <span>REPOSITORY</span>
            <select
              aria-label="Repository"
              disabled={active || loading}
              value={repoId}
              onChange={(e) => selectRepository(e.target.value)}
            >
              {repos.map((r) => (
                <option key={r.id} value={r.id}>
                  {r.name} · {r.commit.slice(0, 7)}
                  {!r.ready ? " · unavailable" : ""}
                </option>
              ))}
            </select>
          </label>
          <span className="sha-tag">{repo?.commit.slice(0, 7) || "…"}</span>
          <ImportRepository disabled={!!active || busy} onImported={r => { setRepos(prev => [...prev.filter(p => p.id !== r.id),r]); selectRepository(r.id); }} />
        </div>
        <div className="config-field model-select">
          <span className="config-icon">
            <Cpu size={18} />
          </span>
          <div className="model-picker-field">
            <span>MODEL</span>
            <ModelSelect label="Model checkpoint" value={modelId} models={models}
              disabled={!!active || loading}
              onChange={id => { selectRepository(repoId); setModelId(id); setHarness("auto"); }} />
          </div>
          <button
            className="icon-button"
            aria-label="Refresh checkpoints"
            title="Refresh checkpoints"
            disabled={busy || active}
            onClick={refresh}
          >
            <RefreshCw size={14} className={busy ? "spin" : ""} />
          </button>
        </div>
        <HarnessSelect value={run?.harness || harness} model={model} disabled={!!active || loading || !!run} onChange={setHarness} />
        <div className="config-capability">
          <span className="capability-dot" />
          {repo?.execution ? "Sandbox execution enabled" : "Source reading"}
          <span className="config-sub">Pinned commit · isolated runs</span>
        </div>
      </div>
      {(error || warning) && (
        <div
          className={"top-notice " + (error ? "error" : "")}
          role={error ? "alert" : "status"}
        >
          <span>{error || warning}</span>
          <button
            aria-label="Dismiss notice"
            onClick={() => {
              setError("");
              setWarning("");
            }}
          >
            <X size={14} />
          </button>
        </div>
      )}
      <nav className="mobile-tabs" aria-label="Workspace panels">
        {[
          ["explorer", "Repository", PanelLeft],
          ["activity", "Activity", Radio],
          ["inspector", "Inspector", Braces],
        ].map(([key, label, Icon]) => (
          <button
            key={key as string}
            className={tab === key ? "selected" : ""}
            onClick={() => setTab(key as string)}
          >
            {typeof Icon !== "string" && <Icon size={15} />} {label as string}
          </button>
        ))}
      </nav>
      <div className="workspace">
        <aside
          aria-label="Repository explorer"
          className={
            "explorer panel " + (tab === "explorer" ? "mobile-visible" : "")
          }
        >
          <Explorer
            repo={run?.repo || repo}
            paths={paths}
            marks={marks}
            selected={selectedPath}
            onOpen={(p) => openFile(p)}
          />
        </aside>
        <main
          className={
            "activity-panel panel " +
            (tab === "activity" ? "mobile-visible" : "")
          }
        >
          <div className="panel-title">
            <span>INVESTIGATION</span>
            <span className="activity-title-right">
              {run ? (
                <>
                  <i className={"dot " + (active ? "live" : "read")} />
                  {statusLabel[run.status]}
                </>
              ) : (
                <>
                  <Circle size={6} fill="currentColor" />
                  Ready when you are
                </>
              )}
            </span>
          </div>
          <div className="activity-scroll">
            {!run ? (
              <div className="welcome">
                <div className="eyebrow">
                  <span />
                  CODEBASE INTELLIGENCE
                </div>
                <h1>
                  Understand the code.
                  <br />
                  <span>Follow the evidence.</span>
                </h1>
                <p>
                  Ask your repository a question. Watch the agent
                  <br className="desktop-break" /> search, inspect, and test its
                  way to an answer.
                </p>
                <div className="welcome-flow">
                  <span>
                    <Search size={16} />
                    Explore
                  </span>
                  <ChevronRightSmall />
                  <span>
                    <Terminal size={16} />
                    Verify
                  </span>
                  <ChevronRightSmall />
                  <span>
                    <Check size={16} />
                    Explain
                  </span>
                </div>
                <button
                  className="example-card"
                  disabled={!repo}
                  onClick={() => {
                    setQuestion(repo?.example || "");
                    composer.current?.focus();
                  }}
                >
                  <span className="example-label">
                    A PLACE TO START
                    <ArrowUpRight size={14} />
                  </span>
                  <span>{repo?.example || "Loading your repositories…"}</span>
                </button>
                <button
                  className="preview-link"
                  disabled={busy || !repo?.ready}
                  onClick={() => start(true)}
                >
                  <Play size={12} />
                  Explore a scripted preview<span>No model calls</span>
                </button>
              </div>
            ) : (
              <>
                {chatHistory.map(previous => <section className="answer-card" key={previous.id} aria-label="Previous chat turn">
                  <h3>{previous.question}</h3>
                  {previous.answer ? <EvidenceAnswer text={previous.answer.text} citations={previous.answer.citations} openFile={openFile} /> : <p>This turn ended without an answer.</p>}
                </section>)}
                <div className="run-question">
                  <div className="eyebrow">YOUR QUESTION</div>
                  <h2>{run.question}</h2>
                  <div className="run-tags">
                    <span>
                      <GitBranch size={12} />
                      {run.repo.commit.slice(0, 7)}
                    </span>
                    <span>
                      <Cpu size={12} />
                      {run.model.name}
                    </span>
                    {run.preview && (
                      <span className="preview-tag">Scripted preview</span>
                    )}
                  </div>
                </div>
                {availableTools.length > 0 && (
                  <div className="tool-strip">
                    <span>TOOLS</span>
                    {availableTools.map((t) => (
                      <span title={t.name} key={t.name}>
                        {t.name.replaceAll("_", " ")}
                      </span>
                    ))}
                  </div>
                )}
                <div className="timeline-header">
                  <span>Agent activity</span>
                  <button
                    className={"follow-button " + (follow ? "following" : "")}
                    onClick={() => setFollow(!follow)}
                  >
                    <Radio size={12} />
                    {follow ? "Following live" : "Follow live"}
                  </button>
                </div>
                <InvestigationMap steps={steps} openFile={openFile} />

                <div className="timeline">
                  {events.filter(event => event.kind === 'submission_rejected' || event.kind === 'action_rejected').map(event => (
                    <div className="notice" role="status" key={event.sequence}>
                      <strong>{event.kind === 'action_rejected' ? 'Correcting response format' : 'Checking source evidence'}</strong>
                      <p>{event.feedback}</p>
                    </div>
                  ))}
                  {steps.map((step) => (
                    <StepCard
                      key={step.call.call_id}
                      step={step}
                      selected={
                        selectedStep?.call.call_id === step.call.call_id
                      }
                      stopped={!active}
                      onSelect={() => {
                        setFollow(false);
                        setSelection({ type: "tool", id: step.call.call_id! });
                        setTab("inspector");
                      }}
                    />
                  ))}
                  {active && (
                    <div className="working" role="status">
                      <LoaderCircle size={14} className="spin" />
                      <span>
                        {phase}
                        <span className="ellipsis">…</span>
                      </span>
                      {!connected && (
                        <small>Reconnecting to activity stream…</small>
                      )}
                    </div>
                  )}
                </div>
                {run.answer && (
                  <section className="answer-card">
                    <div className="answer-heading">
                      <span>
                        <Check size={15} />
                        {run.preview ? "Preview complete" : "Answer"}
                      </span>
                      <span>{run.metrics.elapsed_seconds?.toFixed(1)}s</span>
                    </div>
                    <EvidenceAnswer text={run.answer.text} citations={run.answer.citations} openFile={openFile} />
                  </section>
                )}
                {!active && run.status !== "completed" && (
                  <div className="notice error" role="status">
                    <strong>{statusLabel[run.status]}</strong>
                    <p>
                      {run.error ||
                        "The run ended before a final answer. Completed activity is preserved; you can try another question or checkpoint."}
                    </p>
                  </div>
                )}
                <RunMetrics metrics={run.metrics} />
                {!active && (
                  <div className="run-footer">
                    <span>
                      {run.metrics.tool_calls ?? steps.length} tool calls
                    </span>
                    <span>
                      {run.metrics.output_tokens != null
                        ? run.metrics.output_tokens.toLocaleString() +
                          " output tokens"
                        : "Token usage unavailable"}
                    </span>

                  </div>
                )}
              </>
            )}
            <div ref={bottom} />
          </div>
          <form
            className="composer-wrap"
            onSubmit={(e) => {
              e.preventDefault();
              start();
            }}
          >
            <div className="composer">
              <textarea
                ref={composer}
                aria-label="Ask a repository question"
                placeholder={run ? "Ask a follow-up…" : repo?.example || "What would you like to understand?"}
                aria-describedby="single-composer-hint"
                value={question}
                disabled={active}
                onChange={(e) => setQuestion(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Tab" && !e.shiftKey && !e.ctrlKey && !e.metaKey && !e.altKey && !e.nativeEvent.isComposing && !question && !run && repo?.example) {
                    e.preventDefault();
                    setQuestion(repo.example);
                    return;
                  }
                  if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
                    e.preventDefault();
                    start();
                  }
                }}
                rows={3}
              />
              <div className="composer-bottom">
                <span>
                  <Zap size={12} />
                  {active
                    ? "Following the evidence"
                    : run
                      ? "Continue this chat"
                      : "One question. Full context."}
                </span>
                {active ? (
                  <button className="stop-button" type="button" onClick={stop}>
                    <Square size={11} />
                    Stop
                  </button>
                ) : (
                  <button
                    className="ask-button"
                    type="submit"
                    disabled={
                      busy ||
                      loading ||
                      !repo?.ready ||
                      !model?.ready ||
                      !question.trim()
                    }
                  >
                    {busy ? (
                      <LoaderCircle size={15} className="spin" />
                    ) : (
                      <>
                        Ask
                        <ArrowUp size={16} />
                      </>
                    )}
                  </button>
                )}
              </div>
            </div>
            <div className="composer-hint" id="single-composer-hint">
              {!model?.ready && !active ? (
                <span>{model?.reason || "Loading models…"}</span>
              ) : (
                <span>Source-grounded answers · isolated execution</span>
              )}
              <span>{!question && !run && repo?.example && !active ? "Tab to use suggestion · " : ""}⌘ / Ctrl ↵</span>
            </div>
          </form>
        </main>
        <aside aria-label="Evidence inspector"
          className={
            "inspector panel " + (tab === "inspector" ? "mobile-visible" : "")
          }
        >
          <Inspector
            runId={run?.id}
            repoId={repoId}
            selection={currentSelection}
            step={selectedStep}
            onFile={openFile}
          />
        </aside>
      </div>
      <footer className="status-bar">
        <span>
          <span className="status-led" />{" "}
          {active ? "Run in progress" : "Workspace ready"}
        </span>
        <span>
          Tinker inference<span className="footer-dot">·</span>Modal sandbox
          <span className="footer-dot">·</span>Local artifacts
        </span>
      </footer>
    </div>
  );
}
function ChevronRightSmall() {
  return <span className="flow-arrow">→</span>;
}
