import { useEffect, useState } from "react";
import {
  Braces,
  FileCode2,
  Terminal,
  ArrowUpRight,
  AlertCircle,
  ChevronLeft,
  ChevronRight,
} from "lucide-react";
import Prism from "prismjs";
import "prismjs/components/prism-python";
import "prismjs/components/prism-json";
import { api } from "./types";
import type { Source, ToolStep, Selection } from "./types";
function Code({
  text,
  language = "python",
}: {
  text: string;
  language?: string;
}) {
  const html = Prism.highlight(
    text,
    Prism.languages[language] || Prism.languages.plain,
    language,
  );
  return <code dangerouslySetInnerHTML={{ __html: html }} />;
}
export default function Inspector({
  runId,
  repoId,
  selection,
  step,
  onFile,
}: {
  runId?: string;
  repoId?: string;
  selection: Selection;
  step?: ToolStep;
  onFile: (p: string, s?: number, e?: number) => void;
}) {
  const [source, setSource] = useState<Source | null>(null),
    [error, setError] = useState("");
  const args = step?.call.arguments || {};
  const content = step?.result?.observation?.content;
  const path =
    selection?.type === "file"
      ? selection.path
      : step?.call.name === "read_file" || step?.call.name === "find_symbols"
        ? String(args.path || "")
        : "";
  const start =
    selection?.type === "file" ? selection.start : Number(args.start_line || 1);
  const end =
    selection?.type === "file"
      ? selection.end
      : start + Number(args.line_count || 1) - 1;
  useEffect(() => {
    setSource(null);
    setError("");
    if ((!runId && !repoId) || !path) return;
    const controller = new AbortController();
    api<Source>(
      `${runId ? '/runs/' + runId : '/repos/' + repoId}/source?path=${encodeURIComponent(path)}&start=${Math.max(1, start - 5)}&count=160`,
      { signal: controller.signal },
    )
      .then(setSource)
      .catch((e) => {
        if (e.name !== "AbortError") setError(e.message);
      });
    return () => controller.abort();
  }, [runId, repoId, path, start]);
  const execution =
    step && ["python_probe", "run_tests"].includes(step.call.name || "");
  return (
    <>
      <div className="panel-title">
        <span>INSPECTOR</span>
        <span className="muted">
          {execution ? "Execution" : path ? "Source" : "Evidence"}
        </span>
      </div>
      {error && (
        <div className="notice error">
          <AlertCircle size={15} />
          {error}
        </div>
      )}
      {path ? (
        <>
          <div className="source-heading">
            <FileCode2 size={15} />
            <span title={path}>{path}</span>
          </div>
          <div className="source-meta">
            <span>
              Lines {start}–{end}
            </span>
            <span>{source?.commit.slice(0, 8)}</span>
          </div>
          <div className="source-code">
            {source ? (
              source.lines.map((line, i) => (
                <div
                  className={
                    "code-line " +
                    (source.start_line + i >= start &&
                    source.start_line + i <= end
                      ? "highlighted"
                      : "")
                  }
                  key={i}
                >
                  <span className="line-number">{source.start_line + i}</span>
                  <pre>
                    <Code
                      text={line || " "}
                      language={path.endsWith(".json") ? "json" : "python"}
                    />
                  </pre>
                </div>
              ))
            ) : (
              <p className="empty-note">Loading pinned source…</p>
            )}
          </div>
          {source && (
            <div className="source-paging">
              <button
                disabled={source.start_line === 1}
                onClick={() =>
                  onFile(
                    path,
                    Math.max(1, source.start_line - 150),
                    Math.max(1, source.start_line - 150),
                  )
                }
              >
                <ChevronLeft size={14} />
                Previous
              </button>
              <span>{source.total_lines} lines</span>
              <button
                disabled={
                  source.start_line + source.lines.length > source.total_lines
                }
                onClick={() =>
                  onFile(path, source.start_line + 150, source.start_line + 150)
                }
              >
                Next
                <ChevronRight size={14} />
              </button>
            </div>
          )}
        </>
      ) : step ? (
        <div className="tool-inspector">
          <div className="inspector-tool-title">
            {execution ? <Terminal size={18} /> : <SearchIcon />}
            <strong>{step.call.name}</strong>
          </div>
          <div className="section-label">
            {execution ? "CODE / COMMAND" : "INPUT"}
          </div>
          <pre className="code-block">
            <Code
              text={
                typeof args.code === "string"
                  ? args.code
                  : JSON.stringify(args, null, 2)
              }
              language={args.code ? "python" : "json"}
            />
          </pre>
          {execution ? (
            <>
              <div className="section-label">
                OUTPUT{" "}
                {content && (
                  <span
                    className={
                      content.exit_code === 0 ? "success-text" : "failure-text"
                    }
                  >
                    exit {String(content.exit_code ?? "unknown")}
                  </span>
                )}
              </div>
              {content?.synthetic === true && (
                <div className="preview-tag">
                  Synthetic output · no code executed
                </div>
              )}
              <pre className="terminal-output">
                {content
                  ? String(content.stdout || "(no stdout)")
                  : "Waiting for execution…"}
              </pre>
              {!!content?.stderr && (
                <pre className="terminal-output stderr">
                  {String(content.stderr)}
                </pre>
              )}
              {content?.timed_out === true && (
                <div className="notice error">Execution timed out.</div>
              )}
            </>
          ) : (
            <>
              <div className="section-label">RESULT</div>
              {Array.isArray(content?.matches) ? (
                <div className="search-matches">
                  {(
                    content.matches as {
                      path: string;
                      line: number;
                      text: string;
                    }[]
                  ).map((m, i) => (
                    <button
                      key={i}
                      onClick={() => onFile(m.path, m.line, m.line)}
                    >
                      <span>
                        {m.path}:{m.line}
                        <ArrowUpRight size={12} />
                      </span>
                      <code>{m.text}</code>
                    </button>
                  ))}
                </div>
              ) : (
                <pre className="result-json">
                  {content
                    ? JSON.stringify(content, null, 2)
                    : "Waiting for result…"}
                </pre>
              )}
            </>
          )}
          {step.result?.observation?.truncated && (
            <div className="notice">
              Output truncated. The agent can retrieve bounded artifact ranges.
            </div>
          )}
          {step.result?.observation?.error && (
            <div className="notice error">{step.result.observation.error}</div>
          )}
        </div>
      ) : (
        <div className="inspector-empty">
          <div className="braces-orbit">
            <Braces size={34} strokeWidth={1.2} />
          </div>
          <h3>See the evidence.</h3>
          <p>Source lines, search results, and execution output appear here.</p>
          <small>Select any step to look closer.</small>
        </div>
      )}
    </>
  );
}
function SearchIcon() {
  return <Braces size={18} />;
}
