import { useEffect, useMemo, useRef, useState } from "react";
import {
  ChevronRight,
  FileCode2,
  Folder,
  Search,
  GitBranch,
  Circle,
} from "lucide-react";
import type { Repository } from "./types";
interface Node {
  name: string;
  path: string;
  children: Map<string, Node>;
}
function Branch({
  node,
  marks,
  selected,
  onOpen,
  depth = 0,
}: {
  node: Node;
  marks: Map<string, string>;
  selected?: string;
  onOpen: (p: string) => void;
  depth?: number;
}) {
  if (!node.children.size)
    return (
      <button
        className={"file-row " + (selected === node.path ? "selected" : "")}
        style={{ paddingLeft: 14 + depth * 13 }}
        onClick={() => onOpen(node.path)}
        title={node.path}
      >
        <FileCode2 size={14} />
        <span>{node.name}</span>
        {marks.has(node.path) && (
          <Circle
            className={"file-mark " + marks.get(node.path)}
            size={7}
            fill="currentColor"
          />
        )}
      </button>
    );
  const active = [...marks.keys()].some((p) => p.startsWith(node.path + "/"));
  return (
    <details
      className="folder"
      open={active || (depth === 0 && node.name === "pydantic") || undefined}
    >
      <summary style={{ paddingLeft: 10 + depth * 13 }}>
        <ChevronRight size={12} />
        <Folder size={14} />
        <span>{node.name}</span>
      </summary>
      {[...node.children.values()]
        .sort(
          (a, b) =>
            Number(!!b.children.size) - Number(!!a.children.size) ||
            a.name.localeCompare(b.name),
        )
        .map((child) => (
          <Branch
            key={child.path}
            node={child}
            marks={marks}
            selected={selected}
            onOpen={onOpen}
            depth={depth + 1}
          />
        ))}
    </details>
  );
}
export default function Explorer({
  repo,
  paths,
  marks,
  selected,
  onOpen,
}: {
  repo?: Repository;
  paths: string[];
  marks: Map<string, string>;
  selected?: string;
  onOpen: (p: string) => void;
}) {
  const [query, setQuery] = useState("");
  const treeElement = useRef<HTMLDivElement>(null);
  useEffect(() => {
    treeElement.current?.querySelector(".file-row.selected")?.scrollIntoView({ block: "nearest" });
  }, [selected]);
  const tree = useMemo(() => {
    const root: Node = { name: "", path: "", children: new Map() };
    paths
      .filter((p) => p.toLowerCase().includes(query.toLowerCase()))
      .forEach((path) => {
        let current = root;
        path.split("/").forEach((name, i, parts) => {
          const prefix = parts.slice(0, i + 1).join("/");
          if (!current.children.has(name))
            current.children.set(name, {
              name,
              path: prefix,
              children: new Map(),
            });
          current = current.children.get(name)!;
        });
      });
    return root;
  }, [paths, query]);
  return (
    <>
      <div className="panel-title">
        <span>EXPLORER</span>
        <span className="muted">
          {paths.length ? paths.length + " files" : "Pinned source"}
        </span>
      </div>
      <div className="repo-heading">
        <Folder size={17} />
        <strong>{repo?.name.split("/").at(-1) || "Repository"}</strong>
      </div>
      <div className="commit">
        <GitBranch size={12} />
        {repo?.commit.slice(0, 8) || "Select a repository"}
        <span>read only</span>
      </div>
      <label className="file-search">
        <Search size={14} />
        <input
          aria-label="Filter files"
          placeholder="Find a file…"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
      </label>
      <div className="tree" ref={treeElement}>
        {[...tree.children.values()].map((node) => (
          <Branch
            key={node.path}
            node={node}
            marks={marks}
            selected={selected}
            onOpen={onOpen}
          />
        ))}
        {!paths.length && (
          <div className="explorer-empty">
            <div className="empty-lines">
              <i />
              <i />
              <i />
              <i />
            </div>
            <p>Your repository comes into focus as the agent explores.</p>
            <small>Start a question to browse the pinned source.</small>
          </div>
        )}
        {paths.length > 0 && !tree.children.size && (
          <p className="empty-note">No matching files.</p>
        )}
      </div>
      <div className="explorer-legend">
        <span>
          <i className="dot read" /> Read
        </span>
        <span>
          <i className="dot match" /> Match
        </span>
        <span>
          <i className="dot execute" /> Executed target
        </span>
      </div>
    </>
  );
}
