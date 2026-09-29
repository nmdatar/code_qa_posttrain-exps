"""A deliberately small Mermaid subset and a deterministic semantic renderer."""

import html
import re
from .schema import GRAPH, validate, unique_ids


def normalize(source):
    if source is None:
        return None
    if "mermaid" not in source:
        graph = source
    else:
        lines = [s.strip() for s in source["mermaid"].splitlines() if s.strip()]
        if not lines or lines.pop(0) not in {"flowchart TD", "flowchart LR"}:
            raise ValueError("Expected flowchart TD or flowchart LR")
        nodes, edges = [], []
        for line in lines:
            # Node IDs are canonical entity IDs. Text labels are presentation only.
            node = re.fullmatch(r'([A-Za-z_][\w.]*)\["([^"<>\n]+)"\]', line)
            edge = re.fullmatch(r'([A-Za-z_][\w.]*) -->\|([^|<>\n]+)\| ([A-Za-z_][\w.]*)', line)
            if node:
                nodes.append({"id": node[1], "entity": node[1], "label": node[2], "citations": []})
            elif edge:
                fields = edge[2].split(";", 2)
                if len(fields) != 3:
                    raise ValueError("Edge label must be relation;condition;citation IDs")
                edges.append({"source": edge[1], "target": edge[3], "relation": fields[0].strip(),
                              "condition": fields[1].strip(),
                              "citations": [v.strip() for v in fields[2].split(",") if v.strip()]})
            else:
                raise ValueError(f"Unsupported Mermaid syntax: {line[:100]}")
        graph = {"nodes": nodes, "edges": edges}
    validate(graph, GRAPH)
    unique_ids(graph["nodes"], "graph node")
    ids = {n["id"] for n in graph["nodes"]}
    if any(e["source"] not in ids or e["target"] not in ids for e in graph["edges"]):
        raise ValueError("Graph edge has nonexistent endpoint")
    # Sorting and deduplication make layouts and duplicate arrows score-invariant.
    nodes = sorted(graph["nodes"], key=lambda n: n["id"])
    edges = {tuple((e["source"], e["target"], e["relation"], e["condition"])): e for e in graph["edges"]}
    return {"nodes": nodes, "edges": [edges[k] for k in sorted(edges)]}


def render_svg(graph):
    """Render each relationship as a labeled source → target row.

    Repeated node appearances preserve entity IDs. This avoids graph layout
    heuristics, crossing arrows, hidden conditions, and platform dependencies.
    """
    nodes = {n["id"]: n for n in graph["nodes"]}
    label = lambda id: nodes[id]["label"] + " [" + nodes[id]["entity"] + "]"
    rows = [(label(e["source"]), e["relation"] + (": " + e["condition"] if e["condition"] else ""),
             label(e["target"])) for e in graph["edges"]]
    used = {e[k] for e in graph["edges"] for k in ("source", "target")}
    rows += [(label(n), "", "") for n in sorted(nodes) if n not in used]
    widths = [max(160, max((len(r[i]) for r in rows), default=0) * 10 + 32) for i in range(3)]
    width, height = sum(widths) + 120, len(rows) * 90 + 30
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img">',
           '<title>Verified graph relationships</title>',
           '<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8" fill="#334155"/></marker></defs>',
           '<rect width="100%" height="100%" fill="white"/>']
    for i, row in enumerate(rows):
        y, left, right = 25 + i * 90, 20, widths[0] + widths[1] + 80
        for x, text, w in [(left, row[0], widths[0]), (right, row[2], widths[2])]:
            if text:
                out += [f'<rect x="{x}" y="{y}" width="{w}" height="42" rx="6" fill="#eff6ff" stroke="#334155"/>',
                        f'<text x="{x+12}" y="{y+26}" font-family="monospace" font-size="14">{html.escape(text)}</text>']
        if row[2]:
            out += [f'<path d="M{left+widths[0]+5},{y+24} H{right-6}" stroke="#334155" marker-end="url(#arrow)"/>',
                    f'<text x="{left+widths[0]+20}" y="{y+10}" font-family="monospace" font-size="14">{html.escape(row[1])}</text>']
    return "\n".join(out + ["</svg>"])
