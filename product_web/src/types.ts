export interface Repository {
  id: string;
  name: string;
  commit: string;
  execution: boolean;
  ready: boolean;
  reason: string | null;
  example: string;
}
export interface Model {
  trained_harness?: "bash" | "structured";
  id: string;
  name: string;
  kind: string;
  base_model?: string;
  model_path?: string | null;
  parameters?: number;
  active_parameters?: number;
  pricing?: { input_per_million: number; output_per_million: number; source?: string; checked_at?: string } | null;
  ready?: boolean;
  reason?: string | null;
  run?: string;
  step?: number;
  created_at?: string;
}
export interface Citation {
  claim?: string;
  path: string;
  start_line: number;
  end_line: number;
  verified: boolean;
}
export interface Comparison {
  benchmark?: Benchmark;
  grading?: Grading;
  id: string;
  parent_comparison_id?: string | null;
  repo: Repository;
  question: string;
  preview: boolean;
  status: string;
  runs: Run[];
}
export interface Run {
  harness?: "bash" | "structured";
  comparison_id?: string | null;
  side?: 'left' | 'right';
  parent_run_id?: string | null;
  id: string;
  repo: Repository;
  model: Model;
  question: string;
  preview: boolean;
  status: string;
  created_at: number;
  answer: { text: string; citations: Citation[] } | null;
  metrics: {
    steps?: number;
    cost_usd?: number | null;
    estimated_cost_usd?: number | null;
    token_usage_complete?: boolean;
    model_seconds?: number;
    tool_seconds?: number;
    elapsed_seconds?: number;
    tool_calls?: number;
    input_tokens?: number | null;
    output_tokens?: number | null;
  };
  error?: string;
}
export interface Activity {
  sequence: number;
  kind: string;
  elapsed_seconds: number;
  name?: string;
  call_id?: string;
  arguments?: Record<string, unknown>;
  tools?: { name: string }[];
  observation?: {
    status: string;
    content: Record<string, unknown> | null;
    evidence: {
      path?: string;
      line?: number;
      start_line?: number;
      end_line?: number;
    }[];
    truncated: boolean;
    error?: string;
  };
  artifact_id?: string;
  error?: string;
  feedback?: string;
}
export interface Source {
  path: string;
  start_line: number;
  lines: string[];
  total_lines: number;
  commit: string;
}
export interface ToolStep {
  call: Activity;
  result?: Activity;
}
export type Selection =
  | { type: "tool"; id: string }
  | { type: "file"; path: string; start: number; end: number }
  | null;
export async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch("/api" + path, {
    ...init,
    headers: { "Content-Type": "application/json", ...init?.headers },
  });
  if (!response.ok) {
    const value = await response.json().catch(() => ({}));
    throw new Error(
      typeof value.detail === "string"
        ? value.detail
        : `Request failed (${response.status})`,
    );
  }
  return response.json();
}

export interface Benchmark {
  id: string;
  repo_id: string;
  question: string;
  claim_count: number;
  reference_status: string;
  human_reviewed: boolean;
  reference_answer?: string;
  rubric?: {
    rubric_hash: string;
    claims: { id: string; text: string; weight: number; evidence_ids: string[] }[];
    evidence: { id: string; path: string; start_line: number; end_line: number; text: string }[];
  };
}
export interface AssertionGrade {
  status: string;
  score: number | null;
  reason?: string;
  claims?: { id: string; verdict: string; reason: string; credit: number | null; answer_quotes: string[]; evidence_ids: string[] }[];
  estimated_cost_usd?: number | null;
}
export interface Grading {
  status: string;
  judge_model?: string;
  scores: Record<string, AssertionGrade>;
  error?: string;
  estimated_cost_usd?: number | null;
}
