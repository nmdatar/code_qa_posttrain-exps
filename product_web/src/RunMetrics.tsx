import type { Run } from './types';
export function costLabel(metrics: Run['metrics']) {
  const value = metrics.cost_usd ?? metrics.estimated_cost_usd;
  if (value == null) return 'Cost unavailable';
  const label = metrics.cost_usd != null ? 'Reported token cost' : 'Estimated token cost';
  return `${label}: $${value.toFixed(5)}${metrics.token_usage_complete === false ? ' · partial' : ''}`;
}
export default function RunMetrics({ metrics }: { metrics: Run['metrics'] }) {
  return <div className="compare-metrics">
    <div><strong>{(metrics.elapsed_seconds || 0).toFixed(1)}s</strong><span>Elapsed</span></div>
    <div><strong>{metrics.tool_calls ?? 0}</strong><span>Tool calls</span></div>
    <div><strong>{metrics.steps ?? 0}</strong><span>Model turns</span></div>
    <div><strong>{metrics.input_tokens?.toLocaleString() ?? '—'}</strong><span>Input tokens</span></div>
    <div><strong>{metrics.output_tokens?.toLocaleString() ?? '—'}</strong><span>Output tokens</span></div>
    <p>{costLabel(metrics)}</p>
    <p>{(metrics.model_seconds || 0).toFixed(1)}s model wait · {(metrics.tool_seconds || 0).toFixed(1)}s tools</p>
  </div>;
}
