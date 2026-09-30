"""Aggregate strict evaluation outcomes without hiding unresolved tasks."""
from qa_eval.reporting import mean, percentile


def evaluation_metrics(rows):
    n = len(rows)
    reports = [r['verification'].get('report') for r in rows]
    reports = [r for r in reports if r]
    resolved = [r for r in rows if r['verification']['status'] == 'resolved']
    links = [supported for r in reports if r.get('semantic')
             for supported in {(x['claim_id'], x['citation_id']): x['supported']
                               for x in r['semantic']['citation_links']}.values()]
    metrics = [r['metrics'] for r in reports if r.get('metrics')]
    costs = [m['cost'] for m in metrics if m.get('cost') is not None]
    latencies = [m['latency_seconds'] for m in metrics]
    # Missing assessments contribute zero demonstrated claim coverage; error rate
    # is explicitly observed and accompanied by unresolved counts.
    return {
        'required_claim_coverage': sum((r['verification'].get('coverage') or 0) for r in rows) / n if n else None,
        'observed_material_error_rate': sum(bool(r.get('material_error')) for r in reports) / n if n else None,
        'citation_support_rate': mean(links),
        'scoring_coverage': len(resolved) / n if n else None,
        'unresolved_tasks': n - len(resolved),
        'completion_rate': sum(m.get('termination_reason') == 'completed' for m in metrics) / n if n else None,
        'latency_p50': percentile(latencies, .5), 'latency_p95': percentile(latencies, .95),
        'cost_usd': sum(costs) if len(costs) == n and n else None,
        'known_cost_usd': sum(costs), 'missing_cost_tasks': n - len(costs),
    }
