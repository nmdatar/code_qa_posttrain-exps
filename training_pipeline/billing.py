"""Read provider billing without attributing unrelated account activity."""
from urllib.parse import urlparse
from datetime import datetime, timezone, timedelta


def collect_billing(service, artifact_paths, started_at, timeout=120):
    runs = {urlparse(path).netloc for path in artifact_paths}
    rest = service.create_rest_client()
    ids = rest.list_sessions(limit=100).result(timeout=timeout).sessions
    selected = []
    for ident in ids:
        session = rest.get_session(ident).result(timeout=timeout)
        if runs.intersection(session.training_run_ids):
            selected.append(ident)
    # Billing is bucketed. Session attribution, rather than time alone, excludes
    # simultaneous baseline jobs; delayed buckets remain explicitly incomplete.
    start = datetime.fromisoformat(started_at)
    now = datetime.now(timezone.utc)
    billing = rest.get_billing_usage(start.replace(hour=0, minute=0, second=0, microsecond=0),
                                    now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)).result(timeout=timeout)
    events = [event for event in billing.data if event.session_id in selected]
    known = [event for event in events if event.estimated_cost_usd is not None]
    through = billing.cost_data_through
    return {'provider_reported_estimated_usd': sum(e.estimated_cost_usd for e in known) if known else None,
            'actual_billing_usd': None, 'sessions_matched': len(selected), 'events_matched': len(events),
            'cost_data_through': through.isoformat() if through else None,
            'complete': False,
            'note': 'Provider returns estimated billing events; storage continues until TTL and recent events may lag.'}
