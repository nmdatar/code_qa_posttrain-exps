"""Bounded waves: stable ordering, no detached work after failure or interruption."""
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from functools import wraps


def ordered_map(call, items, workers):
    items = list(items)
    if workers == 1:
        return [call(item) for item in items]
    results = [None] * len(items)
    pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix='qa-rollout')
    pending = {}
    cursor = 0
    try:
        while cursor < len(items) or pending:
            while cursor < len(items) and len(pending) < workers:
                pending[pool.submit(call, items[cursor])] = cursor
                cursor += 1
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            # Check every finished future before dispatching any replacement work.
            for future in done:
                results[pending.pop(future)] = future.result()
        return results
    finally:
        for future in pending:
            future.cancel()
        # In-flight episodes own their cleanup. Never update/close clients until drained.
        pool.shutdown(wait=True, cancel_futures=True)


def synchronized(method):
    @wraps(method)
    def wrapped(self, *args, **kwargs):
        with self._io_lock:
            return method(self, *args, **kwargs)
    return wrapped


class SamplingClient:
    """Serialize SDK request submission, not waiting on independent response futures."""
    def __init__(self, client):
        import threading
        self._client = client
        self._lock = threading.Lock()

    def sample(self, *args, **kwargs):
        with self._lock:
            return self._client.sample(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._client, name)
