"""Conservative, persistent reservations. Unknown calls retain their reservation."""
from datetime import datetime, timezone
import json
import math
import ssl
import fcntl
from contextlib import contextmanager
from pathlib import Path
import urllib.request
from .storage import atomic_json, read


class BudgetLimit(RuntimeError):
    pass


def current_prices(model):
    try:
        import certifi
        context = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        context = ssl.create_default_context()
    url = 'https://tinker-docs.thinkingmachines.ai/tinker/models.json'
    with urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'}), timeout=30, context=context) as response:
        models = json.load(response)
    matches = [m for m in models if m['tinker_id'] == model]
    if len(matches) != 1:
        raise ValueError('No unambiguous published price for model')
    m = matches[0]
    prices = {k: float(m[k].removeprefix('$')) for k in ('prefill', 'sample', 'train')}
    # Storage is not in models.json. Require the published rate to remain visible.
    page = 'https://tinker-docs.thinkingmachines.ai/tinker/models/'
    with urllib.request.urlopen(urllib.request.Request(page, headers={'User-Agent': 'Mozilla/5.0'}), timeout=30, context=context) as response:
        html = response.read().decode()
    if '$0.10 per GB per month' not in html:
        raise ValueError('Cannot verify storage pricing; update estimator before spending')
    return {**prices, 'storage_gb_month': .10, 'params': m['params'], 'source': url,
            'storage_source': page, 'checked_at': datetime.now(timezone.utc).isoformat()}


@contextmanager
def ledger_lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(path.suffix + '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


class SpendLedger:
    def __init__(self, path, cap, prices, ttl_seconds):
        self.path = Path(path)
        if not math.isfinite(cap) or cap <= 0:
            raise ValueError('Invalid spend ceiling')
        if type(ttl_seconds) is not int or ttl_seconds <= 0:
            raise ValueError('Checkpoint TTL must be positive')
        for key in ('prefill', 'sample', 'train', 'params', 'storage_gb_month'):
            if not math.isfinite(prices[key]) or prices[key] <= 0:
                raise ValueError('Invalid pricing')
        self.cap, self.prices, self.ttl = cap, prices, ttl_seconds
        with ledger_lock(self.path):
            self._initialize()

    def _initialize(self):
        cap, prices, ttl_seconds = self.cap, self.prices, self.ttl
        if self.path.exists():
            self.state = read(self.path)
            if self.state['cap'] != cap or self.state['prices'] != prices or self.state['ttl_seconds'] != ttl_seconds:
                raise ValueError('Cannot change an existing spend ledger')
        else:
            self.state = {'cap': cap, 'prices': prices, 'ttl_seconds': ttl_seconds,
                          'reserved_usd': 0.0, 'reservations': [], 'actual_billing_usd': None}
            atomic_json(self.path, self.state)

    def estimate(self, kind, input_tokens=0, output_tokens=0):
        if any(type(v) is not int or v < 0 for v in (input_tokens, output_tokens)):
            raise ValueError('Token counts must be nonnegative integers')
        p = self.prices
        if kind == 'sample':
            return (input_tokens*p['prefill'] + output_tokens*p['sample']) / 1e6
        if kind == 'train':
            return input_tokens*p['train'] / 1e6
        if kind == 'checkpoint':
            # 32 bytes per base-model parameter for BOTH state and sampler artifacts:
            # deliberately much larger than rank-8 adapters + optimizer state.
            gb = p['params'] * 32 / 1e9
            return gb * p['storage_gb_month'] * self.ttl / (28*86400)
        raise ValueError('Unpriced operation')

    def reserve(self, kind, **usage):
        with ledger_lock(self.path):
            self.state = read(self.path)
            return self._reserve(kind, **usage)

    def _reserve(self, kind, **usage):
        amount = self.estimate(kind, **usage)
        if self.state['reserved_usd'] + amount > self.cap:
            raise BudgetLimit('Conservative reservation would exceed spending ceiling')
        self.state['reserved_usd'] += amount
        self.state['reservations'].append({'kind': kind, 'upper_estimate_usd': amount, **usage})
        atomic_json(self.path, self.state)
        return amount

    def reserve_external(self, kind, amount, **metadata):
        if not math.isfinite(amount) or amount < 0:
            raise ValueError('Invalid external reservation')
        with ledger_lock(self.path):
            self.state = read(self.path)
            if self.state['reserved_usd'] + amount > self.cap:
                raise BudgetLimit('External reservation exceeds spending ceiling')
            self.state['reserved_usd'] += amount
            self.state['reservations'].append({'kind': kind, 'upper_estimate_usd': amount, **metadata})
            atomic_json(self.path, self.state)
