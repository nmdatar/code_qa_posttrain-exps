"""Durable single-writer run artifacts and a process-safe shared spending ledger."""
import contextlib
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time
import uuid


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
        os.replace(name, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try: os.fsync(directory)
        finally: os.close(directory)
    finally:
        if os.path.exists(name): os.unlink(name)


@contextlib.contextmanager
def lock(path, blocking=True):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        try: yield
        finally: fcntl.flock(stream, fcntl.LOCK_UN)


def append(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with lock(str(path) + '.lock'):
        with path.open('a') as stream:
            stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + '\n')
            stream.flush(); os.fsync(stream.fileno())


class BudgetExceeded(RuntimeError): pass


class BudgetLedger:
    """Reserve conservative upper estimates. Unknown outcomes retain their full charge.

    This is a dispatch guard, not a provider-enforced billing limit. All paid paths
    must share this ledger and supply externally verified conservative bounds.
    """
    def __init__(self, path, cap_usd=20.0, reserve_usd=2.0):
        self.path = Path(path)
        if not all(math.isfinite(x) for x in (cap_usd, reserve_usd)) or not 0 <= reserve_usd < cap_usd <= 20:
            raise ValueError('Budget requires 0 <= reserve < cap <= $20')
        with lock(str(self.path) + '.lock'):
            if self.path.exists():
                old = read(self.path)
                if old['cap_usd'] != cap_usd or old['reserve_usd'] != reserve_usd:
                    raise ValueError('Existing shared budget cannot silently change')
            else: atomic(self.path, {'cap_usd': cap_usd, 'reserve_usd': reserve_usd, 'calls': {}})

    def reserve(self, provider, upper_usd, metadata=None):
        if not math.isfinite(upper_usd) or upper_usd <= 0:
            raise ValueError('Paid call needs a positive finite upper cost bound')
        with lock(str(self.path) + '.lock'):
            state = read(self.path)
            if state.get('halted'): raise BudgetExceeded('Ledger halted after cost estimate violation')
            committed = sum(c['charged_usd'] for c in state['calls'].values())
            if committed + upper_usd > state['cap_usd'] - state['reserve_usd'] + 1e-9:
                raise BudgetExceeded('Insufficient unreserved budget; no call dispatched')
            key = uuid.uuid4().hex
            state['calls'][key] = dict(provider=provider, upper_usd=upper_usd, charged_usd=upper_usd,
                                      actual_usd=None, status='reserved', created_at=time.time(), metadata=metadata or {})
            atomic(self.path, state)
        return key

    def settle(self, key, actual_usd=None, status='completed'):
        with lock(str(self.path) + '.lock'):
            state = read(self.path); call = state['calls'][key]
            if call['status'] != 'reserved': raise ValueError('Reservation already settled')
            if actual_usd is not None and (not math.isfinite(actual_usd) or actual_usd < 0):
                raise ValueError('Invalid actual cost')
            call.update(actual_usd=actual_usd, charged_usd=actual_usd if actual_usd is not None else call['upper_usd'], status=status)
            if actual_usd is not None and actual_usd > call['upper_usd']: state['halted'] = True
            atomic(self.path, state)
            if actual_usd is not None and actual_usd > call['upper_usd']:
                raise BudgetExceeded('Observed charge exceeded configured estimate; stop paid dispatch')

    def execute(self, provider, upper_usd, operation, metadata=None):
        key = self.reserve(provider, upper_usd, metadata)
        try: result = operation()
        except BaseException:
            self.settle(key, status='unknown_outcome'); raise
        self.settle(key)  # Keep upper reservation unless authoritative billing is supplied.
        return result

    def summary(self):
        state = read(self.path)
        charged = sum(c['charged_usd'] for c in state['calls'].values())
        return {**state, 'charged_or_reserved_usd': charged,
                'available_for_dispatch_usd': max(0, state['cap_usd'] - state['reserve_usd'] - charged),
                'unknown_actual_cost_calls': sum(c['actual_usd'] is None for c in state['calls'].values())}
