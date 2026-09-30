"""POSIX isolated workers for explicitly read-only, independent tool calls.

Fork preserves host resource handles and each child's main-thread deadline guard.
Only parent code commits artifacts/events. Plugins must opt in and must not spawn
background children or mutate external resources when declaring parallel safety.
"""
from __future__ import annotations
import hashlib
import multiprocessing
from multiprocessing.connection import wait
import time
from .artifacts import json_text
from .contracts import ToolContext, ToolObservation
from .registry import ToolInputError, ToolScopeError, ToolTimeoutError, validate_value


class _BufferedArtifacts:
    def __init__(self, store):
        self.store, self.pending = store, {}

    def put(self, episode_id, value):
        # Materialize independent JSON to prevent later plugin mutation.
        import json
        content = json_text(value)
        key = hashlib.sha256(content.encode('utf-8')).hexdigest()
        self.pending[(episode_id, key)] = json.loads(content)
        return key

    def get(self, episode_id, artifact_id):
        key = (episode_id, artifact_id)
        return self.pending[key] if key in self.pending else self.store.get(episode_id, artifact_id)


def _worker(pipe, registry, call, context, request, timeout):
    artifacts = _BufferedArtifacts(context.artifacts)
    started = time.monotonic()
    violation = None
    try:
        observation = registry.dispatch(call, ToolContext(context.resources, artifacts, context.episode_id),
                                        request, timeout_seconds=timeout)
    except ToolScopeError as exc:
        violation = str(exc)
        observation = ToolObservation('error', None, error=str(exc))
    except ToolInputError as exc:
        observation = ToolObservation('error', None, error=str(exc))
    except ToolTimeoutError:
        observation = ToolObservation('timeout', None, error='tool exceeded its deadline')
    except BaseException as exc:
        pipe.send(('error', type(exc).__name__))
        pipe.close()
        return
    pipe.send(('ok', observation, violation, time.monotonic() - started, artifacts.pending))
    pipe.close()


def dispatch_parallel(registry, calls, context, request, timeout):
    """Return observations in input order; terminate and reap every child on exit."""
    specs = {s.name: s for s in registry.available(request)}
    if not calls or len(calls) > request.limits.max_parallel_tool_calls:
        raise ToolInputError('parallel batch exceeds configured limit')
    for call in calls:
        if call.name not in specs or specs[call.name].parallel_safe is not True:
            raise ToolScopeError('tool is not explicitly parallel-safe: ' + call.name)
    for call in calls:
        validate_value(call.arguments, specs[call.name].input_schema)
    if 'fork' not in multiprocessing.get_all_start_methods():
        raise RuntimeError('parallel tools require POSIX fork support')
    mp = multiprocessing.get_context('fork')
    workers, readers, results = [], {}, {}
    deadline = time.monotonic() + timeout
    try:
        for index, call in enumerate(calls):
            reader, writer = mp.Pipe(duplex=False)
            worker = mp.Process(target=_worker, args=(writer, registry, call, context, request,
                                                     max(0, deadline - time.monotonic())))
            worker.start()
            writer.close()
            workers.append(worker)
            readers[reader] = index
        while readers:
            left = deadline - time.monotonic()
            if left <= 0:
                raise ToolTimeoutError('parallel batch deadline exhausted')
            ready = wait(list(readers), timeout=left)
            if not ready:
                raise ToolTimeoutError('parallel batch deadline exhausted')
            for reader in ready:
                index = readers.pop(reader)
                try:
                    result = reader.recv()
                except EOFError as exc:
                    raise RuntimeError('parallel tool worker exited without a result') from exc
                finally:
                    reader.close()
                if result[0] != 'ok':
                    raise RuntimeError('parallel tool worker failed: ' + result[1])
                results[index] = result[1:]
        ordered = []
        for index in range(len(calls)):
            observation, violation, duration, pending = results[index]
            for (episode, _), value in pending.items():
                context.artifacts.put(episode, value)
            ordered.append((observation, violation, duration))
        return ordered
    finally:
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
            worker.join(timeout=1)
            if worker.is_alive():
                worker.kill()
                worker.join()
            worker.close()
        for reader in readers:
            reader.close()
