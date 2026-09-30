"""Local API, subprocess supervisor, durable event replay, and built UI serving."""
from __future__ import annotations
import asyncio
import fcntl
from contextlib import asynccontextmanager
from dataclasses import asdict
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time
import uuid
from urllib.parse import urlparse
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from typing import Literal
from pydantic import BaseModel, Field
from product_api.catalog import ROOT, repo_catalog, public_repo, open_repository, model_catalog, read
from product_api.worker import atomic_write
from product_api.limits import RUN_LIMITS, MAX_REPAIRS, MAX_OUTPUT_PER_CALL, PROVIDER_TIMEOUT_SECONDS, SUPERVISOR_GRACE_SECONDS, MAX_CONCURRENT_RUNS

TERMINAL = {'completed', 'cancelled', 'interrupted', 'agent_error', 'budget_exhausted', 'infrastructure_error'}

class StartRun(BaseModel):
    repo_id: str
    model_id: str = 'base'
    harness: Literal['auto', 'structured', 'bash'] = 'auto'
    question: str = Field(min_length=1, max_length=8000)
    preview: bool = False
    parent_run_id: str | None = None

class StartComparison(BaseModel):
    benchmark_id: str | None = None
    repo_id: str
    left_model_id: str
    left_harness: Literal['auto', 'structured', 'bash'] = 'auto'
    right_harness: Literal['auto', 'structured', 'bash'] = 'auto'
    right_model_id: str
    question: str = Field(min_length=1, max_length=8000)
    parent_comparison_id: str | None = None
    preview: bool = False

class ImportRepo(BaseModel):
    url: str = Field(max_length=500)
    ref: str = Field(default='HEAD', max_length=200)

class RunManager:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.owner_lock = (self.root / '.server.lock').open('a')
        try:
            fcntl.flock(self.owner_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.owner_lock.close()
            raise RuntimeError('Another product server already owns this run directory') from None
        self.lock = threading.Lock()
        self.catalog_lock = threading.Lock()
        self.processes = {}
        self.grading_processes = {}
        self.closing = False
        self.benchmarks = None
        self.watchers = []
        self.models = []
        self.models_at = 0
        self.warning = None
        self.repositories = {r['id']: r for r in repo_catalog()}
        self.import_root = self.root / 'repositories'
        for path in self.import_root.glob('*/repository.json'):
            row = read(path)
            self.repositories[row['id']] = row
        self.source_cache = {}
        self.comparison_root = self.root / 'comparisons'
        self.comparison_root.mkdir(exist_ok=True)
        for directory in self.root.iterdir():
            if directory.is_dir() and (directory / 'config.json').exists() and not (directory / 'result.json').exists():
                atomic_write(directory / 'result.json', {'status': 'interrupted', 'answer': None, 'metrics': {},
                    'error': 'The server restarted before the run completed. Start a new run to retry.'})

        for file in self.comparison_root.glob('*.json'):
            if file.name.endswith('.grade.json'): continue
            record = read(file)
            if not record.get('benchmark'): continue
            grade_path = self.comparison_root / (record['id'] + '.grade.json')
            grade = read(grade_path) if grade_path.exists() else {'scores':{}}
            if grade.get('status') not in ('completed','failed','interrupted','cancelled'):
                atomic_write(grade_path, {**grade, 'status':'interrupted', 'error':'Grading was interrupted by a server restart. Start a new comparison to retry.'})

    def benchmark_catalog(self):
        from product_api.benchmarks import catalog
        if self.benchmarks is None:
            self.benchmarks = catalog(self.repositories)
        return self.benchmarks

    def directory(self, run_id):
        if not re.fullmatch(r'[0-9a-f]{32}', run_id):
            raise HTTPException(404, 'Run not found')
        directory = self.root / run_id
        if not (directory / 'config.json').is_file():
            raise HTTPException(404, 'Run not found')
        return directory

    def catalogs(self, refresh=False):
        with self.catalog_lock:
            if refresh or not self.models or time.monotonic() - self.models_at > 60:
                self.models, self.warning = model_catalog()
                self.models_at = time.monotonic()
        return {'models': self.models, 'warning': self.warning}

    def prepare(self, body, models):
        if not body.question.strip():
            raise HTTPException(422, 'Enter a question')
        repo = self.repositories.get(body.repo_id)
        if not repo or not repo['ready']:
            raise HTTPException(422, 'Pinned repository is not ready')
        parent = None
        if body.parent_run_id:
            parent = self.get(body.parent_run_id)
            if parent['status'] not in TERMINAL or parent['repo']['id'] != body.repo_id or parent['model']['id'] != body.model_id or parent['preview'] != body.preview:
                raise HTTPException(422, 'Follow-ups must keep the finished run’s repository and model')
        if body.preview:
            model = {'id': body.model_id if body.model_id.startswith('preview') else 'preview',
                     'name': 'Scripted preview', 'kind': 'preview'}
        else:
            model = next((m for m in models if m['id'] == body.model_id), None)
            if not model or not model['ready']:
                raise HTTPException(422, (model or {}).get('reason') or 'Model not found')
        harness = model.get('trained_harness', 'structured') if body.harness == 'auto' else body.harness
        if parent and parent.get('harness', 'structured') != harness:
            raise HTTPException(422, 'Start a new chat to change tools')
        model = {**model, 'harness': harness, 'max_tokens_per_call': MAX_OUTPUT_PER_CALL, 'provider_timeout_seconds': PROVIDER_TIMEOUT_SECONDS}
        config = {'id': uuid.uuid4().hex, 'repo': repo, 'model': model, 'question': body.question.strip(),
                  'preview': body.preview, 'harness': harness, 'created_at': time.time(), 'limits': asdict(RUN_LIMITS),
                  'max_action_repairs': MAX_REPAIRS, 'max_submission_repairs': MAX_REPAIRS}
        if parent:
            config['parent_run_id'] = parent['id']
            config['prior_context'] = {'question': parent['question'], 'answer': parent['answer']}
        return config

    def ensure_capacity(self, required=1):
        # Called under self.lock; reserve both comparison workers atomically.
        # Completed investigations may continue grading without blocking new chats.
        active = sum(p.poll() is None for p in self.processes.values())
        if active + required > MAX_CONCURRENT_RUNS:
            raise HTTPException(429, f'All {MAX_CONCURRENT_RUNS} investigation slots are in use, or too few remain for this comparison. Wait for a run to finish or stop one in Recent runs / Recent comparisons.')

    def persist(self, config):
        directory = self.root / config['id']
        directory.mkdir()
        atomic_write(directory / 'config.json', config)

    def launch(self, config):
        run_id = config['id']
        directory = self.root / run_id
        try:
            process = subprocess.Popen([sys.executable, '-m', 'product_api.worker', str(directory)], cwd=ROOT,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
        except OSError:
            atomic_write(directory / 'result.json', {'status': 'infrastructure_error', 'answer': None,
                'metrics': {}, 'error': 'Could not start worker. No retry was submitted.'})
            return
        self.processes[run_id] = process
        watcher = threading.Thread(target=self.watch, args=(run_id, process), daemon=True)
        self.watchers.append(watcher)
        watcher.start()

    def start(self, body):
        # Reject impossible follow-ups before any provider discovery.
        if body.parent_run_id:
            parent = self.get(body.parent_run_id)
            if parent['status'] not in TERMINAL or parent['repo']['id'] != body.repo_id or parent['model']['id'] != body.model_id or parent['preview'] != body.preview:
                raise HTTPException(422, 'Follow-ups must keep the finished run’s repository and model')
        with self.lock:
            self.ensure_capacity()
        models = [] if body.preview else self.catalogs()['models']
        config = self.prepare(body, models)
        with self.lock:
            self.ensure_capacity()
            self.persist(config)
            self.launch(config)
        return self.get(config['id'])

    def start_comparison(self, body):
        if body.left_model_id == body.right_model_id:
            raise HTTPException(422, 'Choose two different models')
        with self.lock:
            self.ensure_capacity(2)
        models = [] if body.preview else self.catalogs()['models']
        benchmark = judge = None
        if body.benchmark_id:
            if body.preview or body.parent_comparison_id:
                raise HTTPException(422, 'Scored benchmarks require fresh live runs without prior chat context')
            row = self.benchmark_catalog().get(body.benchmark_id)
            if not row or row['repo_id'] != body.repo_id or row['question'] != body.question.strip():
                raise HTTPException(422, 'Use the exact benchmark question and pinned repository')
            from product_api.benchmarks import freeze
            from product_api.grading import judge_config
            try:
                benchmark = freeze(row, self.repositories[body.repo_id])
                judge = judge_config(models)
            except (ValueError, OSError) as exc:
                raise HTTPException(422, str(exc)) from None
        parent = self.comparison(body.parent_comparison_id) if body.parent_comparison_id else None
        if parent and (parent['repo']['id'] != body.repo_id or parent['preview'] != body.preview or
                       any(run['status'] not in TERMINAL for run in parent['runs'])):
            raise HTTPException(422, 'Follow-ups require a finished comparison on the same repository')
        configs = []
        comparison_id = uuid.uuid4().hex
        for index, (side, model_id) in enumerate((('left', body.left_model_id), ('right', body.right_model_id))):
            config = self.prepare(StartRun(repo_id=body.repo_id, model_id=model_id, question=body.question,
                preview=body.preview, harness=getattr(body, side+'_harness'), parent_run_id=parent['runs'][index]['id'] if parent else None), models)
            config.update(comparison_id=comparison_id, side=side)
            if benchmark and config['harness'] == 'bash' and 'bash' not in benchmark['permitted_tools']:
                raise HTTPException(422, 'This benchmark requires structured tools; choose Structured tools or a free-form question')
            if benchmark:
                config['benchmark_id'] = benchmark['id']
                config['permitted_tools'] = benchmark['permitted_tools']
            if body.preview:
                config['model'].update(id='preview-' + side, name='Scripted preview ' + ('A' if index == 0 else 'B'))
            configs.append(config)
        record = {'id': comparison_id, 'run_ids': [c['id'] for c in configs],
                  'parent_comparison_id': body.parent_comparison_id, 'created_at': time.time()}
        if benchmark:
            record.update(benchmark=benchmark, judge=judge)
        with self.lock:
            self.ensure_capacity(2)
            for config in configs:
                self.persist(config)
            atomic_write(self.comparison_root / (comparison_id + '.json'), record)
            for config in configs:
                self.launch(config)
        self.maybe_grade(comparison_id)
        return self.comparison(comparison_id)

    def comparison(self, comparison_id):
        if not re.fullmatch(r'[0-9a-f]{32}', comparison_id):
            raise HTTPException(404, 'Comparison not found')
        path = self.comparison_root / (comparison_id + '.json')
        if not path.is_file():
            raise HTTPException(404, 'Comparison not found')
        record = read(path)
        runs = [self.get(run_id) for run_id in record['run_ids']]
        grade_path = self.comparison_root / (comparison_id + '.grade.json')
        grading = read(grade_path) if grade_path.exists() else {'status':'pending' if record.get('benchmark') else 'not_scored','scores':{}}
        # Reference answers and evidence never enter the candidate worker config or prompt.
        return {**{k:v for k,v in record.items() if k != 'judge'}, 'grading':grading,
                'runs': runs, 'repo': runs[0]['repo'], 'question': runs[0]['question'],
                'preview': runs[0]['preview'], 'status': 'running' if any(r['status'] == 'running' for r in runs) or grading['status'] in ('pending','running') else 'finished'}

    def watch(self, run_id, process):
        try:
            config = read(self.root / run_id / 'config.json')
            deadline = config.get('limits', {}).get('wall_time_seconds', RUN_LIMITS.wall_time_seconds)
            process.wait(timeout=deadline + SUPERVISOR_GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=12)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        path = self.root / run_id / 'result.json'
        if not path.exists():
            cancelled = (path.parent / 'cancel-requested.json').exists()
            atomic_write(path, {'status': 'cancelled' if cancelled else 'infrastructure_error',
                'answer': None, 'metrics': {},
                'error': 'Run stopped. Completed activity is preserved.' if cancelled else
                         'Worker exited unexpectedly. Completed activity is preserved.'})

        if config.get('comparison_id'):
            self.maybe_grade(config['comparison_id'])

    def maybe_grade(self, comparison_id):
        with self.lock:
            if self.closing or comparison_id in self.grading_processes: return
            record = read(self.comparison_root / (comparison_id + '.json'))
            if not record.get('benchmark') or not all((self.root/i/'result.json').exists() for i in record['run_ids']): return
            grade_path = self.comparison_root / (comparison_id + '.grade.json')
            if grade_path.exists(): return
            atomic_write(grade_path, {'status':'running','scores':{},'judge_model':record['judge']['base_model']})
            try:
                process = subprocess.Popen([sys.executable,'-m','product_api.grading',str(self.root),comparison_id],
                    cwd=ROOT,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)
            except OSError:
                atomic_write(grade_path, {'status':'failed','scores':{},'error':'Could not start the rubric grader.'})
                return
            self.grading_processes[comparison_id] = process
            watcher = threading.Thread(target=self.watch_grading,args=(comparison_id,process),daemon=True)
            self.watchers.append(watcher)
            watcher.start()

    def watch_grading(self, comparison_id, process):
        try:
            process.wait(timeout=600)
        except subprocess.TimeoutExpired:
            process.terminate()
            try: process.wait(timeout=12)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        path = self.comparison_root / (comparison_id + '.grade.json')
        grade = read(path)
        if grade['status'] not in ('completed','cancelled'):
            atomic_write(path, {**grade,'status':'interrupted' if self.closing else 'failed',
                'error':'Grading stopped before completion; available assertion results are preserved.'})

    def get(self, run_id):
        directory = self.directory(run_id)
        config = read(directory / 'config.json')
        result = read(directory / 'result.json') if (directory / 'result.json').exists() else {'status': 'running', 'answer': None, 'metrics': {}}
        from product_api.metrics import summarize
        result['metrics'] = summarize(event_rows(directory), result, config)
        return {'comparison_id': config.get('comparison_id'), 'side': config.get('side'), 'id': run_id, 'repo': public_repo(config['repo']), 'model': config['model'],
            'parent_run_id': config.get('parent_run_id'), 'limits': config.get('limits'),
            'harness': config.get('harness', 'structured'), 'question': config['question'], 'preview': config['preview'], 'created_at': config['created_at'], **result}

    def cancel(self, run_id):
        directory = self.directory(run_id)
        process = self.processes.get(run_id)
        if process and process.poll() is None:
            atomic_write(directory / 'cancel-requested.json', {'requested_at': time.time()})
            process.terminate()
            def enforce_stop():
                try:
                    process.wait(timeout=12)
                except subprocess.TimeoutExpired:
                    process.kill()
            stopper = threading.Thread(target=enforce_stop, daemon=True)
            self.watchers.append(stopper)
            stopper.start()
        return {'status': 'cancelling' if process and process.poll() is None else self.get(run_id)['status']}

    def source(self, run_id):
        if run_id not in self.source_cache:
            config = read(self.directory(run_id) / 'config.json')
            self.source_cache[run_id] = open_repository(config['repo'])[0]
        return self.source_cache[run_id]

    def shutdown(self):
        self.closing = True
        for process in self.grading_processes.values():
            if process.poll() is None: process.terminate()
        for run_id, process in list(self.processes.items()):
            if process.poll() is None:
                self.cancel(run_id)
                try:
                    process.wait(timeout=12)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        for watcher in self.watchers:
            watcher.join(timeout=15)
        fcntl.flock(self.owner_lock.fileno(), fcntl.LOCK_UN)
        self.owner_lock.close()


def event_rows(directory):
    path = directory / 'events.jsonl'
    if not path.exists():
        return []
    lines = path.read_text().splitlines(keepends=True)
    return [json.loads(line) for line in lines if line.endswith('\n')]


def create_app(storage=None):
    @asynccontextmanager
    async def lifespan(app):
        app.state.manager = RunManager(storage or os.environ.get('PRODUCT_RUNS_DIR', ROOT / 'artifacts/product-runs'))
        yield
        await asyncio.to_thread(app.state.manager.shutdown)

    app = FastAPI(title='action-trace', lifespan=lifespan)

    @app.middleware('http')
    async def local_only(request: Request, call_next):
        host = request.url.hostname
        origin = request.headers.get('origin')
        if host not in ('127.0.0.1', 'localhost', '::1', 'testserver') or (origin and urlparse(origin).hostname not in ('127.0.0.1', 'localhost', '::1', 'testserver')):
            return JSONResponse({'detail': 'Local access only'}, status_code=403)
        return await call_next(request)

    def manager(request):
        return request.app.state.manager

    @app.get('/api/models')
    def models(request: Request, refresh: bool = False):
        return manager(request).catalogs(refresh)

    @app.get('/api/repos')
    def repos(request: Request):
        return [public_repo(r) for r in manager(request).repositories.values()]

    @app.post('/api/repos', status_code=201)
    def import_repo(body: ImportRepo, request: Request):
        from product_api.imports import import_repository
        m = manager(request)
        try:
            with m.lock:
                row = import_repository(body.url, body.ref, m.import_root)
                m.repositories[row['id']] = row
            return public_repo(row)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from None

    def catalog_source(repo_id, request):
        m = manager(request)
        config = m.repositories.get(repo_id)
        if config is None:
            raise HTTPException(404, 'Repository not found')
        if not config['ready']:
            raise HTTPException(422, config['reason'])
        key = 'repo:' + repo_id
        if key not in m.source_cache:
            m.source_cache[key] = open_repository(config)[0]
        return m.source_cache[key]

    @app.get('/api/repos/{repo_id}/files')
    def repo_files(repo_id: str, request: Request):
        try:
            return {'paths': catalog_source(repo_id, request).files()}
        except (ValueError, OSError):
            raise HTTPException(422, 'Pinned source unavailable') from None

    @app.get('/api/repos/{repo_id}/source')
    def repo_source(repo_id: str, request: Request, path: str, start: int = 1, count: int = 160):
        if start < 1 or not 1 <= count <= 400:
            raise HTTPException(422, 'Invalid source range')
        try:
            repo = catalog_source(repo_id, request)
            text, digest = repo.text(path)
            lines = text.splitlines()
            return {'path': path, 'commit': repo.commit, 'start_line': start,
                    'lines': lines[start - 1:start - 1 + count], 'total_lines': len(lines), 'sha256': digest}
        except (ValueError, OSError, UnicodeError):
            raise HTTPException(422, 'Pinned source unavailable or invalid path') from None

    @app.get('/api/benchmarks')
    def benchmarks(request: Request, repo_id: str | None = None):
        from product_api.benchmarks import public
        rows = manager(request).benchmark_catalog().values()
        return [public(r) for r in rows if repo_id is None or r['repo_id'] == repo_id]

    @app.get('/api/comparisons')
    def comparisons(request: Request):
        m = manager(request)
        paths = sorted((p for p in m.comparison_root.glob('*.json') if not p.name.endswith('.grade.json')), key=lambda p: p.stat().st_mtime, reverse=True)[:30]
        return [m.comparison(p.stem) for p in paths]

    @app.post('/api/comparisons', status_code=201)
    def start_comparison(body: StartComparison, request: Request):
        return manager(request).start_comparison(body)

    @app.get('/api/comparisons/{comparison_id}')
    def comparison(comparison_id: str, request: Request):
        return manager(request).comparison(comparison_id)

    @app.post('/api/comparisons/{comparison_id}/cancel')
    def cancel_comparison(comparison_id: str, request: Request):
        m = manager(request)
        pair = m.comparison(comparison_id)
        if pair.get('benchmark'):
            with m.lock:
                grade_path = m.comparison_root / (comparison_id + '.grade.json')
                grade = read(grade_path) if grade_path.exists() else {'scores':{}}
                if grade.get('status') != 'completed':
                    atomic_write(grade_path, {**grade,'status':'cancelled','error':'Grading stopped by request.'})
        for run in pair['runs']:
            m.cancel(run['id'])
        process = m.grading_processes.get(comparison_id)
        if process and process.poll() is None: process.terminate()
        return m.comparison(comparison_id)

    @app.get('/api/runs')
    def runs(request: Request):
        m = manager(request)
        ids = sorted((d.name for d in m.root.iterdir() if (d / 'config.json').exists()),
                     key=lambda i: (m.root / i).stat().st_mtime, reverse=True)[:30]
        return [m.get(i) for i in ids]

    @app.post('/api/runs', status_code=201)
    def start(body: StartRun, request: Request):
        return manager(request).start(body)

    @app.get('/api/runs/{run_id}')
    def get(run_id: str, request: Request):
        return manager(request).get(run_id)

    @app.post('/api/runs/{run_id}/cancel')
    def cancel(run_id: str, request: Request):
        return manager(request).cancel(run_id)

    @app.get('/api/runs/{run_id}/events')
    async def events(run_id: str, request: Request, after: int = -1):
        m = manager(request)
        directory = m.directory(run_id)
        try:
            cursor = int(request.headers.get('last-event-id', after))
        except ValueError:
            raise HTTPException(422, 'Invalid event cursor') from None
        async def stream():
            nonlocal cursor
            heartbeat = 0
            while not await request.is_disconnected():
                for event in await asyncio.to_thread(event_rows, directory):
                    if event['sequence'] > cursor:
                        cursor = event['sequence']
                        yield f'id: {cursor}\ndata: {json.dumps(event)}\n\n'
                state = await asyncio.to_thread(m.get, run_id)
                if state['status'] in TERMINAL:
                    # Worker writes its final events before atomically publishing result.
                    for event in await asyncio.to_thread(event_rows, directory):
                        if event['sequence'] > cursor:
                            cursor = event['sequence']
                            yield f'id: {cursor}\ndata: {json.dumps(event)}\n\n'
                    yield f'event: terminal\ndata: {json.dumps(state)}\n\n'
                    return
                if heartbeat % 4 == 0:
                    yield f'event: metrics\ndata: {json.dumps(state["metrics"])}\n\n'
                heartbeat += 1
                if heartbeat % 40 == 0:
                    yield ': keepalive\n\n'
                await asyncio.sleep(0.25)
        return StreamingResponse(stream(), media_type='text/event-stream',
            headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})

    @app.get('/api/runs/{run_id}/files')
    def files(run_id: str, request: Request):
        try:
            return {'paths': manager(request).source(run_id).files()}
        except (ValueError, OSError):
            raise HTTPException(422, 'Pinned source unavailable') from None

    @app.get('/api/runs/{run_id}/source')
    def source(run_id: str, request: Request, path: str, start: int = 1, count: int = 160):
        if start < 1 or not 1 <= count <= 400:
            raise HTTPException(422, 'Invalid source range')
        try:
            repo = manager(request).source(run_id)
            text, digest = repo.text(path)
            lines = text.splitlines()
            return {'path': path, 'commit': repo.commit, 'start_line': start,
                    'lines': lines[start - 1:start - 1 + count], 'total_lines': len(lines), 'sha256': digest}
        except (ValueError, OSError, UnicodeError):
            raise HTTPException(422, 'Pinned source unavailable or invalid path') from None

    @app.get('/api/runs/{run_id}/artifacts/{artifact_id}')
    def artifact(run_id: str, artifact_id: str, request: Request):
        from agent_harness.artifacts import ArtifactStore
        m = manager(request)
        directory = m.directory(run_id)
        allowed = {e.get('artifact_id') for e in event_rows(directory) if e['kind'] == 'tool_observation'}
        if artifact_id not in allowed:
            raise HTTPException(404, 'Public tool artifact not found')
        value = ArtifactStore(directory / 'episodes').get(run_id, artifact_id)
        if value.get('error'):
            value['error'] = 'Tool failed; check input or environment availability.'
        return value

    dist = ROOT / 'product_web/dist'
    if dist.exists():
        app.mount('/assets', StaticFiles(directory=dist / 'assets'), name='assets')
        @app.get('/')
        def index():
            return FileResponse(dist / 'index.html')
    return app

app = create_app()
