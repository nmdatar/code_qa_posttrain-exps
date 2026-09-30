"""Product contract and real subprocess lifecycle tests; no paid service calls."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
import unittest
from unittest.mock import patch, MagicMock, AsyncMock
from types import SimpleNamespace
from datetime import datetime, timezone, timedelta
try:
    from fastapi.testclient import TestClient
    from product_api.app import create_app, RunManager, event_rows
    PRODUCT_AVAILABLE = True
except ImportError:
    PRODUCT_AVAILABLE = False
from product_api.catalog import PinnedRepository
from product_api.model import parse_action
from product_api.worker import PublicStore, project
from agent_harness.contracts import ToolCall, FinalAnswer

@unittest.skipUnless(PRODUCT_AVAILABLE, 'Install requirements-product.txt for product tests')
class ProductTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        source = self.root / 'source'
        source.mkdir()
        (source / 'example.py').write_text('class Example:\n    frozen = True\n    value = 1\n')
        for args in (['init', '-q'], ['add', '.'], ['-c','user.name=Test','-c','user.email=test@localhost','-c','commit.gpgsign=false','commit','-qm','fixture']):
            subprocess.run(['git','-C',str(source),*args],check=True,capture_output=True)
        commit = subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
        files = {'example.py': hashlib.sha256((source/'example.py').read_bytes()).hexdigest()}
        manifest = {'commit':commit,'snapshot_files':files}
        manifest_path = self.root/'manifest.json'
        manifest_path.write_text(json.dumps(manifest))
        self.repo = {'id':'fixture','name':'example/fixture','commit':commit,'execution':True,'ready':True,'reason':None,'example':'How does Example work?', 'source_path':str(source),'manifest_path':str(manifest_path)}
        self.manifest = manifest
    def tearDown(self):
        self.temp.cleanup()

    def test_harness_auto_and_explicit_override(self):
        from product_api.app import StartRun
        manager=RunManager(self.root/'runs')
        manager.repositories={'fixture':self.repo}
        models=[{'id':'trained','name':'Bash trained','ready':True,'trained_harness':'bash'}]
        automatic=manager.prepare(StartRun(repo_id='fixture',model_id='trained',question='Question'),models)
        self.assertEqual(automatic['harness'],'bash')
        explicit=manager.prepare(StartRun(repo_id='fixture',model_id='trained',question='Question',harness='structured'),models)
        self.assertEqual(explicit['harness'],'structured')
        self.assertEqual(explicit['model']['harness'],'structured')

    def test_bash_readiness_chunks_large_repository_manifest(self):
        import base64, zlib
        from product_api.harness import create_bash_sandbox
        files = {f'package/file_{i}.py': hashlib.sha256(str(i).encode()).hexdigest() for i in range(4000)}
        digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
        repo = SimpleNamespace(commit='pinned', snapshot_hashes=lambda: files)
        manifest = dict(status='ready', backend='modal', commit='pinned', snapshot_sha256=digest, image_digest='im-test')
        with patch('agent_harness.modal_backend.ModalSandboxBackend') as backend:
            create_bash_sandbox(repo, manifest, self.root, 'episode', {'wall_time_seconds':900,'max_tool_calls':64})
        argv = backend.call_args.args[0]['recipe']['readiness_commands'][0]
        self.assertTrue(all(len(part.encode()) <= 32000 for part in argv))
        self.assertEqual(json.loads(zlib.decompress(base64.b64decode(''.join(argv[5:])))), files)
        compile(argv[4], '<readiness>', 'exec')

    def test_bash_preview_has_only_bash_tool(self):
        from product_api.app import StartRun
        from product_api.worker import main
        manager=RunManager(self.root/'runs')
        manager.repositories={'fixture':self.repo}
        config=manager.prepare(StartRun(repo_id='fixture',question='Question',preview=True,harness='bash'),[])
        manager.persist(config)
        directory=manager.root/config['id']
        # Use the actual worker entry point in a subprocess, without provider calls.
        subprocess.run([os.sys.executable,'-m','product_api.worker',str(directory)],check=True,timeout=15)
        result=json.loads((directory/'result.json').read_text())
        self.assertEqual(result['status'],'completed')
        events=[json.loads(line) for line in (directory/'events.jsonl').read_text().splitlines()]
        started=next(e for e in events if e['kind']=='started')
        self.assertEqual([t['name'] for t in started['tools']],['bash'])
        self.assertEqual([e['name'] for e in events if e['kind']=='tool_call'],['bash'])

    def test_json_protocol_and_invalid_actions(self):
        call = parse_action('{"tool":"read_file","arguments":{"path":"a.py","start_line":4,"end_line":8}}')
        self.assertIsInstance(call, ToolCall)
        self.assertEqual(call.arguments['line_count'],5)
        self.assertIsInstance(parse_action('```json\n{"answer":"Done","citations":[]}\n```'),FinalAnswer)
        for text in ('[]','{"tool":"read_file","arguments":{"end_line":999}}','{"answer":NaN,"citations":[]}','{"answer":"","citations":[]}'):
            with self.assertRaises(ValueError): parse_action(text)

    def test_catalog_exposes_all_environment_revisions(self):
        from product_api.catalog import repo_catalog
        release=self.root/'release'
        (release/'public').mkdir(parents=True)
        rows=[]
        for index,name in enumerate(('org/one','org/two','org/three')):
            env_id=f'env-{index}'
            rows.append({'environment_id':env_id,'capability':'source_reading',
                'repository':{'family_id':name,'url':'https://github.com/'+name,'commit':self.repo['commit']}})
            evidence=release/'private/environment-evidence'/env_id
            evidence.mkdir(parents=True)
            (evidence/'source-environment.json').write_text(json.dumps({'snapshot_path':self.repo['source_path']}))
            (evidence/'result.json').write_text(json.dumps(self.manifest))
        (release/'public/environments.jsonl').write_text('\n'.join(map(json.dumps,rows)))
        with patch('product_api.catalog.RELEASE',release): catalog=repo_catalog()
        self.assertEqual({r['name'] for r in catalog},{'org/one','org/two','org/three'})
        self.assertTrue(all(r['ready'] for r in catalog))

    def test_token_context_compaction_preserves_latest_evidence_and_question(self):
        from product_api.model import ProductModel
        from agent_harness.contracts import ModelActionError
        model=ProductModel({'base_model':'test/model'})
        def prompt(messages):
            size=sum(len(m['content']) for m in messages)
            if size>=1000: raise ValueError('Context budget exceeded')
            return [1]*size
        model.renderer=SimpleNamespace(prompt=prompt,context_limit=1000)
        messages=[{'role':'system','content':'instructions'}, {'role':'user','content':'original question'},
            {'role':'assistant','content':'old tool action'},
            {'role':'tool','content':json.dumps({'text':'x'*1000,'artifact_id':'saved-output','status':'ok'})},
            {'role':'assistant','content':'recent read action'},
            {'role':'tool','content':'{"text":"latest source evidence"}'}]
        converted=json.loads(json.dumps(messages))
        tokens=model._fit_prompt(converted,messages,200)
        self.assertLessEqual(len(tokens)+200,1000)
        self.assertEqual(converted[:2],messages[:2])
        self.assertEqual(converted[-2:],messages[-2:])
        self.assertIn('saved-output',converted[3]['content'])
        self.assertIn('x'*1000,messages[3]['content'])
        # An irreducibly large question is a known local budget limit, not a provider failure.
        huge=[{'role':'system','content':'system'},{'role':'user','content':'x'*1200}]
        with self.assertRaises(ModelActionError) as caught:
            model._fit_prompt(huge,huge,200)
        self.assertEqual(caught.exception.termination_reason,'budget_exhausted')
        self.assertEqual(caught.exception.usage.output_tokens,0)

    def test_action_repair_preserves_history_usage_and_limits(self):
        from agent_harness.runner import AgentRunner
        from agent_harness.registry import ToolRegistry
        from agent_harness.contracts import ModelActionError, ModelResponse, ResearchRequest, RunLimits, Usage
        calls=[]
        class Model:
            def generate(self, messages, tools, max_output_tokens, **kwargs):
                calls.append((json.loads(json.dumps(messages)),max_output_tokens))
                if len(calls)==1:
                    raise ModelActionError('Invalid JSON',Usage(3,5,0),
                        raw_response='{"answer":"oops"}',repair_feedback='Missing citations array.')
                return ModelResponse(FinalAnswer('fixed'),usage=Usage(4,2,0))
        store=PublicStore(self.root/'repair',self.root/'repair.jsonl')
        result=AgentRunner(Model(),ToolRegistry(),store,max_action_repairs=2).run(
            ResearchRequest('repair','Question',{},limits=RunLimits(max_output_tokens=20)))
        self.assertEqual(result.termination_reason,'completed')
        self.assertEqual(result.metrics['output_tokens'],7)
        self.assertEqual(calls[1][1],15)
        self.assertEqual(calls[1][0][-2:],[
            {'role':'assistant','content':'{"answer":"oops"}'},
            {'role':'user','content':'Missing citations array.'}])
        events=[json.loads(line) for line in Path(result.trajectory_path).read_text().splitlines()]
        invalid=next(e for e in events if e['kind']=='invalid_action')
        self.assertEqual(invalid['raw_response'],'{"answer":"oops"}')
        self.assertNotIn('raw_response',project(invalid))
        class AlwaysInvalid:
            def generate(self,*args,**kwargs):
                raise ModelActionError('Invalid JSON',Usage(1,5,0),repair_feedback='Fix JSON.')
        for budget,expected_steps,reason in [(100,3,'agent_error'),(5,1,'budget_exhausted')]:
            result=AgentRunner(AlwaysInvalid(),ToolRegistry(),store,max_action_repairs=2).run(
                ResearchRequest('bounded','Question',{},limits=RunLimits(max_output_tokens=budget)))
            self.assertEqual(result.metrics['steps'],expected_steps)
            self.assertEqual(result.termination_reason,reason)
        from agent_harness.code_tools import code_understanding_tools
        class Intermittent:
            calls=0
            def generate(self,*args,**kwargs):
                self.calls+=1
                if self.calls in (1,3):
                    raise ModelActionError('Bad JSON',Usage(1,1,0),repair_feedback='Fix JSON.')
                action=ToolCall('list_files',{}) if self.calls==2 else FinalAnswer('done')
                return ModelResponse(action,usage=Usage(1,1,0))
        repo=PinnedRepository(self.repo['source_path'],self.repo['commit'],self.manifest)
        result=AgentRunner(Intermittent(),ToolRegistry(code_understanding_tools()),store,max_action_repairs=1).run(
            ResearchRequest('intermittent','Question',{'repository':repo}))
        self.assertEqual(result.termination_reason,'completed')
        self.assertEqual(result.metrics['steps'],4)

    def test_repository_browsing_does_not_require_a_run(self):
        with patch('product_api.app.repo_catalog',return_value=[self.repo]), TestClient(create_app(self.root/'runs')) as client:
            self.assertEqual(client.get('/api/repos/fixture/files').json()['paths'],['example.py'])
            source=client.get('/api/repos/fixture/source',params={'path':'example.py'})
            self.assertEqual(source.json()['commit'],self.repo['commit'])
            self.assertEqual(client.get('/api/repos/fixture/source',params={'path':'../manifest.json'}).status_code,422)
            self.assertEqual(client.get('/api/repos/missing/files').status_code,404)
            self.assertEqual(client.get('/api/runs').json(),[])

    def test_followup_pins_parent_and_retains_context(self):
        with patch('product_api.app.repo_catalog',return_value=[self.repo]), TestClient(create_app(self.root/'runs')) as client:
            parent=client.post('/api/runs',json={'repo_id':'fixture','question':'Original question','preview':True}).json()
            client.get(f'/api/runs/{parent["id"]}/events')
            result=client.post('/api/runs',json={'repo_id':'fixture','model_id':'preview','question':'Follow up','preview':True,'parent_run_id':parent['id']})
            self.assertEqual(result.status_code,201)
            child=result.json()
            self.assertEqual(child['parent_run_id'],parent['id'])
            config=json.loads((self.root/'runs'/child['id']/'config.json').read_text())
            self.assertEqual(config['prior_context']['question'],'Original question')
            client.get(f'/api/runs/{child["id"]}/events')
            mismatch=client.post('/api/runs',json={'repo_id':'fixture','model_id':'base','question':'Follow up','parent_run_id':parent['id']})
            self.assertEqual(mismatch.status_code,422)

    def test_claim_mapping_requires_exact_answer_text(self):
        store=PublicStore(self.root/'claims',self.root/'claim-events.jsonl')
        store.observed={'example.py':{1,2}}
        refs=[dict(path='example.py',start_line=1,end_line=2,claim='Example is frozen.'),
              dict(path='example.py',start_line=1,end_line=2,claim='Invented claim')]
        answer=store.validate_answer({'text':'Example is frozen.','citations':refs})
        self.assertEqual(answer['citations'][0]['claim'],'Example is frozen.')
        self.assertNotIn('claim',answer['citations'][1])

    def test_import_rejects_non_github_and_option_refs(self):
        from product_api.imports import import_repository
        for url,ref in [('file:///tmp/repo','HEAD'),('https://example.com/repo','HEAD'),('https://github.com/org/repo','--upload-pack=evil')]:
            with self.assertRaises(ValueError): import_repository(url,ref,self.root/'imports')

    def test_unsupported_answer_research_repair_is_bounded(self):
        from agent_harness.runner import AgentRunner
        from agent_harness.models import ScriptedModel
        from agent_harness.registry import ToolRegistry
        from agent_harness.code_tools import code_understanding_tools
        from agent_harness.contracts import ResearchRequest,ModelResponse,Usage
        repo=PinnedRepository(self.repo['source_path'],self.repo['commit'],self.manifest)
        store=PublicStore(self.root/'episodes',self.root/'events.jsonl')
        answer=FinalAnswer({'text':'It is frozen.','citations':[{'path':'example.py','start_line':2,'end_line':2}]})
        actions=[answer,ToolCall('read_file',{'path':'example.py','start_line':1,'line_count':3}),answer]
        model=ScriptedModel([ModelResponse(a,usage=Usage(10,5,0)) for a in actions])
        request=ResearchRequest('test','How does Example work?',{'repository':repo})
        result=AgentRunner(model,ToolRegistry(code_understanding_tools()),store,
            submission_feedback=store.answer_feedback).run(request)
        self.assertEqual(result.termination_reason,'completed')
        self.assertEqual(result.metrics['tool_calls'],1)
        self.assertEqual(result.metrics['output_tokens'],15)
        self.assertEqual(len([e for e in event_rows(self.root) if e['kind']=='submission_rejected']),1)
        failing=PublicStore(self.root/'fail',self.root/'failed.jsonl')
        model=ScriptedModel([ModelResponse(answer,usage=Usage(1,1,0)) for _ in range(4)])
        result=AgentRunner(model,ToolRegistry(),failing,submission_feedback=failing.answer_feedback).run(request)
        self.assertEqual(result.termination_reason,'agent_error')
        self.assertIsNone(result.submission)
        self.assertEqual(result.metrics['steps'],3)

    def test_credentials_accept_saved_sdk_login_without_exposing_key(self):
        from product_api.catalog import credentials_available
        store=MagicMock()
        store.get_default_key.return_value=SimpleNamespace(key='test-only-secret')
        credentials=SimpleNamespace(JsonCredentialStore=lambda path:store,
                                    default_credentials_path=lambda:self.root/'credentials.json')
        with patch.dict(os.environ,{},clear=True), patch.dict('sys.modules',{'tinker.lib.credentials':credentials}):
            self.assertIs(credentials_available(),True)
            store.get_default_key.return_value=None
            self.assertIs(credentials_available(),False)
        with patch.dict(os.environ,{'TINKER_API_KEY':'test-only-secret'},clear=True):
            self.assertIs(credentials_available(),True)
        with patch.dict(os.environ,{'TINKER_CREDENTIAL_CMD':'configured-command'},clear=True):
            self.assertIs(credentials_available(),True)

    def test_projection_excludes_training_and_prompt_data(self):
        event = {'sequence':0,'elapsed_seconds':1,'kind':'model_request','messages':[{'content':'SECRET'}],'token_ids':[1]}
        self.assertNotIn('SECRET',json.dumps(project(event)))
        error = project({**event,'kind':'error','error':'secret API header'})
        self.assertNotIn('header',json.dumps(error))

    def test_citations_require_observed_complete_lines(self):
        store = PublicStore(self.root/'episodes',self.root/'public.jsonl')
        store.create_episode('test')
        store.append_event('test',{'sequence':0,'elapsed_seconds':0,'kind':'tool_observation','call_id':'c','name':'read_file','observation':{'status':'ok','content':{'path':'a.py','start_line':2,'text':'complete\npartial'},'truncated':True}})
        answer = store.validate_answer({'text':'Answer','citations':[{'path':'a.py','start_line':2,'end_line':2},{'path':'a.py','start_line':2,'end_line':3},{'path':'b.py','start_line':1,'end_line':1}]})
        self.assertEqual([c['verified'] for c in answer['citations']],[True,False,False])

    def test_source_is_pinned_and_rejects_traversal(self):
        repo = PinnedRepository(self.repo['source_path'],self.repo['commit'],self.manifest)
        (Path(self.repo['source_path'])/'example.py').write_text('modified')
        self.assertIn('frozen',repo.text('example.py')[0])
        for path in ('../manifest.json','/etc/passwd','missing.py'):
            with self.assertRaises(ValueError): repo.blob(path)
        corrupt = {**self.manifest,'snapshot_files':{'example.py':'0'*64}}
        with self.assertRaises(ValueError): PinnedRepository(self.repo['source_path'],self.repo['commit'],corrupt).blob('example.py')

    def test_preview_subprocess_events_source_artifacts_and_replay(self):
        with patch('product_api.app.repo_catalog',return_value=[self.repo]), TestClient(create_app(self.root/'runs')) as client:
            with self.assertRaises(RuntimeError):
                RunManager(self.root/'runs')
            r = client.post('/api/runs',json={'repo_id':'fixture','question':'How does it work?','preview':True})
            self.assertEqual(r.status_code,201,r.text)
            run_id = r.json()['id']
            other = client.post('/api/runs',json={'repo_id':'fixture','question':'Another?','preview':True})
            self.assertEqual(other.status_code,201,other.text)
            self.assertNotEqual(other.json()['id'],run_id)
            stream = client.get(f'/api/runs/{run_id}/events').text
            result = client.get(f'/api/runs/{run_id}').json()
            self.assertEqual(result['status'],'completed',result)
            self.assertTrue(result['answer']['citations'][0]['verified'])
            self.assertIn('SCRIPTED PREVIEW',stream)
            self.assertIn('event: terminal',stream)
            events = event_rows(self.root/'runs'/run_id)
            ids = [e['sequence'] for e in events]
            self.assertEqual(ids,sorted(set(ids)))
            last = ids[-1]
            replay = client.get(f'/api/runs/{run_id}/events',headers={'Last-Event-ID':str(last)}).text
            self.assertNotIn('\nid:',replay)
            self.assertIn('event: terminal',replay)
            source = client.get(f'/api/runs/{run_id}/source',params={'path':'example.py'})
            self.assertEqual(source.status_code,200)
            self.assertEqual(source.json()['commit'],self.repo['commit'])
            self.assertEqual(client.get(f'/api/runs/{run_id}/source',params={'path':'../manifest.json'}).status_code,422)
            observation = next(e for e in events if e['kind']=='tool_observation')
            self.assertEqual(client.get(f'/api/runs/{run_id}/artifacts/{observation["artifact_id"]}').status_code,200)
            self.assertEqual(client.get(f'/api/runs/{run_id}/artifacts/{"0"*64}').status_code,404)
            self.assertEqual(client.post('/api/runs',json={'repo_id':'fixture','question':'x','preview':True},headers={'Origin':'https://evil.example'}).status_code,403)
            self.assertEqual(client.get('/api/repos',headers={'Host':'evil.example'}).status_code,403)

    def test_cancel_and_restart(self):
        with patch('product_api.app.repo_catalog',return_value=[self.repo]), TestClient(create_app(self.root/'runs')) as client:
            run = client.post('/api/runs',json={'repo_id':'fixture','question':'Question','preview':True}).json()
            # Allow worker to register its signal handler.
            time.sleep(.4)
            client.post(f'/api/runs/{run["id"]}/cancel')
            result = None
            for _ in range(50):
                result=client.get(f'/api/runs/{run["id"]}').json()
                if result['status']!='running': break
                time.sleep(.1)
            self.assertEqual(result['status'],'cancelled',result)
        orphan=self.root/'runs'/('a'*32)
        orphan.mkdir()
        (orphan/'config.json').write_text('{}')
        with patch('product_api.app.repo_catalog',return_value=[]):
            recovered = RunManager(self.root/'runs')
            recovered.shutdown()
        self.assertEqual(json.loads((orphan/'result.json').read_text())['status'],'interrupted')

    def test_immediate_stop_and_worker_crash(self):
        with patch('product_api.app.repo_catalog',return_value=[self.repo]), TestClient(create_app(self.root/'runs')) as client:
            run = client.post('/api/runs',json={'repo_id':'fixture','question':'Question','preview':True}).json()
            client.post(f'/api/runs/{run["id"]}/cancel')
            client.get(f'/api/runs/{run["id"]}/events')
            self.assertEqual(client.get(f'/api/runs/{run["id"]}').json()['status'],'cancelled')
            run = client.post('/api/runs',json={'repo_id':'fixture','question':'Question','preview':True}).json()
            client.app.state.manager.processes[run['id']].kill()
            client.get(f'/api/runs/{run["id"]}/events')
            self.assertEqual(client.get(f'/api/runs/{run["id"]}').json()['status'],'infrastructure_error')

    def test_catalog_pagination_expiry_and_project_scope(self):
        from product_api.catalog import model_catalog
        directory = self.root/'artifacts/project/checkpoints'
        directory.mkdir(parents=True)
        path='tinker://project-run/sampler_weights/one'
        (directory/'one.json').write_text(json.dumps({'identity':{'renderer':'hf-chat-no-thinking-v1','base_model':'test/model'},'artifacts':{'sampler':path},'config':{'run_id':'project','environment':{'solver_tools':'bash-only-v1'}}}))
        (directory/'fake.json').write_text(json.dumps({'identity':{'renderer':'hf-chat-no-thinking-v1','base_model':'test/model'},'artifacts':{'sampler':'tinker://fake-run/sampler_weights/fake'}}))
        now=datetime.now(timezone.utc)
        def checkpoint(name, kind='sampler', expired=False, run='project-run'):
            return SimpleNamespace(tinker_path=f'tinker://{run}/{"sampler_weights" if kind=="sampler" else "weights"}/{name}', checkpoint_id=name, checkpoint_type=kind,time=now,expires_at=now-timedelta(days=1) if expired else None)
        pages=[SimpleNamespace(checkpoints=[checkpoint('one'),checkpoint('expired',expired=True)],cursor=SimpleNamespace(offset=0,limit=2,total_count=4)),SimpleNamespace(checkpoints=[checkpoint('training',kind='training'),checkpoint('unrelated',run='other-run'),checkpoint('sibling')],cursor=SimpleNamespace(offset=2,limit=2,total_count=4))]
        rest=MagicMock()
        rest.list_user_checkpoints.side_effect=[SimpleNamespace(result=lambda timeout, p=p:p) for p in pages]
        sdk=SimpleNamespace(ServiceClient=lambda **kwargs:SimpleNamespace(create_rest_client=lambda:rest, get_server_capabilities_async=AsyncMock(return_value=SimpleNamespace(supported_models=[]))))
        with patch('product_api.catalog.ROOT',self.root),patch.dict(os.environ,{'TINKER_API_KEY':'test-key'}),patch.dict('sys.modules',{'tinker':sdk}):
            models,warning=model_catalog()
        self.assertIsNone(warning)
        by_name={r['name']:r for r in models}
        self.assertTrue(by_name['one']['ready'])
        # Both registered and remotely discovered checkpoints retain their training tools.
        from product_api.app import StartRun
        manager = RunManager(self.root/'runs')
        manager.repositories = {'fixture': self.repo}
        for name in ('one', 'sibling'):
            self.assertEqual(by_name[name]['trained_harness'], 'bash')
            config = manager.prepare(StartRun(repo_id='fixture', model_id=by_name[name]['id'], question='Question'), models)
            self.assertEqual(config['model']['harness'], 'bash')
        self.assertNotIn('expired',by_name)
        self.assertNotIn('training',by_name)
        self.assertNotIn('fake',by_name)
        self.assertNotIn('unrelated',by_name)
        self.assertEqual(rest.list_user_checkpoints.call_args_list[1].kwargs['offset'],2)
        with patch('product_api.catalog.ROOT',self.root),patch('product_api.catalog.credentials_available',return_value=False):
            models,_=model_catalog()
            self.assertEqual([m['id'] for m in models],['base'])
        rest.list_user_checkpoints.side_effect=RuntimeError('Unavailable')
        with patch('product_api.catalog.ROOT',self.root),patch('product_api.catalog.credentials_available',return_value=True),patch.dict('sys.modules',{'tinker':sdk}):
            models,warning=model_catalog()
            self.assertEqual([m['id'] for m in models],['base'])
            self.assertIsNotNone(warning)

    def test_inference_uses_checkpoint_without_base_fallback(self):
        from product_api.model import ProductModel
        from agent_harness.contracts import ModelError
        service=MagicMock()
        service.create_sampling_client.side_effect=RuntimeError('secret provider header')
        sdk=SimpleNamespace(ServiceClient=lambda **kwargs:service)
        retry=SimpleNamespace(RetryConfig=lambda **kwargs:kwargs)
        model=ProductModel({'base_model':'test/model','model_path':'tinker://project/sampler_weights/one'})
        with patch.dict('sys.modules',{'tinker':sdk,'tinker.lib.retry_handler':retry}):
            with self.assertRaises(ModelError) as caught:
                model.generate([{'role':'system','content':'system'}],[],100)
        self.assertNotIn('secret',str(caught.exception))
        self.assertEqual(service.create_sampling_client.call_count,1)
        self.assertEqual(service.create_sampling_client.call_args.kwargs['model_path'],'tinker://project/sampler_weights/one')
        self.assertNotIn('base_model',service.create_sampling_client.call_args.kwargs)

    def test_inference_round_trip_translates_tool_history_and_usage(self):
        from product_api.model import ProductModel
        from agent_harness.code_tools import code_understanding_tools
        model=ProductModel({'base_model':'test/model','repository':{'name':'org/repo','commit':'abc'}})
        tokenizer=SimpleNamespace(eos_token_id=0,decode=lambda tokens,**kwargs:'{"answer":"Found the source","citations":[{"path":"example.py","start_line":1,"end_line":2}]}')
        renderer=MagicMock()
        renderer.tokenizer=tokenizer
        renderer.context_limit=8192
        renderer.prompt.return_value=[1,2,3]
        model.renderer=renderer
        sample=SimpleNamespace(sequences=[SimpleNamespace(tokens=[4,5],stop_reason='stop')])
        model.sampling=MagicMock()
        model.sampling.sample.return_value.result.return_value=sample
        model.sdk=SimpleNamespace(ModelInput=SimpleNamespace(from_ints=lambda x:x),SamplingParams=lambda **kwargs:kwargs)
        response=model.generate([{'role':'system','content':'Budgets'}, {'role':'user','content':'Question'},
            {'role':'assistant','tool_calls':[{'function':{'name':'read_file','arguments':'{"path":"example.py","start_line":1,"line_count":2}'}}]},
            {'role':'tool','content':'{"text":"source"}'}],[tool.spec for tool in code_understanding_tools()],100)
        self.assertIsInstance(response.action,FinalAnswer)
        self.assertEqual(response.usage.output_tokens,2)
        history=renderer.prompt.call_args.args[0]
        self.assertIn('org/repo',history[0]['content'])
        self.assertIn('no URL or cloning is required',history[0]['content'])
        self.assertEqual(json.loads(history[2]['content'])['arguments']['end_line'],2)
        self.assertEqual(history[3]['role'],'user')
        self.assertIsNone(response.usage.cost_usd)
        from agent_harness.contracts import ModelActionError
        tokenizer.decode=lambda tokens,**kwargs:'{"answer":"Missing citations"}'
        with self.assertRaises(ModelActionError) as caught:
            model.generate([{'role':'system','content':'Budgets'}],[],100)
        self.assertEqual(caught.exception.raw_response,'{"answer":"Missing citations"}')
        self.assertIn('separate JSON array',caught.exception.repair_feedback)
        self.assertEqual(caught.exception.usage.output_tokens,2)

if __name__=='__main__': unittest.main()
