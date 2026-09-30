"""Comparison contracts and real isolated preview workers; no paid inference."""
import json
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch, AsyncMock
from fastapi.testclient import TestClient
from product_api.app import create_app, StartRun
from product_api.hosted import discover, valid_price
from product_api.metrics import summarize
from tests import test_product as fixtures


class ComparisonTests(unittest.TestCase):
    setUp = fixtures.ProductTests.setUp
    tearDown = fixtures.ProductTests.tearDown

    def body(self, **kwargs):
        return {'repo_id': 'fixture', 'left_model_id': 'preview-left', 'right_model_id': 'preview-right',
                'question': 'How does Example work?', 'preview': True, **kwargs}

    def finish(self, client, pair):
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            value = client.get('/api/comparisons/' + pair['id']).json()
            if value['status'] != 'running': return value
            time.sleep(.1)
        self.fail('Preview workers did not finish')

    def test_concurrent_workers_replay_context_and_restore(self):
        storage = self.root / 'runs'
        with patch('product_api.app.repo_catalog', return_value=[self.repo]):
            with TestClient(create_app(storage)) as client:
                response = client.post('/api/comparisons', json=self.body())
                self.assertEqual(response.status_code, 201, response.text)
                pair = response.json()
                ids = [r['id'] for r in pair['runs']]
                self.assertEqual(len(set(ids)), 2)
                self.assertTrue(all(client.app.state.manager.processes[i].poll() is None for i in ids))
                single = client.post('/api/runs', json={'repo_id':'fixture','question':'Another','preview':True})
                self.assertEqual(single.status_code, 201, single.text)
                second = client.post('/api/comparisons', json=self.body())
                self.assertEqual(second.status_code, 201, second.text)
                self.assertTrue(all(client.app.state.manager.processes[i].poll() is None for i in ids))
                client.post('/api/comparisons/'+second.json()['id']+'/cancel')
                client.post('/api/runs/'+single.json()['id']+'/cancel')
                pair = self.finish(client, pair)
                self.assertEqual([r['status'] for r in pair['runs']], ['completed','completed'])
                self.assertNotEqual(pair['runs'][0]['metrics']['tool_calls'], pair['runs'][1]['metrics']['tool_calls'])
                for run in pair['runs']:
                    replay = client.get(f'/api/runs/{run["id"]}/events').text
                    self.assertIn('tool_call', replay)
                    self.assertIn('event: terminal', replay)
                    self.assertIn('event: terminal', client.get(f'/api/runs/{run["id"]}/events',headers={'Last-Event-ID':'99999'}).text)
                child = client.post('/api/comparisons',json=self.body(question='Follow up',parent_comparison_id=pair['id'])).json()
                for index, run in enumerate(child['runs']):
                    config = json.loads((storage/run['id']/'config.json').read_text())
                    self.assertEqual(config['parent_run_id'],ids[index])
                    self.assertEqual(config['prior_context']['answer'],pair['runs'][index]['answer'])
                client.post('/api/comparisons/'+child['id']+'/cancel')
                self.finish(client, child)
            with TestClient(create_app(storage)) as client:
                restored = client.get('/api/comparisons/'+pair['id']).json()
                self.assertEqual([r['id'] for r in restored['runs']], ids)
                self.assertEqual(restored['status'],'finished')

    def test_one_sided_stop_and_launch_failure(self):
        with patch('product_api.app.repo_catalog',return_value=[self.repo]), TestClient(create_app(self.root/'runs')) as client:
            pair = client.post('/api/comparisons',json=self.body()).json()
            client.post('/api/runs/'+pair['runs'][0]['id']+'/cancel')
            result=self.finish(client,pair)
            self.assertEqual(result['runs'][0]['status'],'cancelled')
            self.assertEqual(result['runs'][1]['status'],'completed')
            import subprocess
            real = subprocess.Popen
            count=0
            def launch(*args,**kwargs):
                nonlocal count
                count+=1
                if count==1: raise OSError('test')
                return real(*args,**kwargs)
            with patch('product_api.app.subprocess.Popen',side_effect=launch):
                pair=client.post('/api/comparisons',json=self.body()).json()
            result=self.finish(client,pair)
            self.assertEqual(result['runs'][0]['status'],'infrastructure_error')
            self.assertEqual(result['runs'][1]['status'],'completed')

    def test_validation_before_launch(self):
        with patch('product_api.app.repo_catalog',return_value=[self.repo]), TestClient(create_app(self.root/'runs')) as client:
            self.assertEqual(client.post('/api/comparisons',json=self.body(right_model_id='preview-left')).status_code,422)
            self.assertEqual(client.post('/api/comparisons',json=self.body(question=' ')).status_code,422)
            models=[{'id':'base','ready':True,'base_model':'Qwen/Qwen3.5-4B'}]
            with patch.object(client.app.state.manager,'catalogs',return_value={'models':models}):
                self.assertEqual(client.post('/api/comparisons',json=self.body(preview=False,left_model_id='base',right_model_id='missing')).status_code,422)
            self.assertEqual(client.app.state.manager.processes,{})
            self.assertEqual(client.get('/api/comparisons').json(),[])

    def test_capacity_reserves_both_sides_and_recovers_after_completion(self):
        with patch('product_api.app.repo_catalog', return_value=[self.repo]), TestClient(create_app(self.root/'capacity')) as client:
            manager = client.app.state.manager
            pending = SimpleNamespace(poll=lambda: None)
            finished = SimpleNamespace(poll=lambda: 0)
            with patch.dict(manager.processes, {'occupied': pending}), patch('product_api.app.MAX_CONCURRENT_RUNS', 2):
                response = client.post('/api/comparisons', json=self.body())
                self.assertEqual(response.status_code, 429)
                self.assertIn('slots',response.json()['detail'])
                self.assertEqual(client.get('/api/comparisons').json(), [])
                self.assertEqual(list((self.root/'capacity').glob('*/config.json')), [])
                manager.processes['occupied'] = finished
            # A running grader does not monopolize investigation slots.
            with patch.dict(manager.grading_processes, {'grading': pending}):
                pair = client.post('/api/comparisons', json=self.body())
                self.assertEqual(pair.status_code, 201, pair.text)
                self.assertEqual(self.finish(client, pair.json())['status'],'finished')

    def test_hosted_397b_discovery(self):
        names=['Qwen/Qwen3.5-4B','Qwen/Qwen3.5-397B-A17B','unknown/model']
        service=SimpleNamespace(get_server_capabilities_async=AsyncMock(return_value=SimpleNamespace(supported_models=[SimpleNamespace(model_name=n,sampleable=True,max_context_length=65536) for n in names])))
        with patch('product_api.hosted.metadata',return_value={names[1]:{'prefill':'$3.00','sample':'$7.50','params':397000000000,'active_params':17000000000}}):
            rows=discover(service,names[0])
        self.assertEqual(rows[0]['id'],'base')
        large=next(r for r in rows if r['base_model']==names[1])
        self.assertTrue(large['ready'])
        self.assertEqual(large['context_tokens'],32768)
        self.assertEqual(large['pricing']['input_per_million'],3)
        self.assertFalse(next(r for r in rows if r['base_model']==names[2])['ready'])

    def test_usage_repairs_partial_and_prices(self):
        config={'created_at':time.time(),'model':{'pricing':{'input_per_million':3,'output_per_million':7.5}}}
        events=[{'kind':'model_request','elapsed_seconds':0},
                {'kind':'invalid_action','elapsed_seconds':2,'usage':{'input_tokens':100,'output_tokens':20,'cost_usd':None}},
                {'kind':'model_request','elapsed_seconds':2},
                {'kind':'model_response','elapsed_seconds':3,'usage':{'input_tokens':200,'output_tokens':30,'cost_usd':None}}]
        metrics=summarize(events,{'status':'completed','metrics':{'elapsed_seconds':3}},config)
        self.assertEqual(metrics['input_tokens'],300)
        self.assertEqual(metrics['steps'],2)
        self.assertEqual(metrics['model_seconds'],3)
        self.assertAlmostEqual(metrics['estimated_cost_usd'],.001275)
        self.assertTrue(metrics['token_usage_complete'])
        partial=summarize(events,{'status':'cancelled','metrics':{'elapsed_seconds':3}},config)
        self.assertFalse(partial['token_usage_complete'])
        self.assertEqual(partial['input_tokens'],300)
        self.assertIsNone(summarize(events,{'status':'completed'}, {'created_at':time.time(),'model':{}})['estimated_cost_usd'])

    def test_restart_marks_both_unfinished_sides_interrupted(self):
        from product_api.worker import atomic_write
        storage = self.root / 'restart'
        comparison_id = 'a' * 32
        with patch('product_api.app.repo_catalog', return_value=[self.repo]):
            with TestClient(create_app(storage)) as client:
                manager = client.app.state.manager
                ids = []
                for side in ('left', 'right'):
                    config = manager.prepare(StartRun(repo_id='fixture', model_id='preview-' + side, question='Question', preview=True), [])
                    config.update(comparison_id=comparison_id, side=side)
                    manager.persist(config)
                    ids.append(config['id'])
                atomic_write(manager.comparison_root / (comparison_id + '.json'), {'id': comparison_id, 'run_ids': ids})
            with TestClient(create_app(storage)) as client:
                pair = client.get('/api/comparisons/' + comparison_id).json()
                self.assertEqual([r['status'] for r in pair['runs']], ['interrupted', 'interrupted'])
                self.assertEqual(client.app.state.manager.processes, {})

    def test_unknown_prices_and_failed_elapsed_do_not_grow(self):
        self.assertFalse(valid_price({'input_per_million':-1,'output_per_million':2}))
        self.assertFalse(valid_price({'input_per_million':float('nan'),'output_per_million':2}))
        config = {'created_at':time.time()-1000, 'model':{}}
        metrics = summarize([], {'status':'infrastructure_error','metrics':{}}, config)
        self.assertEqual(metrics['elapsed_seconds'], 0)
        self.assertIsNone(metrics['cost_usd'])
        self.assertIsNone(metrics['estimated_cost_usd'])
