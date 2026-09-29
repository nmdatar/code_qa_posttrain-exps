import json
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from training_eval.checkpoints import LocalCheckpointStore
from training_eval.contracts import BackendArtifacts, LoopState, ModelIdentity, RunSpec, StageSpec
from training_eval.tracking import JsonTracker, WandbSink


class StorageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name).resolve()
        self.spec = RunSpec('run', (StageSpec('sft', 2, .001),))
        self.backend = SimpleNamespace(identity=ModelIdentity('fake', 'base', 'tok-v1', 'render-v1'),
            save=lambda name: BackendArtifacts(f'train/{name}', f'sample/{name}'),
            verify_artifacts=lambda artifacts: None)

    def commit(self, store, **kwargs):
        return store.commit(self.backend, run_spec=self.spec,
            state=kwargs.pop('state', LoopState()), bindings=kwargs.pop('bindings', {}),
            parent=kwargs.pop('parent', None), **kwargs)

    def test_checkpoint_roundtrip_portable_and_immutable(self):
        store = LocalCheckpointStore(self.directory / 'checkpoints')
        state = LoopState(stage=1, optimizer_step=7, data_cursor=12,
            random_state=random.Random(42).getstate(), strategy_state={'schedule': [1, 2]})
        first = self.commit(store, state=state)
        content = Path(first).read_bytes()
        state.optimizer_step = 9
        second = self.commit(store, state=state, parent=first)
        self.assertNotEqual(first, second)
        self.assertEqual(Path(first).read_bytes(), content)
        manifest = LocalCheckpointStore(self.directory / 'other').read(first)
        self.assertEqual(manifest['state']['optimizer_step'], 7)
        self.assertEqual(manifest['state']['data_cursor'], 12)
        self.assertEqual(manifest['state']['strategy_state'], {'schedule': [1, 2]})
        self.assertEqual(store.read(second)['parent'], first)
        self.assertEqual(manifest['state']['random_state'][0], state.random_state[0])

    def test_failed_verification_does_not_publish(self):
        store = LocalCheckpointStore(self.directory)
        first = self.commit(store)
        def fail(artifacts):
            raise OSError('sampling snapshot unavailable')
        self.backend.verify_artifacts = fail
        with self.assertRaises(OSError):
            self.commit(store)
        self.assertEqual(list(self.directory.glob('*.json')), [Path(first)])
        self.assertEqual(list(self.directory.glob('*.pending')), [])

    def test_partial_backend_save_does_not_publish(self):
        def fail(name):
            raise OSError('training saved but sampler failed')
        self.backend.save = fail
        with self.assertRaises(OSError):
            self.commit(LocalCheckpointStore(self.directory))
        self.assertEqual(list(self.directory.iterdir()), [])

    def test_manifest_rejects_bad_versions_and_artifact_purposes(self):
        store = LocalCheckpointStore(self.directory)
        reference = self.commit(store)
        manifest = store.read(reference)
        for mutation in ({'schema_version': 99}, {'artifacts': {
                'training': 'tinker://run/sampler_weights/a',
                'sampling': 'tinker://run/weights/a'}}):
            bad = self.directory / 'invalid.json'
            bad.write_text(json.dumps({**manifest, **mutation}))
            with self.assertRaises(ValueError):
                store.read(str(bad))

    def test_checkpoint_credentials_rejected_before_backend_save(self):
        def unexpected(name):
            self.fail('Backend must not save invalid config')
        self.backend.save = unexpected
        with self.assertRaises(ValueError):
            self.commit(LocalCheckpointStore(self.directory), bindings={'nested': {'api_key': 'secret'}})

    def test_outage_keeps_events_artifacts_and_replays_once(self):
        sink = SimpleNamespace(available=False, received=[])
        def emit(event):
            if not sink.available:
                raise ConnectionError('outage')
            sink.received.append(event)
        sink.emit, sink.close = emit, lambda: None
        tracker = JsonTracker(self.directory, 'run', sink=sink)
        event_id = tracker.log('metrics', {'loss': 1.2, 'api_key': 'secret'}, step=4)
        path = tracker.artifact('full-rollout', {'turns': [{'tokens': [1, 2]}], 'reward': 1})
        self.assertEqual(json.loads(Path(path).read_text())['turns'][0]['tokens'], [1, 2])
        events = [json.loads(line) for line in tracker.events_path.read_text().splitlines()]
        self.assertEqual(events[0]['event_id'], event_id)
        self.assertEqual(events[0]['payload']['api_key'], '[REDACTED]')
        self.assertEqual(sink.received, [])
        sink.available = True
        tracker.replay()
        tracker.replay()
        tracker.close()
        self.assertEqual([event['event_id'] for event in sink.received], [e['event_id'] for e in events])
        self.assertEqual(tracker.artifact('same-content', {'reward': 1, 'turns': [{'tokens': [1, 2]}]}), path)

    def test_existing_spool_replays_with_original_identity(self):
        tracker = JsonTracker(self.directory, 'run')
        event_id = tracker.log('metrics', {'reward': 0})
        received = []
        restored = JsonTracker(self.directory, 'run', sink=SimpleNamespace(emit=received.append, close=lambda: None))
        restored.replay()
        self.assertEqual(received[0]['event_id'], event_id)
        with self.assertRaises(ValueError):
            JsonTracker(self.directory, 'other-run')

    def test_wandb_lazy_init_streaming_table_and_artifact(self):
        records, artifacts, options = [], [], []
        run = SimpleNamespace(log=records.append, log_artifact=artifacts.append, finish=lambda: None)
        def initialize(**kwargs):
            options.append(kwargs)
            return run
        module = SimpleNamespace(init=initialize, Table=lambda **kwargs: kwargs,
            Artifact=lambda **kwargs: SimpleNamespace(add_file=lambda path: None, **kwargs))
        sink = WandbSink('project', 'run', mode='offline', config={'api_key': 'hidden'}, wandb_module=module)
        self.assertEqual(options, [])
        tracker = JsonTracker(self.directory, 'run', sink=sink)
        tracker.log('metrics', {'loss': .5}, step=1)
        tracker.log('rollout', {'task_id': 'task', 'episode_id': 'ep', 'answer': 'hi', 'reward': 1, 'reason': 'correct', 'metrics': {'tokens': 3}})
        tracker.artifact('trajectory', {'observations': ['hello'], 'answer': 'hi'})
        tracker.close()
        self.assertEqual(options[0]['mode'], 'offline')
        self.assertEqual(options[0]['config']['api_key'], '[REDACTED]')
        self.assertEqual(records[0]['metrics/loss'], .5)
        self.assertEqual(records[1]['rollouts']['columns'][0], 'task_id')
        self.assertEqual(records[1]['rollouts']['data'][0][0], 'task')
        self.assertIn('correct', records[1]['rollouts']['data'][0])
        self.assertEqual(records[1]['rollout/metrics/tokens'], 3)
        self.assertEqual(len(artifacts), 1)
        self.assertEqual(len(records), 3)
        restarted = JsonTracker(self.directory, 'run', sink=WandbSink(
            'project', 'run', mode='offline', wandb_module=module))
        restarted.replay()
        self.assertEqual(len(records), 3)

    def test_wandb_online_entity_routes_and_separates_receipts(self):
        options = []
        run = SimpleNamespace(log=lambda record: None, finish=lambda: None)
        def initialize(**kwargs):
            options.append(kwargs)
            return run
        module = SimpleNamespace(init=initialize)
        sink = WandbSink('project', 'run', mode='online', entity='team-a', wandb_module=module,
                         group='experiment-1', name='seed-42', job_type='evaluation',
                         tags=['baseline'], notes='Held-out evaluation')
        tracker = JsonTracker(self.directory, 'run', sink=sink)
        tracker.log('metrics', {'loss': .5})
        tracker.close()
        self.assertEqual(options[0]['entity'], 'team-a')
        self.assertEqual(options[0]['group'], 'experiment-1')
        self.assertEqual(options[0]['name'], 'seed-42')
        self.assertEqual(options[0]['job_type'], 'evaluation')
        self.assertEqual(options[0]['tags'], ['baseline'])
        self.assertEqual(options[0]['notes'], 'Held-out evaluation')
        self.assertEqual(options[0]['mode'], 'online')
        other = WandbSink('project', 'run', mode='online', entity='team-b', wandb_module=module)
        self.assertNotEqual(sink.delivery_key, other.delivery_key)
        replay = JsonTracker(self.directory, 'run', sink=other)
        replay.replay()
        replay.close()
        self.assertEqual(options[1]['entity'], 'team-b')

    def test_wandb_bounded_rollout_tables_keep_full_local_log(self):
        records = []
        run = SimpleNamespace(log=records.append, finish=lambda: None)
        module = SimpleNamespace(init=lambda **kwargs: run, Table=lambda **kwargs: kwargs)
        sink = WandbSink('project', 'run', wandb_module=module, max_rollouts_per_step=1)
        tracker = JsonTracker(self.directory, 'run', sink=sink)
        tracker.log('rollout', {'episode_id': 'ep1'}, step=2)
        tracker.log('rollout', {'episode_id': 'ep2'}, step=2)
        tracker.log('rollout', {'episode_id': 'ep3'}, step=3)
        self.assertIn('rollouts', records[0])
        self.assertNotIn('rollouts', records[1])
        self.assertIn('rollouts', records[2])
        self.assertEqual(len(tracker.events_path.read_text().splitlines()), 3)
        tracker.log('checkpoint', {'reference': '/local/private/checkpoint-id.json'}, step=3)
        self.assertEqual(records[3]['checkpoint/id'], 'checkpoint-id')
        self.assertNotIn('/local/private', json.dumps(records[3]))

    def test_wandb_failure_storm_backoff_and_replay(self):
        now, attempts, records = [100.0], [], []
        run = SimpleNamespace(log=records.append, finish=lambda: None)
        def initialize(**options):
            attempts.append(options)
            if len(attempts) == 1:
                raise ConnectionError('localhost unavailable')
            return run
        module = SimpleNamespace(init=initialize)
        sink = WandbSink('project', 'run', wandb_module=module,
                         directory=self.directory / 'wandb-output', clock=lambda: now[0])
        tracker = JsonTracker(self.directory / 'events', 'run', sink=sink)
        for index in range(100):
            tracker.log('metrics', {'loss': index}, step=index)
        tracker.replay()
        self.assertEqual(len(attempts), 1)
        self.assertEqual(len(tracker.sink_errors), 2)
        self.assertEqual(len(tracker._delivered[0]), 0)
        now[0] += 31
        tracker.replay()
        self.assertEqual(len(attempts), 2)
        self.assertEqual(len(records), 100)
        self.assertEqual(attempts[-1]['dir'], str(self.directory / 'wandb-output'))
        tracker.replay()
        self.assertEqual(len(records), 100)

    def test_wandb_delivery_failure_backoff(self):
        now, attempts = [100.0], []
        def log(record):
            attempts.append(record)
            if len(attempts) == 1:
                raise ConnectionError('delivery failed')
        module = SimpleNamespace(init=lambda **options: SimpleNamespace(log=log))
        sink = WandbSink('project', 'run', wandb_module=module, clock=lambda: now[0])
        tracker = JsonTracker(self.directory, 'run', sink=sink)
        tracker.log('metrics', {'loss': 1})
        tracker.log('metrics', {'loss': 2})
        self.assertEqual(len(attempts), 1)
        now[0] += 31
        tracker.replay()
        self.assertEqual(len(attempts), 3)
        self.assertEqual(len(tracker._delivered[0]), 2)


if __name__ == '__main__':
    unittest.main()
