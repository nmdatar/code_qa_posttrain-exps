import json
from pathlib import Path
import tempfile
import threading
import unittest
from posttrain.tracking import EventLog, sync_tracking
from posttrain.report import create_report
from posttrain.remote import RemoteConfig


class FakeRun:
    def __init__(self, fail=False):
        self.logs = []
        self.fail = fail
    def define_metric(self, *args, **kwargs):
        pass
    def log(self, data, step):
        self.logs.append((data, step))
        if self.fail:
            raise ConnectionError('secret must not be recorded')
    def finish(self, **kwargs):
        pass


class FakeWandb:
    def __init__(self, fail=False):
        self.run = FakeRun(fail)
        self.calls = []
    def init(self, **kwargs):
        self.calls.append(kwargs)
        return self.run


class TrackingTests(unittest.TestCase):
    def test_concurrent_events_and_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            threads = [threading.Thread(target=lambda: EventLog(tmp).emit('update', loss=.2)) for _ in range(12)]
            for t in threads: t.start()
            for t in threads: t.join()
            log = EventLog(tmp)
            self.assertEqual([e['event_id'] for e in log.read()], list(range(1,13)))
            with log.path.open('ab') as out: out.write(b'{"partial":')
            self.assertEqual(len(log.read()),12)
            self.assertEqual(log.emit('recovery')['event_id'],13)
            self.assertEqual(len(log.read()),13)

    def test_tracking_failure_and_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            EventLog(tmp).emit('update', loss=.4, prompt='private reference')
            failed = FakeWandb(True)
            self.assertEqual(sync_tracking(tmp,'online',wandb_module=failed)['status'],'pending')
            success = FakeWandb()
            self.assertEqual(sync_tracking(tmp,'online',wandb_module=success)['events'],1)
            self.assertEqual(failed.calls[0]['id'],success.calls[0]['id'])
            self.assertNotIn('prompt',success.run.logs[0][0])
            self.assertEqual(sync_tracking(tmp,'online',wandb_module=success)['events'],0)
            self.assertEqual(len(success.run.logs),1)
            self.assertNotIn('secret',Path(tmp,'tracking-state.json').read_text())

    def test_modes_and_destination(self):
        with tempfile.TemporaryDirectory() as tmp:
            EventLog(tmp).emit('update', loss=.2)
            self.assertEqual(sync_tracking(tmp)['status'],'disabled')
            fake = FakeWandb()
            self.assertEqual(sync_tracking(tmp,'offline',wandb_module=fake)['status'],'queued_offline')
            self.assertEqual(sync_tracking(tmp,'online',wandb_module=fake)['status'],'uploaded')
            with self.assertRaises(ValueError): sync_tracking(tmp,'online',project='other',wandb_module=fake)

    def test_report_escaping_and_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp,'trajectory.json').write_text('{}')
            EventLog(tmp).emit('update', loss=.3, step=1, status='<script>bad()</script>', trajectory_path='trajectory.json')
            EventLog(tmp).emit('evaluation', step=2, metrics={'accepted_rate':.5})
            text = create_report(tmp).read_text()
            self.assertNotIn('<script>',text)
            self.assertIn('&lt;script&gt;',text)
            self.assertIn('<svg',text)
            self.assertIn('accepted_rate',text)
            self.assertIn('trajectory.json',text)

    def test_remote_volume_separation(self):
        RemoteConfig().validate()
        with self.assertRaises(ValueError): RemoteConfig(runs_volume='same',private_volume='same').validate()
        with self.assertRaises(ValueError): RemoteConfig(timeout_seconds=999999).validate()

    def test_reserved_event_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError): EventLog(tmp).emit('update',event_id=900)
            with self.assertRaises(ValueError): EventLog(tmp).emit('update',loss=float('nan'))
            self.assertEqual(EventLog(tmp).read(),[])


if __name__ == '__main__': unittest.main()
