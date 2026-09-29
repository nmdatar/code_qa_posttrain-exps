import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from agent_harness.remote_cli import main, submit_bundle
from agent_harness.remote_contracts import job_hash


class RemoteCLITest(unittest.TestCase):
    def fixture(self, root):
        manifest = {'run_id': 'run-1', 'jobs': []}
        for kind in ('public', 'private'):
            (root / kind / 'run-1').mkdir(parents=True)
        (root / 'public' / 'run-1' / 'manifest.json').write_text(json.dumps(manifest))
        (root / 'bundle.json').write_text(json.dumps({'run_id': 'run-1', 'manifest_hash': job_hash(manifest)}))

    def fake_modal(self, events, fail=False):
        class Volume:
            def __init__(self, name): self.name = name
            def batch_upload(self, force=False):
                self.force = force
                return self
            def __enter__(self): return self
            def __exit__(self, *args): return False
            def put_directory(self, local, remote):
                events.append(('upload', self.name, str(local), remote, self.force))
                if fail: raise RuntimeError('upload failed')
        class Function:
            def spawn(self, run_id):
                events.append(('spawn', run_id))
                return SimpleNamespace(object_id='fc-test')
        return SimpleNamespace(Volume=SimpleNamespace(from_name=Volume),
                               Function=SimpleNamespace(from_name=lambda app, name: Function()))

    def test_private_upload_first_then_detached_submit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); self.fixture(root)
            events = []
            with patch.dict(sys.modules, modal=self.fake_modal(events)):
                result = submit_bundle(root, 'harness')
            self.assertEqual(result['call_id'], 'fc-test')
            self.assertEqual([e[1] for e in events[:2]], ['harness-private', 'harness-public'])
            self.assertTrue(all(e[4] is False for e in events[:2]))
            self.assertEqual(events[-1], ('spawn', 'run-1'))

    def test_failed_upload_does_not_launch_remote_work(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); self.fixture(root)
            events = []
            with patch.dict(sys.modules, modal=self.fake_modal(events, fail=True)):
                with self.assertRaises(RuntimeError): submit_bundle(root, 'harness')
            self.assertFalse(any(e[0] == 'spawn' for e in events))

    def test_changed_manifest_rejected_before_cloud_calls(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); self.fixture(root)
            (root / 'public' / 'run-1' / 'manifest.json').write_text('{"run_id":"run-2"}')
            events = []
            with patch.dict(sys.modules, modal=self.fake_modal(events)):
                with self.assertRaises(ValueError): submit_bundle(root, 'harness')
            self.assertEqual(events, [])

    def test_pending_status(self):
        def pending(**kwargs): raise TimeoutError()
        fake = SimpleNamespace(FunctionCall=SimpleNamespace(from_id=lambda _: SimpleNamespace(get=pending)))
        stream = io.StringIO()
        with patch.dict(sys.modules, modal=fake), patch.object(sys, 'argv', ['remote', 'status', '--call-id', 'fc-test']), contextlib.redirect_stdout(stream):
            main()
        self.assertEqual(json.loads(stream.getvalue())['status'], 'pending')


if __name__ == '__main__':
    unittest.main()
