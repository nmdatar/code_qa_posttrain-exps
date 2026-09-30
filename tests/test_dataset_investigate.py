import hashlib
import json
from pathlib import Path
import subprocess
import time

import unittest
import tempfile
from dataset_builder import investigate as inv
from dataset_builder.contracts import canonical_hash


class Backend:
    def __init__(self, stdout='ok\n', fail=False):
        self.calls=[]; self.stdout=stdout; self.fail=fail
    def run(self, image, command, limits, stdin=None):
        self.calls.append((image, command, limits, stdin))
        if self.fail: raise RuntimeError('provider unavailable')
        return {'stdout':self.stdout,'stderr':'','exit_code':0,'timed_out':False,'truncated':False,'sandbox_id':'test-sandbox','duration_seconds':0.1}


def bundle(tmp_path):
    repo=tmp_path/'repo'; repo.mkdir()
    subprocess.run(['git','init',str(repo)],check=True,capture_output=True)
    (repo/'code.py').write_text('value = 7\n')
    subprocess.run(['git','-C',str(repo),'add','.'],check=True)
    subprocess.run(['git','-C',str(repo),'-c','user.name=Test','-c','user.email=test@example.com','commit','-m','test'],check=True,capture_output=True)
    commit=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD']).decode().strip()
    root=tmp_path/'bundle'; (root/'public').mkdir(parents=True); (root/'environment-build').mkdir()
    repository={'url':'https://github.com/test/repo','commit':commit,'family_id':'test'}
    task={'schema_version':'1.0','split':'development','id':'test','repository':repository,'environment_id':'env-test','user_prompt':'What is value?', 'system_prompt':'Inspect source',
          'permitted_tools':['read_file','list_files','search_code','python_probe'],
          'budgets':{'latency_seconds':1200,'max_tool_calls':4,'max_submission_bytes':1000,'compute_units':100000,'max_output_tokens':4000}}
    (root/'public/tasks.jsonl').write_text(json.dumps(task)+'\n')
    env={'id':'env-test','repository':repository,'snapshot_path':str(repo)}
    inv._write(root/'public/environment.json',env)
    hashes={'code.py':hashlib.sha256((repo/'code.py').read_bytes()).hexdigest()}
    built={'status':'ready','commit':commit,'source_environment_sha256':canonical_hash(env),'backend':'modal','image_digest':'im-TEST',
           'readiness':{'exit_code':0,'timed_out':False,'truncated':False},'snapshot_files':hashes,
           'snapshot_sha256':hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()}
    inv._write(root/'environment-build/result.json',built)
    inv._write(root/'manifest.json',{'artifacts':{p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in ['public/tasks.jsonl','public/environment.json']}})
    # Deliberately unreadable/invalid private record proves no gold is needed.
    (root/'private').mkdir(); (root/'private/tasks.jsonl').write_text('SECRET INVALID GOLD')
    return root


def test_start_only_public_and_snapshot_integrity(bundle,tmp_path):
    attempt=tmp_path/'attempt'; record=inv.start(bundle,'test',attempt)
    assert 'SECRET' not in json.dumps(record)
    assert record['token_usage'] is None and record['cost_usd'] is None
    assert set(p.name for p in attempt.iterdir())=={'attempt.json','trajectory.jsonl'}
    repo=Path(inv._read(bundle/'public/environment.json')['snapshot_path'])
    (repo/'code.py').write_text('changed')
    with unittest.TestCase().assertRaisesRegex(ValueError,'pristine'): inv.start(bundle,'test',tmp_path/'bad')


def test_unready_binding_and_hash_rejected(bundle,tmp_path):
    path=bundle/'environment-build/result.json'; built=inv._read(path); built['status']='quarantined'; inv._write(path,built)
    with unittest.TestCase().assertRaisesRegex(ValueError,'not ready'): inv.start(bundle,'test',tmp_path/'bad')
    (bundle/'public/tasks.jsonl').write_text('{}')
    with unittest.TestCase().assertRaisesRegex(ValueError,'hash mismatch'): inv.start(bundle,'test',tmp_path/'bad')


def test_dispatch_log_replay_and_finish(bundle,tmp_path):
    attempt=tmp_path/'attempt'; inv.start(bundle,'test',attempt); backend=Backend()
    event=inv.tool(attempt,'read_file',{'path':'code.py'},backend)
    assert event['status']=='ok' and event['observation']['sandbox_id']=='test-sandbox'
    image,command,limits,stdin=backend.calls[0]
    assert image=='im-TEST' and command[:2]==['python','-I']
    assert limits.pids==64 and limits.memory_mb==512 and limits.output_bytes==65536
    assert 'SECRET' not in stdin
    assert inv.replay(attempt,tmp_path/'replay.json',backend)['status']=='passed'
    assert inv.replay(attempt,tmp_path/'mismatch.json',Backend('different'))['status']=='not_passed'
    answer=tmp_path/'answer.txt'; answer.write_text('value is 7 [code.py:1]')
    result=inv.finish(attempt,answer)
    assert result['citation_validation']=='pending independent review'
    assert inv.tool(attempt,'list_files',{},backend)['status']=='error'


def test_traversal_budget_errors_and_failure_logged(bundle,tmp_path):
    attempt=tmp_path/'attempt'; inv.start(bundle,'test',attempt); backend=Backend()
    for path in ('../../private/tasks.jsonl','/etc/passwd'):
        assert inv.tool(attempt,'read_file',{'path':path},backend)['status']=='error'
    assert not backend.calls
    failure=inv.tool(attempt,'python_probe',{'code':'print(7)'},Backend(fail=True))
    assert failure['status']=='error' and 'provider unavailable' in failure['error']
    assert inv.tool(attempt,'list_files',{},backend)['status']=='ok'
    assert 'budget exhausted' in inv.tool(attempt,'list_files',{},backend)['error']
    assert len(backend.calls)==1


def test_deadline_and_tampering(bundle,tmp_path):
    attempt=tmp_path/'attempt'; inv.start(bundle,'test',attempt)
    record=inv._read(attempt/'attempt.json'); record['started_at']=time.time()-2000; inv._write(attempt/'attempt.json',record)
    backend=Backend(); assert 'deadline' in inv.tool(attempt,'list_files',{},backend)['error']; assert not backend.calls
    path=attempt/'trajectory.jsonl'; path.write_text(path.read_text().replace('deadline exceeded','other reason'))
    with unittest.TestCase().assertRaisesRegex(ValueError,'hash chain'): inv.replay(attempt,tmp_path/'replay.json',backend)


def test_symlink_escape_rejected(bundle,tmp_path):
    repo=Path(inv._read(bundle/'public/environment.json')['snapshot_path'])
    (repo/'code.py').unlink(); (repo/'code.py').symlink_to('/etc/passwd')
    with unittest.TestCase().assertRaisesRegex(ValueError,'pristine'): inv.start(bundle,'test',tmp_path/'attempt')


class InvestigationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.bundle = bundle(self.root)

    def test_start_only_public_and_snapshot_integrity(self):
        test_start_only_public_and_snapshot_integrity(self.bundle, self.root)

    def test_unready_binding_and_hash_rejected(self):
        test_unready_binding_and_hash_rejected(self.bundle, self.root)

    def test_dispatch_log_replay_and_finish(self):
        test_dispatch_log_replay_and_finish(self.bundle, self.root)

    def test_traversal_budget_errors_and_failure_logged(self):
        test_traversal_budget_errors_and_failure_logged(self.bundle, self.root)

    def test_deadline_and_tampering(self):
        test_deadline_and_tampering(self.bundle, self.root)

    def test_symlink_escape_rejected(self):
        test_symlink_escape_rejected(self.bundle, self.root)


    def test_finish_rejects_empty_and_expired(self):
        attempt=self.root/'attempt'; inv.start(self.bundle,'test',attempt)
        answer=self.root/'empty.txt'; answer.write_text('   ')
        with self.assertRaisesRegex(ValueError,'nonempty'):
            inv.finish(attempt,answer)
        self.assertFalse((attempt/'answer.json').exists())
        record=inv._read(attempt/'attempt.json'); record['started_at']=time.time()-2000
        inv._write(attempt/'attempt.json',record)
        answer.write_text('Some answer')
        with self.assertRaisesRegex(ValueError,'deadline'):
            inv.finish(attempt,answer)

    def test_changed_binding_rejected_before_dispatch(self):
        attempt=self.root/'attempt'; inv.start(self.bundle,'test',attempt)
        backend=Backend(); inv.tool(attempt,'list_files',{},backend)
        record=inv._read(attempt/'attempt.json'); record['image_id']='im-OTHER'
        inv._write(attempt/'attempt.json',record)
        event=inv.tool(attempt,'list_files',{},backend)
        self.assertIn('binding changed',event['error'])
        self.assertEqual(len(backend.calls),1)

    def test_private_field_in_public_task_rejected(self):
        path=self.bundle/'public/tasks.jsonl'; task=json.loads(path.read_text())
        task['reference_answer']='PRIVATE'; path.write_text(json.dumps(task)+'\n')
        manifest=inv._read(self.bundle/'manifest.json')
        manifest['artifacts']['public/tasks.jsonl']=hashlib.sha256(path.read_bytes()).hexdigest()
        inv._write(self.bundle/'manifest.json',manifest)
        with self.assertRaises(ValueError):
            inv.start(self.bundle,'test',self.root/'attempt')
