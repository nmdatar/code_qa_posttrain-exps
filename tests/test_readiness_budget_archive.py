import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
from training_pipeline.storage import atomic_json,read
from training_pipeline.remote import reallocate_ledger,reconcile_ledger
from training_pipeline.contracts import ConfigurationError
from training_pipeline.tinker_backend import TinkerBackend

class ReadinessSafetyTests(unittest.TestCase):
    def test_reallocation_preserves_spending_and_requires_explicit_reconciliation(self):
        seed={'cap':60,'prices':{},'ttl_seconds':10,'reserved_usd':13,'reservations':[{'amount':13}]}
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'ledger.json';atomic_json(p,seed)
            reallocate_ledger(p,550,'allocation')
            remote=read(p)
            self.assertEqual(remote['reservations'],seed['reservations'])
            self.assertEqual(remote['reserved_usd'],13)
            with self.assertRaises(ConfigurationError):reconcile_ledger(remote,seed)
            reconcile_ledger(remote,seed,550)
            with self.assertRaises(ConfigurationError):reallocate_ledger(p,12,'too-small')
            self.assertEqual(read(p),remote)

    def test_archive_requires_both_downloads_and_optimizer_reload(self):
        backend=object.__new__(TinkerBackend)
        backend.ledger=Mock();backend.service=Mock();backend.timeout=1;backend.load=Mock();backend.create_trainer=Mock()
        with tempfile.TemporaryDirectory() as d, patch('urllib.request.urlopen',side_effect=lambda *a,**k:io.BytesIO(b'checkpoint')), patch('training_pipeline.budget.persist_remote_reservation'):
            result=backend.archive({'training':'training-ref','sampler':'sampler-ref'},d,3600)
            backend.load.assert_called_once_with({'training':'training-ref','sampler':'sampler-ref'},'resume')
            backend.create_trainer.assert_called_once_with(0)
            self.assertTrue(result['optimizer_restore_acknowledged'])
            self.assertFalse(result['archive_import_to_tinker_supported'])
            self.assertEqual(len(result['files']),1)
            self.assertFalse(result['optimizer_archive_download_supported'])
            self.assertEqual(read(Path(d)/'archive.json'),result)

    def test_empty_archive_cannot_claim_restore(self):
        backend=object.__new__(TinkerBackend)
        backend.ledger=Mock();backend.service=Mock();backend.timeout=1;backend.load=Mock()
        with tempfile.TemporaryDirectory() as d, patch('urllib.request.urlopen',return_value=io.BytesIO(b'')):
            with self.assertRaises(ConfigurationError):backend.archive({'training':'t','sampler':'s'},d,3600)
            backend.load.assert_not_called()
            self.assertFalse((Path(d)/'archive.json').exists())

    def test_oversized_evidence_keeps_all_lines_and_claims(self):
        from training_pipeline.strict_grading import normalize_extraction
        original={'claims':[{'id':'a1','evidence_requests':[{'path':'a.py','file_sha256':'hash','start_line':1,'end_line':914}]}]}
        result=normalize_extraction(original)
        refs=result['claims'][0]['evidence_requests']
        self.assertEqual([(r['start_line'],r['end_line']) for r in refs],[(1,600),(601,914)])
        self.assertEqual(original['claims'][0]['evidence_requests'][0]['end_line'],914)
        self.assertTrue(all(r['file_sha256']=='hash' for r in refs))

    def test_best_pointer_changes_only_after_verified_archive(self):
        from training_pipeline.orchestrator import Pipeline
        p=object.__new__(Pipeline);p.backend=Mock();p.tracker=Mock()
        p.config={'best_checkpoint':{'retention_seconds':3600}};p.state={'optimizer_step':1};p.last_path=Path('checkpoint.json')
        with tempfile.TemporaryDirectory() as d:
            p.root=Path(d)
            target=p.root/'checkpoints/best.json'
            old={'quality':.1,'output_tokens':10,'checkpoint_id':'old'};atomic_json(target,old)
            manifest={'id':'new','artifacts':{'training':'t','sampler':'s'}}
            report={'demonstrated_quality':.2,'results':[{'usage':{'output_tokens':20}}]}
            p.backend.archive.side_effect=RuntimeError('restore failed')
            with self.assertRaises(RuntimeError):p.retain_best(manifest,report,'eval.json')
            self.assertEqual(read(target),old)
            p.backend.archive.side_effect=None;p.backend.archive.return_value={'verified':True}
            p.retain_best(manifest,report,'eval.json')
            self.assertEqual(read(target)['checkpoint_id'],'new')

    def test_confirmation_execution_requires_safe_checkpoint(self):
        from training_pipeline.remote import validate_execution
        c={'kind':'modal','operation':'evaluate','cpu':2,'memory_mib':4096,'timeout_seconds':1800,'cohort':'confirmation','checkpoint':'artifacts/run/checkpoints/best.json'}
        validate_execution(c)
        for path in ('../best.json','/private/best.json'):
            with self.assertRaises(ConfigurationError):validate_execution({**c,'checkpoint':path})
        c.pop('checkpoint')
        with self.assertRaises(ConfigurationError):validate_execution(c)
