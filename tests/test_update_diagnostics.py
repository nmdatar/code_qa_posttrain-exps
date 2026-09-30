"""Sanitized failure-phase reporting never permits retry of a mutable client."""
import io
import json
import unittest
from contextlib import redirect_stderr
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

from training_pipeline.contracts import AmbiguousUpdate
from training_pipeline.tinker_backend import TinkerBackend
from training_pipeline.cli import entrypoint


class UpdateDiagnosticsTests(unittest.TestCase):
    def backend(self):
        backend = TinkerBackend.__new__(TinkerBackend)
        backend.poisoned = False
        backend.timeout = 1
        backend.ledger = None
        backend.datum = Mock(return_value='datum')
        backend.sdk = NS(AdamParams=lambda **kwargs: kwargs)
        backend.trainer = Mock()
        backend.trainer.forward_backward.return_value.result.return_value = NS(metrics={})
        backend.trainer.optim_step.return_value.result.return_value = NS(metrics={})
        return backend

    def failure(self, phase):
        backend = self.backend()
        secret = 'Bearer secret-token https://private-request/payload'
        if phase == 'forward_backward':
            backend.trainer.forward_backward.return_value.result.side_effect = TimeoutError(secret)
        if phase == 'optimizer_step':
            backend.trainer.optim_step.return_value.result.side_effect = TimeoutError(secret)
        reduction = ValueError(secret) if phase == 'loss_reduction_validation' else None
        with patch('training_pipeline.tinker_backend.check_reduction', side_effect=reduction, return_value=0):
            with self.assertRaises(AmbiguousUpdate) as caught:
                backend.update([NS(weights=[1], input_tokens=[1])], 'importance_sampling', 1e-5)
        error = caught.exception
        self.assertEqual(error.update_diagnostics['phase'], phase)
        self.assertEqual(error.update_diagnostics['optimizer_call_attempted'], phase == 'optimizer_step')
        self.assertFalse(error.update_diagnostics['optimizer_acknowledged'])
        self.assertFalse(error.update_diagnostics['replay_safe'])
        self.assertTrue(backend.poisoned)
        if phase != 'optimizer_step':
            backend.trainer.optim_step.assert_not_called()
        calls = list(backend.trainer.mock_calls)
        with self.assertRaises(AmbiguousUpdate):
            backend.update([NS(weights=[1], input_tokens=[1])], 'importance_sampling', 1e-5)
        self.assertEqual(calls, backend.trainer.mock_calls)
        stderr = io.StringIO()
        with patch('training_pipeline.cli.main', side_effect=error), redirect_stderr(stderr):
            self.assertEqual(entrypoint(), 1)
        emitted = stderr.getvalue()
        self.assertNotIn(secret, emitted)
        self.assertNotIn('secret-token', emitted)
        payload = json.loads(emitted.splitlines()[1])
        self.assertEqual(payload['phase'], phase)
        self.assertEqual(payload['event'], 'ambiguous_update')

    def test_forward_backward_failure(self):
        self.failure('forward_backward')

    def test_loss_validation_failure_never_calls_optimizer(self):
        self.failure('loss_reduction_validation')

    def test_optimizer_failure_remains_uncertain(self):
        self.failure('optimizer_step')
