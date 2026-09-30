"""Unsplit RPCs preserve the global objective and permit exactly one optimizer."""
import unittest
from types import SimpleNamespace as NS
from training_pipeline.contracts import AmbiguousUpdate, ConfigurationError
from training_pipeline.tinker_backend import TinkerBackend, training_chunks


def row(size=10, weight=.01):
    return NS(input_tokens=[1]*size, weights=[weight]*size, logprobs=[0.]*size)


class Trainer:
    def __init__(self, bad_chunk=None):
        self.calls = []
        self.bad_chunk = bad_chunk
        self.holder = NS(_client_config=NS(fwdbwd_max_chunk_bytes_count=5_000_000,
                                         fwdbwd_max_chunk_len=1024))
    def forward_backward(self, data, loss_fn):
        chunk_index = len(self.calls)
        self.calls.append(('forward', data))
        loss = -sum(sum(r.weights) for r in data)
        if chunk_index == self.bad_chunk:
            loss *= 2
        out = NS(metrics={'loss:sum': loss},
                 loss_fn_outputs=[{'logprobs': NS(data=[0.]*len(r.weights))} for r in data])
        return NS(result=lambda **kwargs: out)
    def optim_step(self, params):
        self.calls.append(('optimizer', params))
        return NS(result=lambda **kwargs: NS(metrics={}))


def backend(trainer):
    value = TinkerBackend.__new__(TinkerBackend)
    value.trainer = trainer
    value.poisoned = False
    value.timeout = 1
    value.ledger = None
    value.datum = lambda r, loss: r
    value.sdk = NS(AdamParams=lambda **kwargs: kwargs)
    return value


class TrainingChunkTests(unittest.TestCase):
    def test_sequential_chunks_one_optimizer_unchanged_weight_objects(self):
        rows = [row(weight=1/700) for _ in range(70)]
        trainer = Trainer()
        result = backend(trainer).update(rows, 'importance_sampling', 1e-5)
        self.assertEqual([x[0] for x in trainer.calls], ['forward', 'forward', 'forward', 'optimizer'])
        self.assertEqual([len(x[1]) for x in trainer.calls[:-1]], [32, 32, 6])
        flattened = [r for _, chunk in trainer.calls[:-1] for r in chunk]
        self.assertTrue(all(a is b for a, b in zip(flattened, rows)))
        self.assertAlmostEqual(result['verified_sum_loss'], -1.)
        self.assertAlmostEqual(result['metrics']['loss:sum'], -1.)
        self.assertEqual(result['forward_backward_chunks'], 3)

    def test_chunk_failure_prevents_all_remaining_calls_and_poisoned_retry(self):
        rows = [row() for _ in range(70)]
        trainer = Trainer(bad_chunk=1)
        value = backend(trainer)
        with self.assertRaises(AmbiguousUpdate) as failure:
            value.update(rows, 'importance_sampling', 1e-5)
        self.assertEqual([x[0] for x in trainer.calls], ['forward', 'forward'])
        self.assertEqual(failure.exception.update_diagnostics['verified_chunks'], 1)
        self.assertEqual(failure.exception.update_diagnostics['chunk_index'], 1)
        self.assertFalse(failure.exception.update_diagnostics['optimizer_call_attempted'])
        with self.assertRaises(AmbiguousUpdate):
            value.update(rows, 'importance_sampling', 1e-5)
        self.assertEqual(len(trainer.calls), 2)

    def test_byte_limit_and_live_sdk_limits(self):
        rows = [row(size=10000) for _ in range(8)]
        chunks = training_chunks(rows, rows, Trainer())
        self.assertEqual([len(x[0]) for x in chunks], [4, 4])
        trainer = Trainer()
        trainer.holder._client_config.fwdbwd_max_chunk_len = 2
        tiny = [row(size=1) for _ in range(5)]
        self.assertEqual([len(x[0]) for x in training_chunks(tiny, tiny, trainer)], [2, 2, 1])
        trainer.holder._client_config.fwdbwd_max_chunk_bytes_count = 40
        self.assertEqual([len(x[0]) for x in training_chunks(tiny, tiny, trainer)], [1]*5)

    def test_oversized_datum_rejected_without_paid_calls(self):
        trainer = Trainer()
        trainer.holder._client_config.fwdbwd_max_chunk_bytes_count = 40
        with self.assertRaises(ConfigurationError):
            backend(trainer).update([row(size=2)], 'importance_sampling', 1e-5)
        self.assertEqual(trainer.calls, [])
