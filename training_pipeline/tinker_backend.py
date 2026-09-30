"""Thin SDK adapter. All mutable calls are serialized by the orchestrator."""
import importlib.metadata
from datetime import datetime, timezone
import math
import queue
import threading
from .contracts import AmbiguousUpdate, ConfigurationError, Generation, InfrastructureError
from .rendering import ChatRenderer
from .concurrency import SamplingClient


def bounded(call, timeout):
    """Bound SDK setup calls too; daemon transport work cannot block shutdown."""
    result = queue.Queue()
    def invoke():
        try:
            result.put((True, call()))
        except BaseException as exc:
            result.put((False, exc))
    threading.Thread(target=invoke, daemon=True).start()
    try:
        ok, value = result.get(timeout=timeout)
    except queue.Empty:
        raise InfrastructureError('Provider setup timed out') from None
    if not ok:
        raise InfrastructureError('Provider setup failed: ' + type(value).__name__) from None
    return value


def _reduction_failure(reason, **summary):
    error = ValueError('Provider loss reduction validation failed: ' + reason)
    # Fixed categorical reason plus aggregate numeric diagnostics only: never
    # token IDs, text, SDK exception messages or per-token observations.
    error.reduction_diagnostics = {'reason': reason, **summary}
    return error


def check_reduction(rows, loss, output):
    """Validate sum loss; unchanged tolerances, with safe failure diagnostics."""
    scores = output.loss_fn_outputs
    if len(scores) != len(rows):
        raise _reduction_failure('output_count_mismatch', expected_rows=len(rows), returned_rows=len(scores))
    expected = 0.0
    terms = []
    wire_inputs = []
    positive = negative = 0
    for row_index, (row, score) in enumerate(zip(rows, scores)):
        logprobs = list(score['logprobs'].data)
        if len(logprobs) != len(row.weights):
            raise _reduction_failure('token_count_mismatch', row_index=row_index,
                                     expected_tokens=len(row.weights), returned_tokens=len(logprobs))
        for i, weight in enumerate(row.weights):
            if not weight:
                continue
            lp = logprobs[i]
            if not math.isfinite(lp):
                raise _reduction_failure('nonfinite_learner_logprob', row_index=row_index)
            factor = lp if loss == 'cross_entropy' else math.exp(lp-row.logprobs[i])
            term = -weight * factor
            expected += term
            terms.append(term)
            wire_inputs.append((weight, lp, None if loss == 'cross_entropy' else row.logprobs[i]))
            positive += term > 0
            negative += term < 0
    measured = output.metrics['loss:sum']
    if not math.isclose(expected, measured, rel_tol=1e-3, abs_tol=1e-5):
        finite = lambda value: value if math.isfinite(value) else None
        absolute_sum = math.fsum(abs(term) for term in terms)
        stable_expected = math.fsum(terms)
        # Inputs are serialized as float32 by datum(). Quantify, but do not
        # silently substitute, the effect of that rounding on local validation.
        import struct
        f32 = lambda value: struct.unpack('f', struct.pack('f', value))[0]
        wire_expected = math.fsum(-f32(weight) * (lp if old is None else math.exp(lp-f32(old)))
                                  for weight, lp, old in wire_inputs)
        raise _reduction_failure('sum_mismatch', expected=finite(expected), measured=finite(measured),
            absolute_error=finite(abs(expected-measured)), sum_absolute_terms=finite(absolute_sum),
            stable_expected=finite(stable_expected), float32_input_expected=finite(wire_expected),
            accepted_tolerance=finite(max(1e-5, 1e-3*max(abs(expected), abs(measured)))),
            cancellation_ratio=finite(absolute_sum/abs(stable_expected)) if stable_expected else None,
            weighted_tokens=len(terms), positive_terms=positive, negative_terms=negative,
            rows=len(rows), measured_finite=math.isfinite(measured))
    return expected


def training_chunks(rows, data, trainer):
    """Keep each checked RPC below SDK splitting thresholds, preserving weights.

    Dense text RL datums use 20 encoded bytes/token (int32 input, int64
    target, float32 advantage and old logprob). Budget 24 conservatively.
    SFT uses fewer bytes. No token sequence or trajectory is split here.
    """
    max_bytes, max_rows = 1024 * 1024, 32
    config = getattr(getattr(trainer, 'holder', None), '_client_config', None)
    for name, default in (('fwdbwd_max_chunk_bytes_count', max_bytes),
                          ('fwdbwd_max_chunk_len', max_rows)):
        value = getattr(config, name, None)
        if type(value) is int and value > 0:
            if name.endswith('bytes_count'):
                max_bytes = min(default, value)
            else:
                max_rows = min(default, value)
    if len(rows) != len(data):
        raise ValueError('Training datum alignment mismatch')
    chunks, row_chunk, data_chunk, size = [], [], [], 0
    for row, datum in zip(rows, data):
        row_bytes = 24 * len(row.input_tokens)
        if row_bytes > max_bytes:
            raise ConfigurationError('Single training datum exceeds safe unsplit request bound')
        if row_chunk and (len(row_chunk) >= max_rows or size + row_bytes > max_bytes):
            chunks.append((row_chunk, data_chunk))
            row_chunk, data_chunk, size = [], [], 0
        row_chunk.append(row)
        data_chunk.append(datum)
        size += row_bytes
    if row_chunk:
        chunks.append((row_chunk, data_chunk))
    return chunks


class TinkerBackend:
    def __init__(self, model, limits, ledger=None, service=None, sdk=None):
        if sdk is None:
            import tinker as sdk
        self._render_lock = threading.Lock()
        self.sdk = sdk
        self.service = service or sdk.ServiceClient(timeout=limits['provider_timeout_seconds'], max_retries=0)
        self.model, self.limits, self.ledger = model, limits, ledger
        self.timeout = limits['provider_timeout_seconds']
        caps = {m.model_name: m for m in bounded(self.service.get_server_capabilities, self.timeout).supported_models}
        cap = caps.get(model['base_model'])
        if not cap or not cap.sampleable or not cap.trainable or not cap.max_context_length:
            raise ConfigurationError('Configured model must support sampling and training')
        if limits['context_tokens'] > cap.max_context_length:
            raise ConfigurationError('Requested context exceeds provider capability')
        self.sampler = self._sampler(base_model=model['base_model'])
        self.renderer = ChatRenderer(self.sampler.get_tokenizer(), limits['context_tokens'])
        self.identity = {'base_model': model['base_model'], 'rank': model['rank'],
                         **self.renderer.identity, 'sdk': importlib.metadata.version('tinker'),
                         'context_tokens': limits['context_tokens'],
                         'losses': ['cross_entropy', 'importance_sampling']}
        self.policy_id = 'base:' + model['base_model']
        self.trainer = None
        self.poisoned = False

    def _sampler(self, **kwargs):
        from tinker.lib.retry_handler import RetryConfig
        return SamplingClient(bounded(lambda: self.service.create_sampling_client(**kwargs,
            retry_config=RetryConfig(enable_retry_logic=False, progress_timeout=self.timeout)), self.timeout))

    def create_trainer(self, seed):
        self.trainer = bounded(lambda: self.service.create_lora_training_client(
            base_model=self.model['base_model'], rank=self.model['rank'], seed=seed), self.timeout)
        self.poisoned = False

    def sample(self, messages, max_tokens, temperature):
        with self._render_lock:
            prompt = self.renderer.prompt(messages)
        if len(prompt) + max_tokens > self.limits['context_tokens']:
            raise ValueError('Context overflow; no silent truncation')
        if self.ledger:
            self.ledger.reserve('sample', input_tokens=len(prompt), output_tokens=max_tokens)
        try:
            response = self.sampler.sample(prompt=self.sdk.ModelInput.from_ints(prompt), num_samples=1,
                sampling_params=self.sdk.SamplingParams(max_tokens=max_tokens, temperature=temperature,
                    stop=[self.renderer.tokenizer.eos_token_id])).result(timeout=self.timeout)
        except Exception as exc:
            raise InfrastructureError('Tinker sampling failed: ' + type(exc).__name__) from None
        sequence = response.sequences[0]
        with self._render_lock:
            text = self.renderer.tokenizer.decode(sequence.tokens, skip_special_tokens=True)
        result = Generation(prompt, list(sequence.tokens), list(sequence.logprobs or []),
            text,
            str(sequence.stop_reason), self.policy_id)
        result.validate()
        return result

    def datum(self, row, loss):
        row.validate()
        if len(row.input_tokens) > self.limits['context_tokens']:
            raise ValueError('Training context overflow')
        values = {'target_tokens': self.sdk.TensorData(data=row.target_tokens, dtype='int64')}
        if loss == 'cross_entropy':
            if row.logprobs is not None or any(w < 0 for w in row.weights):
                raise ValueError('Invalid SFT row')
            values['weights'] = self.sdk.TensorData(data=row.weights, dtype='float32')
        elif loss == 'importance_sampling':
            if row.logprobs is None:
                raise ValueError('RL requires behavior probabilities')
            values['advantages'] = self.sdk.TensorData(data=row.weights, dtype='float32')
            values['logprobs'] = self.sdk.TensorData(data=row.logprobs, dtype='float32')
        else:
            raise ConfigurationError('Unsupported loss')
        return self.sdk.Datum(model_input=self.sdk.ModelInput.from_ints(row.input_tokens), loss_fn_inputs=values)

    def update(self, rows, loss, learning_rate, optimizer=None):
        if self.trainer is None or self.poisoned:
            raise AmbiguousUpdate('Training client unavailable; restore a checkpoint')
        if not rows or not any(any(r.weights) for r in rows):
            raise ValueError('Cannot update an empty or zero-contribution batch')
        data = [self.datum(row, loss) for row in rows]
        chunks = training_chunks(rows, data, self.trainer)
        # Validate optimizer settings locally before any paid forward/backward call.
        params = self.sdk.AdamParams(learning_rate=learning_rate, **(optimizer or {}))
        if self.ledger:
            self.ledger.reserve('train', input_tokens=sum(len(r.input_tokens) for r in rows))
        phase = 'forward_backward'
        chunk_index = 0
        verified_losses, chunk_metrics = [], []
        try:
            # The provider accumulates gradients across calls. Validate each
            # sequential response before sending the next; update weights ONCE
            # after the entire original batch, with its original global weights.
            for chunk_index, (chunk_rows, chunk_data) in enumerate(chunks):
                phase = 'forward_backward'
                fw = self.trainer.forward_backward(chunk_data, loss_fn=loss).result(timeout=self.timeout)
                phase = 'loss_reduction_validation'
                verified_losses.append(check_reduction(chunk_rows, loss, fw))
                chunk_metrics.append(fw.metrics)
            phase = 'optimizer_step'
            op = self.trainer.optim_step(params).result(timeout=self.timeout)
        except Exception as exc:
            self.poisoned = True
            failure = AmbiguousUpdate('Unknown training outcome; restore last committed checkpoint (' +
                                      type(exc).__name__ + ')')
            # Only locally generated categorical fields are safe to expose. SDK
            # exception text may contain credentials or request payloads.
            failure.update_diagnostics = {
                'phase': phase,
                'cause_type': ''.join(c for c in type(exc).__name__ if c.isalnum() or c == '_')[:64],
                'optimizer_call_attempted': phase == 'optimizer_step',
                'optimizer_acknowledged': False,
                'trainer_poisoned': True,
                'replay_safe': False,
                'chunk_index': chunk_index,
                'chunk_count': len(chunks),
                'verified_chunks': len(verified_losses),
            }
            if phase == 'loss_reduction_validation' and hasattr(exc, 'reduction_diagnostics'):
                failure.update_diagnostics['reduction'] = exc.reduction_diagnostics
            raise failure from None
        metrics = dict(chunk_metrics[0]) if len(chunk_metrics) == 1 else {
            'loss:sum': math.fsum(item['loss:sum'] for item in chunk_metrics)}
        return {'acknowledged': True, 'loss': loss, 'metrics': metrics,
                'verified_sum_loss': math.fsum(verified_losses),
                'forward_backward_chunks': len(chunks), 'chunk_metrics': chunk_metrics,
                'optimizer_metrics': getattr(op, 'metrics', {})}

    def inspect_loss(self, rows, loss):
        """Paid forward-only numerical check; does not accumulate gradients or update."""
        if self.trainer is None or self.poisoned:
            raise AmbiguousUpdate('No usable training client')
        data = [self.datum(row, loss) for row in rows]
        if self.ledger:
            self.ledger.reserve('train', input_tokens=sum(len(r.input_tokens) for r in rows))
        output = self.trainer.forward(data, loss_fn=loss).result(timeout=self.timeout)
        expected = check_reduction(rows, loss, output)
        return {'loss': loss, 'provider_sum': output.metrics['loss:sum'], 'local_sum': expected,
                'rows': len(rows), 'optimizer_updated': False}

    def save(self, name):
        if self.trainer is None or self.poisoned:
            raise AmbiguousUpdate('Cannot save uncertain training state')
        if self.ledger:
            self.ledger.reserve('checkpoint', ttl_seconds=self.model['checkpoint_ttl_seconds'])
        ttl = self.model['checkpoint_ttl_seconds']
        state = self.trainer.save_state(name, ttl_seconds=ttl).result(timeout=self.timeout)
        sample = self.trainer.save_weights_for_sampler(name, ttl_seconds=ttl).result(timeout=self.timeout)
        return {'training': state.path, 'sampler': sample.path, 'ttl_seconds': ttl}

    def verify_artifacts(self, artifacts, purpose=None):
        rest = self.service.create_rest_client()
        for key in (('sampler',) if purpose == 'evaluate' else ('training', 'sampler')):
            path = artifacts.get(key)
            if not path:
                raise ConfigurationError('Missing ' + key + ' checkpoint')
            run = rest.get_training_run_by_tinker_path(path).result(timeout=self.timeout)
            if run.base_model != self.model['base_model'] or run.lora_rank != self.model['rank'] or run.corrupted:
                raise ConfigurationError('Checkpoint model/adaptation mismatch')
            checkpoints = rest.list_checkpoints(run.training_run_id).result(timeout=self.timeout)
            match = [c for c in checkpoints.checkpoints if c.tinker_path == path and c.checkpoint_type == key]
            if not match or (match[0].expires_at is not None and match[0].expires_at <= datetime.now(timezone.utc)):
                raise ConfigurationError('Checkpoint expired or unavailable')

    def use_sampler(self, artifacts):
        self.sampler = self._sampler(model_path=artifacts['sampler'])
        self.policy_id = artifacts['sampler']

    def load(self, artifacts, purpose):
        if purpose not in {'evaluate', 'fork', 'resume'}:
            raise ValueError('Unknown checkpoint operation')
        self.verify_artifacts(artifacts, purpose)
        if purpose != 'evaluate':
            if self.trainer is None:
                raise ValueError('Create a fresh trainer before loading state')
            method = self.trainer.load_state_with_optimizer if purpose == 'resume' else self.trainer.load_state
            method(artifacts['training']).result(timeout=self.timeout)
            self.poisoned = False
        self.use_sampler(artifacts)
        self.load_receipt = {'purpose': purpose, 'artifacts': artifacts, 'acknowledged': True}

    def archive(self, artifacts, directory, retention_seconds):
        """Keep private checkpoint bytes on the durable volume; verify live restore.

        Downloaded archives do not imply a supported Tinker upload/import API.
        Remote references retain a separately budgeted, bounded restoration window.
        """
        import hashlib
        import os
        from pathlib import Path
        import urllib.request
        from .storage import atomic_json
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        if self.ledger:
            self.ledger.reserve('checkpoint', ttl_seconds=retention_seconds)
        rest = self.service.create_rest_client()
        files = {}
        for kind in ('training','sampler'):
            rest.set_checkpoint_ttl_from_tinker_path(artifacts[kind], retention_seconds).result(timeout=self.timeout)
        # The live service rejects archive export for optimizer-state checkpoints.
        # Preserve those remotely for the explicitly budgeted restoration window.
        for kind in ('sampler',):
            url = rest.get_checkpoint_archive_url_from_tinker_path(artifacts[kind]).result(timeout=max(self.timeout,300)).url
            path = root/(kind+'.tar'); pending = root/(kind+'.pending')
            sha = hashlib.sha256(); size = 0
            with urllib.request.urlopen(url, timeout=self.timeout) as source, pending.open('wb') as target:
                while chunk := source.read(1024*1024):
                    size += len(chunk)
                    if size > 8*1024**3:
                        raise ConfigurationError('Checkpoint archive exceeds 8 GiB bound')
                    target.write(chunk); sha.update(chunk)
                target.flush(); os.fsync(target.fileno())
            if not size:
                raise ConfigurationError('Empty checkpoint archive')
            os.replace(pending,path)
            check = hashlib.sha256()
            with path.open('rb') as source:
                while chunk := source.read(1024*1024): check.update(chunk)
            if check.hexdigest() != sha.hexdigest():
                raise ConfigurationError('Checkpoint archive readback failed')
            files[kind] = {'path':str(path),'sha256':sha.hexdigest(),'bytes':size}
        # No uncommitted update exists at a selection evaluation boundary.
        # Fail closed on ambiguous restoration instead of continuing training.
        self.poisoned = True
        # Tinker only accepts LoadWeights at a fresh client's first operation.
        # A new client also verifies recovery independently of the live trainer.
        self.create_trainer(0)
        self.poisoned = True
        self.load(artifacts,'resume')
        result = {'files':files,'remote_artifacts':artifacts,'remote_retention_seconds':retention_seconds,
                  'optimizer_restore_acknowledged':True,'sampler_reload_acknowledged':True,
                  'optimizer_archive_download_supported':False,
                  'archive_import_to_tinker_supported':False}
        atomic_json(root/'archive.json',result)
        from .budget import persist_remote_reservation
        persist_remote_reservation(root/'archive.json')
        return result

    def close(self, status='success'):
        self.service.close(status).result(timeout=self.timeout)
