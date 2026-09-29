"""Thin SDK adapter. All mutable calls are serialized by the orchestrator."""
import importlib.metadata
from datetime import datetime, timezone
import math
import queue
import threading
from .contracts import AmbiguousUpdate, ConfigurationError, Generation, InfrastructureError
from .rendering import ChatRenderer


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


def check_reduction(rows, loss, output):
    """Check the provider's reported sum loss against its returned token scores."""
    scores = output.loss_fn_outputs
    if len(scores) != len(rows):
        raise ValueError('Provider returned misaligned training outputs')
    expected = 0.0
    for row, score in zip(rows, scores):
        logprobs = list(score['logprobs'].data)
        if len(logprobs) != len(row.weights):
            raise ValueError('Provider returned misaligned token scores')
        for i, weight in enumerate(row.weights):
            if not weight:
                continue
            lp = logprobs[i]
            if not math.isfinite(lp):
                raise ValueError('Nonfinite learner probability')
            expected -= weight * (lp if loss == 'cross_entropy' else math.exp(lp-row.logprobs[i]))
    measured = output.metrics['loss:sum']
    if not math.isclose(expected, measured, rel_tol=1e-3, abs_tol=1e-5):
        raise ValueError('Provider loss reduction differs from configured sum objective')
    return expected


class TinkerBackend:
    def __init__(self, model, limits, ledger=None, service=None, sdk=None):
        if sdk is None:
            import tinker as sdk
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
        return bounded(lambda: self.service.create_sampling_client(**kwargs,
            retry_config=RetryConfig(enable_retry_logic=False, progress_timeout=self.timeout)), self.timeout)

    def create_trainer(self, seed):
        self.trainer = bounded(lambda: self.service.create_lora_training_client(
            base_model=self.model['base_model'], rank=self.model['rank'], seed=seed), self.timeout)
        self.poisoned = False

    def sample(self, messages, max_tokens, temperature):
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
        result = Generation(prompt, list(sequence.tokens), list(sequence.logprobs or []),
            self.renderer.tokenizer.decode(sequence.tokens, skip_special_tokens=True),
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

    def update(self, rows, loss, learning_rate):
        if self.trainer is None or self.poisoned:
            raise AmbiguousUpdate('Training client unavailable; restore a checkpoint')
        if not rows or not any(any(r.weights) for r in rows):
            raise ValueError('Cannot update an empty or zero-contribution batch')
        data = [self.datum(row, loss) for row in rows]
        if self.ledger:
            self.ledger.reserve('train', input_tokens=sum(len(r.input_tokens) for r in rows))
        try:
            fw = self.trainer.forward_backward(data, loss_fn=loss).result(timeout=self.timeout)
            checked_loss = check_reduction(rows, loss, fw)
            op = self.trainer.optim_step(self.sdk.AdamParams(learning_rate=learning_rate)).result(timeout=self.timeout)
        except Exception as exc:
            self.poisoned = True
            raise AmbiguousUpdate('Unknown training outcome; restore last committed checkpoint (' +
                                  type(exc).__name__ + ')') from None
        return {'acknowledged': True, 'loss': loss, 'metrics': fw.metrics,
                'verified_sum_loss': checked_loss,
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
            self.ledger.reserve('checkpoint')
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

    def close(self, status='success'):
        self.service.close(status).result(timeout=self.timeout)
