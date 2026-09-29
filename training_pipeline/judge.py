"""Frozen sampling-only judge with independent rendering and shared spending cap."""
import importlib.metadata
from .contracts import ConfigurationError, Generation, InfrastructureError
from .storage import digest
from .tinker_backend import bounded


class TinkerJudge:
    def __init__(self, config, ledger=None, service=None, sdk=None, renderer_factory=None):
        if renderer_factory is None:
            try:
                from tinker_cookbook.renderers import get_renderer
            except ImportError:
                raise ConfigurationError('Independent judge requires the optional [tinker] dependencies') from None
            renderer_factory = get_renderer
        if sdk is None:
            import tinker as sdk
        self.config, self.ledger, self.sdk = config, ledger, sdk
        self.timeout = config['provider_timeout_seconds']
        self.service = service or sdk.ServiceClient(timeout=self.timeout, max_retries=0)
        try:
            caps = bounded(self.service.get_server_capabilities, self.timeout).supported_models
            cap = next((m for m in caps if m.model_name == config['base_model']), None)
            if not cap or not cap.sampleable or not cap.max_context_length:
                raise ConfigurationError('Judge model must support sampling')
            if config['context_tokens'] > cap.max_context_length:
                raise ConfigurationError('Judge context exceeds provider capability')
            from tinker.lib.retry_handler import RetryConfig
            self.sampler = bounded(lambda: self.service.create_sampling_client(
                base_model=config['base_model'], retry_config=RetryConfig(enable_retry_logic=False,
                progress_timeout=self.timeout)), self.timeout)
            tokenizer = bounded(self.sampler.get_tokenizer, self.timeout)
            self.renderer = renderer_factory(config['renderer'], tokenizer, model_name=config['base_model'])
            self.identity = {'base_model': config['base_model'], 'renderer': config['renderer'],
                'tokenizer_class': type(tokenizer).__name__,
                'template_hash': digest(getattr(tokenizer, 'chat_template', None)),
                'sdk': importlib.metadata.version('tinker'),
                'context_tokens': config['context_tokens'], 'temperature': 0}
            try:
                self.identity['cookbook'] = importlib.metadata.version('tinker-cookbook')
            except importlib.metadata.PackageNotFoundError:
                self.identity['cookbook'] = 'injected-test-renderer'
            self.policy_id = 'judge:'+digest(self.identity)
            self.renderer.build_generation_prompt([{'role':'user','content':'Preflight'}])
        except BaseException:
            self.close('errored')
            raise

    def sample(self, messages, max_tokens, temperature):
        if temperature != 0 or max_tokens != self.config['max_tokens']:
            raise ValueError('Judge sampling settings must match frozen configuration')
        prompt = self.renderer.build_generation_prompt(messages)
        tokens = prompt.to_ints()
        if len(tokens) + max_tokens > self.config['context_tokens']:
            raise ValueError('Judge context overflow; no silent truncation')
        prices = self.config['prices']
        if self.ledger:
            amount = (len(tokens)*prices['prefill']+max_tokens*prices['sample'])/1e6
            self.ledger.reserve_external('judge_sample', amount, model=self.config['base_model'],
                judge_identity=self.identity, prices=prices, input_tokens=len(tokens), output_tokens=max_tokens)
        try:
            response = self.sampler.sample(prompt=prompt, num_samples=1,
                sampling_params=self.sdk.SamplingParams(max_tokens=max_tokens, temperature=temperature,
                    stop=self.renderer.get_stop_sequences())).result(timeout=self.timeout)
        except Exception as exc:
            raise InfrastructureError('Judge sampling failed: '+type(exc).__name__) from None
        if len(response.sequences) != 1:
            raise ValueError('Judge returned unexpected sequence count')
        sequence = response.sequences[0]
        # Preserve raw decoded output even when parsing fails or the response is truncated.
        raw = self.renderer.tokenizer.decode(sequence.tokens, skip_special_tokens=True)
        try:
            parsed, _ = self.renderer.parse_response(sequence.tokens)
            content = parsed['content']
            if isinstance(content, list):
                content = ''.join(p['text'] for p in content if p.get('type') == 'text')
            text = content if isinstance(content, str) else raw
        except (ValueError, KeyError, TypeError):
            text = raw
        result = Generation(tokens, list(sequence.tokens), list(sequence.logprobs or []), text,
                            str(sequence.stop_reason), self.policy_id)
        result.validate()
        return result

    def close(self, status='success'):
        result = self.service.close(status)
        if hasattr(result, 'result'):
            result.result(timeout=self.timeout)
