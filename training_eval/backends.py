"""Optional Tinker adapter and an explicitly synthetic, offline test backend."""
from __future__ import annotations

import json
import math
import random
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

from .contracts import (BackendArtifacts, CapabilityError, InfrastructureError,
                        ModelIdentity, Sample, UncertainUpdateError, UpdateBatch)

CAPABILITIES = frozenset({"cross_entropy", "importance_sampling", "sampling", "checkpoint", "resume"})


def _validate_batch(batch: UpdateBatch, context_limit: int, learning_rate: float) -> None:
    if batch.loss not in {"cross_entropy", "importance_sampling"}:
        raise CapabilityError(f"Unsupported loss: {batch.loss}")
    if not batch.rows or not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("An update requires rows and a finite positive learning rate")
    for row in batch.rows:
        n = len(row.input_tokens)
        if not n or n > context_limit or len(row.target_tokens) != n or len(row.weights) != n:
            raise ValueError("Training arrays must have equal nonzero lengths within context limit")
        if any(type(token) is not int or token < 0 for token in (*row.input_tokens, *row.target_tokens)):
            raise ValueError("Token IDs must be nonnegative integers")
        if any(not math.isfinite(w) or w < 0 for w in row.weights):
            raise ValueError("Loss weights must be finite and nonnegative")
        if batch.loss == "importance_sampling":
            if len(row.old_logprobs) != n or len(row.advantages) != n:
                raise ValueError("RL rows require aligned old logprobs and advantages")
            if any(not math.isfinite(x) for x in (*row.old_logprobs, *row.advantages)):
                raise ValueError("RL values must be finite")
            if any(x > 0 for x in row.old_logprobs):
                raise ValueError("Sampling log probabilities must be nonpositive")


def _validate_sample(prompt: Sequence[int], max_tokens: int, temperature: float, limit: int):
    if not prompt or type(max_tokens) is not int or max_tokens <= 0 or len(prompt) + max_tokens > limit:
        raise ValueError("Sampling requires a prompt and generation budget within context limit")
    if any(type(token) is not int or token < 0 for token in prompt):
        raise ValueError("Prompt token IDs must be nonnegative integers")
    if not math.isfinite(temperature) or temperature < 0:
        raise ValueError("Temperature must be finite and nonnegative")


@dataclass(frozen=True)
class FakePolicy:
    """A two-token Bernoulli model; intentionally not a language model."""
    version: str
    weight: float
    context_limit: int

    def sample(self, prompt_tokens, *, max_tokens, temperature, seed):
        _validate_sample(prompt_tokens, max_tokens, temperature, self.context_limit)
        scaled = max(-700.0, min(700.0, self.weight / max(temperature, 1e-8)))
        p = 1 / (1 + math.exp(-scaled)) if temperature else float(self.weight >= 0)
        outcome = int(random.Random(seed).random() < p)
        probability = p if outcome else 1 - p
        return Sample(tuple(prompt_tokens), (48 + outcome,), (math.log(max(probability, 1e-30)),), str(outcome))


class FakeBackend:
    """Stateful mock for orchestration tests, with local durable checkpoints."""
    capabilities = CAPABILITIES

    def __init__(self, base_model, artifact_dir, *, training=True, context_limit=4096):
        self.identity = ModelIdentity("fake", base_model, "fake-tokenizer-v1", "tokens-v1")
        self.context_limit = context_limit
        self.training = training
        self.artifact_dir = Path(artifact_dir)
        self.weight = 0.0
        self.optimizer_steps = 0
        self.momentum = 0.0

    def update(self, batch, learning_rate):
        if not self.training:
            raise CapabilityError("Evaluation backend cannot train")
        _validate_batch(batch, self.context_limit, learning_rate)
        p = 1 / (1 + math.exp(-self.weight))
        loss = gradient = 0.0
        for row in batch.rows:
            for i, (target, weight) in enumerate(zip(row.target_tokens, row.weights)):
                if not weight:
                    continue
                y = float(target % 2)
                lp = math.log(p if y else 1 - p)
                if batch.loss == "cross_entropy":
                    loss -= weight * lp
                    gradient += weight * (p - y)
                else:
                    ratio = math.exp(lp - row.old_logprobs[i])
                    # GRPOStrategy includes token/trajectory reduction in advantages.
                    scale = row.advantages[i] * ratio
                    loss -= scale
                    gradient -= scale * (y - p)
        self.momentum = 0.9 * self.momentum + gradient
        self.weight = max(-20.0, min(20.0, self.weight - learning_rate * self.momentum))
        self.optimizer_steps += 1
        return {"loss": loss, "mock_weight": self.weight}

    def snapshot(self, name):
        return FakePolicy(f"fake:{name}:{uuid.uuid4().hex}", self.weight, self.context_limit)

    def save(self, name):
        if not self.training:
            raise CapabilityError("Evaluation backend cannot save training state")
        directory = self.artifact_dir / uuid.uuid4().hex
        directory.mkdir(parents=True)
        content = {"identity": asdict(self.identity), "weight": self.weight,
                   "optimizer_steps": self.optimizer_steps, "momentum": self.momentum}
        train, sample = directory / "training.json", directory / "sampling.json"
        train.write_text(json.dumps(content))
        sample.write_text(json.dumps({"identity": content["identity"], "weight": self.weight}))
        return BackendArtifacts(str(train.resolve()), str(sample.resolve()))

    def _read(self, reference):
        data = json.loads(Path(reference).read_text())
        if data["identity"] != asdict(self.identity):
            raise CapabilityError("Checkpoint model identity mismatch")
        return data

    def load_training(self, reference, *, optimizer):
        if not self.training:
            raise CapabilityError("Evaluation backend cannot load training state")
        data = self._read(reference)
        self.weight = data["weight"]
        self.optimizer_steps = data["optimizer_steps"] if optimizer else 0
        self.momentum = data["momentum"] if optimizer else 0.0

    def load_sampling(self, reference):
        return FakePolicy(reference, self._read(reference)["weight"], self.context_limit)

    def verify_artifacts(self, artifacts):
        self._read(artifacts.training)
        self._read(artifacts.sampling)


class FakeBackendFactory:
    def __init__(self, artifact_dir, context_limit=4096):
        self.artifact_dir, self.context_limit = artifact_dir, context_limit

    def __call__(self, base_model, *, training):
        return FakeBackend(base_model, self.artifact_dir, training=training, context_limit=self.context_limit)


@dataclass(frozen=True)
class TinkerPolicy:
    version: str
    client: object
    sdk: object
    context_limit: int
    tokenizer: object
    stop: object = None

    def sample(self, prompt_tokens, *, max_tokens, temperature, seed):
        _validate_sample(prompt_tokens, max_tokens, temperature, self.context_limit)
        try:
            response = self.client.sample(
                prompt=self.sdk.ModelInput.from_ints(list(prompt_tokens)), num_samples=1,
                sampling_params=self.sdk.SamplingParams(max_tokens=max_tokens, temperature=temperature,
                                                        seed=seed, stop=self.stop),
            ).result()
            seq = response.sequences[0]
            if not seq.tokens or len(seq.tokens) > max_tokens:
                raise InfrastructureError("Tinker returned an empty or over-budget completion")
            if seq.logprobs is None or len(seq.logprobs) != len(seq.tokens):
                raise InfrastructureError("Tinker returned missing or misaligned sampling logprobs")
            if any(not math.isfinite(x) for x in seq.logprobs):
                raise InfrastructureError("Tinker returned nonfinite logprobs")
            return Sample(tuple(prompt_tokens), tuple(seq.tokens), tuple(seq.logprobs),
                          self.tokenizer.decode(seq.tokens), str(seq.stop_reason))
        except InfrastructureError:
            raise
        except Exception as exc:
            raise InfrastructureError("Tinker sampling failed") from exc


class TinkerBackend:
    capabilities = CAPABILITIES

    def __init__(self, service, sdk, base_model, *, renderer, context_limit, rank=32, training=True, stop=None):
        self.service, self.sdk = service, sdk
        self.context_limit = context_limit
        stop_identity = list(stop) if isinstance(stop, (list, tuple)) else stop
        self.identity = ModelIdentity("tinker", base_model, base_model, renderer, {"rank": rank, "stop": stop_identity})
        self.client = service.create_lora_training_client(base_model=base_model, rank=rank) if training else None
        self.base_sampler = None if training else service.create_sampling_client(base_model=base_model)
        self.tokenizer = (self.client or self.base_sampler).get_tokenizer()
        self.stop = tuple(stop) if isinstance(stop, list) else stop
        self._saved = set()
        self._uncertain = False

    def update(self, batch, learning_rate):
        if self.client is None:
            raise CapabilityError("Evaluation backend cannot train")
        if self._uncertain:
            raise UncertainUpdateError("Discard this backend and resume a committed checkpoint")
        _validate_batch(batch, self.context_limit, learning_rate)
        data = []
        for row in batch.rows:
            td = lambda values, dtype: self.sdk.TensorData(data=list(values), dtype=dtype, shape=[len(values)])
            inputs = {"target_tokens": td(row.target_tokens, "int64")}
            if batch.loss == "cross_entropy":
                inputs["weights"] = td(row.weights, "float32")
            else:
                inputs["logprobs"] = td(row.old_logprobs, "float32")
                inputs["advantages"] = td(row.advantages, "float32")
            data.append(self.sdk.Datum(model_input=self.sdk.ModelInput.from_ints(list(row.input_tokens)),
                                       loss_fn_inputs=inputs))
        try:
            result = self.client.forward_backward(data, batch.loss).result()
            self.client.optim_step(self.sdk.AdamParams(learning_rate=learning_rate)).result()
            return {"loss": float(result.metrics["loss:sum"])}
        except Exception as exc:
            self._uncertain = True
            raise UncertainUpdateError("Tinker update failed; do not retry, resume a committed checkpoint") from exc

    def _policy(self, client, version):
        return TinkerPolicy(version, client, self.sdk, self.context_limit, self.tokenizer, self.stop)

    def snapshot(self, name):
        if self._uncertain:
            raise UncertainUpdateError("Cannot snapshot an uncertain update")
        client = self.client.save_weights_and_get_sampling_client() if self.client else self.base_sampler
        return self._policy(client, f"tinker:{name}:{uuid.uuid4().hex}")

    def save(self, name):
        if self.client is None:
            raise CapabilityError("Evaluation backend cannot save training state")
        if self._uncertain:
            raise UncertainUpdateError("Cannot checkpoint an uncertain update")
        artifacts = BackendArtifacts(self.client.save_state(name).result().path,
                                     self.client.save_weights_for_sampler(name).result().path)
        self._saved.add((artifacts.training, artifacts.sampling))
        return artifacts

    def load_training(self, reference, *, optimizer):
        if self.client is None:
            raise CapabilityError("Evaluation backend cannot load training state")
        if self._uncertain:
            raise UncertainUpdateError("Discard this backend and resume using a new backend")
        method = self.client.load_state_with_optimizer if optimizer else self.client.load_state
        method(reference).result()

    def load_sampling(self, reference):
        return self._policy(self.service.create_sampling_client(model_path=reference), reference)

    def verify_artifacts(self, artifacts):
        # Both acknowledged durable save results are required before manifest commit.
        if (artifacts.training, artifacts.sampling) not in self._saved:
            raise InfrastructureError("Artifacts were not acknowledged by this backend")
        if not all(p.startswith("tinker://") for p in (artifacts.training, artifacts.sampling)):
            raise InfrastructureError("Tinker returned invalid artifact paths")


class TinkerBackendFactory:
    """Token-native adapter; caller supplies a verified model context limit.

    No tokenizer or chat-template compatibility is inferred from a label. Supply
    the renderer's stop sequences explicitly for tool/action boundaries. The
    context bound is validated locally and the service remains authoritative.
    """

    def __init__(self, *, renderer, context_limit, rank=32, service=None, sdk=None, stop=None):
        if type(context_limit) is not int or context_limit <= 0 or type(rank) is not int or rank <= 0 or not renderer:
            raise ValueError("Renderer, positive context limit and rank are required")
        if stop is not None and not isinstance(stop, str):
            if not isinstance(stop, (tuple, list)) or not (
                all(type(value) is int and value >= 0 for value in stop)
                or all(isinstance(value, str) and value for value in stop)
            ):
                raise ValueError("Stop must be a string or homogeneous sequence of token IDs/strings")
            stop = list(stop)
        if stop == "":
            raise ValueError("Stop strings cannot be empty")
        self.renderer, self.context_limit, self.rank = renderer, context_limit, rank
        self.service, self.sdk = service, sdk
        self.stop = stop

    def __call__(self, base_model, *, training):
        if self.sdk is None:
            try:
                import tinker
            except ImportError as exc:
                raise CapabilityError("Install the optional Tinker dependencies to use this backend") from exc
            self.sdk = tinker
        if self.service is None:
            self.service = self.sdk.ServiceClient()
        names = {m.model_name for m in self.service.get_server_capabilities().supported_models}
        if base_model not in names:
            raise CapabilityError(f"Tinker does not advertise model {base_model!r}")
        return TinkerBackend(self.service, self.sdk, base_model, renderer=self.renderer,
                             context_limit=self.context_limit, rank=self.rank, training=training, stop=self.stop)
