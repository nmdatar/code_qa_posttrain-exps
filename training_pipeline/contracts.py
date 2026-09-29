"""Provider-independent records. No SDK tensors cross these boundaries."""
from dataclasses import asdict, dataclass, field
import math
from typing import Protocol


class ConfigurationError(ValueError):
    pass


class InfrastructureError(RuntimeError):
    pass


class AmbiguousUpdate(InfrastructureError):
    """Discard this training client and restore a committed boundary."""


@dataclass
class Generation:
    prompt: list[int]
    tokens: list[int]
    logprobs: list[float]
    text: str
    stop_reason: str
    policy_id: str

    def validate(self):
        if not self.prompt or not self.tokens or len(self.tokens) != len(self.logprobs):
            raise ValueError('Empty generation or misaligned log probabilities')
        if any(type(t) is not int or t < 0 for t in self.prompt + self.tokens):
            raise ValueError('Invalid token ID')
        if any(not math.isfinite(p) for p in self.logprobs):
            raise ValueError('Nonfinite behavior log probability')
        if not self.policy_id:
            raise ValueError('Missing policy identity')


@dataclass
class VerificationResult:
    status: str
    reward: float | None
    version: str
    reasons: list[str] = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)
    retryable: bool = False

    def validate(self):
        if self.status not in {'resolved', 'unresolved'} or not self.version:
            raise ValueError('Invalid verification result')
        if self.status == 'resolved' and (self.reward is None or not math.isfinite(self.reward)):
            raise ValueError('Resolved reward must be finite')
        if self.status == 'unresolved' and self.reward is not None:
            raise ValueError('Unresolved reward must be null')


@dataclass
class Trajectory:
    run_id: str
    stage: int
    task_id: str
    task_hash: str
    group_id: str
    episode_id: str
    policy_id: str
    environment_id: str
    experiment_hash: str
    split: str
    generations: list[Generation] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)
    submission: object = None
    termination: str = 'infrastructure_error'
    usage: dict = field(default_factory=dict)
    verification: VerificationResult | None = None

    def to_dict(self):
        return asdict(self)


@dataclass
class LossRow:
    input_tokens: list[int]
    target_tokens: list[int]
    weights: list[float]
    logprobs: list[float] | None = None

    def validate(self):
        n = len(self.input_tokens)
        if not n or len(self.target_tokens) != n or len(self.weights) != n:
            raise ValueError('Misaligned loss row')
        if self.logprobs is not None and len(self.logprobs) != n:
            raise ValueError('Misaligned behavior probabilities')
        if any(type(t) is not int or t < 0 for t in self.input_tokens + self.target_tokens):
            raise ValueError('Invalid loss token')
        if any(not math.isfinite(v) for v in self.weights + (self.logprobs or [])):
            raise ValueError('Nonfinite loss input')


class Backend(Protocol):
    identity: dict
    policy_id: str
    def sample(self, messages, max_tokens: int, temperature: float) -> Generation: ...
    def update(self, rows: list[LossRow], loss: str, learning_rate: float) -> dict: ...
    def save(self, name: str) -> dict: ...
    def load(self, artifacts: dict, purpose: str): ...


class Episode(Protocol):
    messages: list[dict]
    def step(self, action: dict) -> tuple[bool, object]: ...
    def verify(self, trajectory: Trajectory) -> VerificationResult: ...
    def close(self): ...
