"""Provider-neutral contracts for training/eval; environments live in other packages."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Mapping, Protocol, Sequence


class CapabilityError(ValueError):
    """A model/backend cannot perform the requested operation."""


class InfrastructureError(RuntimeError):
    """An attempt is unresolved rather than a negative policy outcome."""


class UncertainUpdateError(RuntimeError):
    """Do not retry: a remote optimizer update may have completed."""


@dataclass(frozen=True)
class ModelRef:
    base_model: str | None = None
    checkpoint: str | None = None

    def __post_init__(self):
        if bool(self.base_model) == bool(self.checkpoint):
            raise ValueError("Specify exactly one base_model or checkpoint")


@dataclass(frozen=True)
class ModelIdentity:
    backend: str
    base_model: str
    tokenizer: str
    renderer: str
    adaptation: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Sample:
    prompt_tokens: tuple[int, ...]
    tokens: tuple[int, ...]
    logprobs: tuple[float, ...]
    text: str = ""
    stop_reason: str = "stop"


class SamplingPolicy(Protocol):
    version: str

    def sample(self, prompt_tokens: Sequence[int], *, max_tokens: int,
               temperature: float, seed: int) -> Sample: ...


@dataclass(frozen=True)
class TrainingRow:
    """Already shifted input/target arrays; weights include desired loss reduction."""
    input_tokens: tuple[int, ...]
    target_tokens: tuple[int, ...]
    weights: tuple[float, ...]
    old_logprobs: tuple[float, ...] = ()
    advantages: tuple[float, ...] = ()


@dataclass(frozen=True)
class UpdateBatch:
    loss: str
    rows: tuple[TrainingRow, ...]


@dataclass(frozen=True)
class BackendArtifacts:
    training: str
    sampling: str


class Backend(Protocol):
    """One serialized training client plus immutable sampling handles."""
    identity: ModelIdentity
    capabilities: frozenset[str]
    context_limit: int

    def update(self, batch: UpdateBatch, learning_rate: float) -> Mapping[str, float]: ...
    def snapshot(self, name: str) -> SamplingPolicy: ...
    def save(self, name: str) -> BackendArtifacts: ...
    def load_training(self, reference: str, *, optimizer: bool) -> None: ...
    def load_sampling(self, reference: str) -> SamplingPolicy: ...
    def verify_artifacts(self, artifacts: BackendArtifacts) -> None: ...


class BackendFactory(Protocol):
    def __call__(self, base_model: str, *, training: bool) -> Backend: ...


@dataclass(frozen=True)
class SFTExample:
    example_id: str
    tokens: tuple[int, ...]
    loss_mask: tuple[float, ...]
    split: str = "train"
    task_id: str | None = None
    family_id: str | None = None


class SFTDataset(Protocol):
    identity: str
    tokenizer: str
    renderer: str

    def __len__(self) -> int: ...
    def get(self, index: int) -> SFTExample: ...


@dataclass(frozen=True)
class Task:
    task_id: str
    split: str
    family_id: str
    payload: Mapping[str, Any]


class TaskSource(Protocol):
    identity: str

    def __len__(self) -> int: ...
    def get(self, index: int) -> Task: ...


@dataclass(frozen=True)
class RolloutRequest:
    run_id: str
    stage: int
    group_id: str
    episode_id: str
    seed: int
    max_tokens: int
    max_turns: int
    temperature: float


@dataclass(frozen=True)
class Trajectory:
    task_id: str
    episode_id: str
    policy_version: str
    environment_version: str
    turns: tuple[Sample, ...]
    answer: str
    termination: str = "completed"
    metrics: Mapping[str, float | None] = field(default_factory=dict)
    observations: tuple[Any, ...] = ()


class Environment(Protocol):
    """Integration boundary to the external harness, NOT a tools/sandbox implementation.

    Own task setup, action parsing, tool budgets, isolation, and cleanup. Preserve
    the exact policy token records and keep private grading context off the policy.
    """
    version: str

    def run(self, task: Task, policy: SamplingPolicy, request: RolloutRequest) -> Trajectory: ...


@dataclass(frozen=True)
class Verification:
    status: Literal["resolved", "unresolved"]
    reward: float | None
    components: Mapping[str, float] = field(default_factory=dict)
    reason: str = ""


class Verifier(Protocol):
    version: str

    def verify(self, task: Task, trajectory: Trajectory, *, role: str) -> Verification: ...


@dataclass(frozen=True)
class ScoredTrajectory:
    trajectory: Trajectory
    verification: Verification


@dataclass(frozen=True)
class StageSpec:
    strategy: str
    steps: int
    learning_rate: float
    batch_size: int = 1
    group_size: int = 2


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    stages: tuple[StageSpec, ...]
    seed: int = 0
    max_tokens: int = 64
    max_turns: int = 4
    temperature: float = 1.0
    checkpoint_every: int = 1
    eval_every: int = 1
    group_retries: int = 1

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class LoopState:
    stage: int = 0
    stage_attempt: int = 0
    optimizer_step: int = 0
    attempted_batches: int = 0
    data_cursor: int = 0
    random_state: Any = None
    strategy_state: dict[str, Any] = field(default_factory=dict)


class TrainingStrategy(Protocol):
    name: str
    input_kind: str
    requirements: frozenset[str]

    def build_batch(self, items: Sequence[Any], *, context_limit: int) -> UpdateBatch | None: ...
    def configuration(self) -> Mapping[str, Any]: ...
    def state_dict(self) -> Mapping[str, Any]: ...
    def load_state_dict(self, state: Mapping[str, Any]) -> None: ...


class CheckpointStore(Protocol):
    def commit(self, backend: Backend, *, run_spec: RunSpec, state: LoopState,
               bindings: Mapping[str, Any], parent: str | None) -> str: ...
    def read(self, reference: str) -> dict[str, Any]: ...


class TrackingSink(Protocol):
    def emit(self, event: Mapping[str, Any]) -> None: ...
    def close(self) -> None: ...


class Tracker(Protocol):
    def log(self, kind: str, payload: Mapping[str, Any], *, step: int = 0) -> str: ...
    def artifact(self, name: str, payload: Any) -> str: ...
    def close(self) -> None: ...


class Evaluation(Protocol):
    def evaluate(self, policy: SamplingPolicy, *, checkpoint: str | None,
                 step: int = 0, final_test: bool = False) -> dict[str, Any]: ...
