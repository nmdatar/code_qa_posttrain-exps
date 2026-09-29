"""Provider- and domain-neutral contracts for bounded research episodes."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    from .artifacts import ArtifactStore


@dataclass(frozen=True)
class ToolSpec:
    name: str
    version: str
    type: str
    capabilities: tuple[str, ...]
    input_schema: dict[str, Any]
    output_schema: dict[str, Any] | None = None
    timeout_seconds: float = 10.0
    max_output_bytes: int = 32_000
    description: str = ""
    required_resources: tuple[str, ...] = ()


@dataclass
class ToolObservation:
    status: str
    content: Any
    evidence: list[Any] = field(default_factory=list)
    error: str | None = None
    truncated: bool = False


@dataclass(frozen=True)
class ToolContext:
    resources: Mapping[str, Any]
    artifacts: ArtifactStore
    episode_id: str


class Tool(Protocol):
    spec: ToolSpec

    def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolObservation: ...


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict[str, Any]
    call_id: str = ""


@dataclass(frozen=True)
class FinalAnswer:
    value: Any


@dataclass(frozen=True)
class Usage:
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None


@dataclass(frozen=True)
class ModelResponse:
    action: ToolCall | FinalAnswer
    usage: Usage = field(default_factory=Usage)
    token_ids: tuple[int, ...] | None = None
    logprobs: tuple[float, ...] | None = None
    conditioning_token_ids: tuple[int, ...] | None = None


class ModelAdapter(Protocol):
    def generate(self, messages: list[dict[str, Any]], tools: list[ToolSpec],
                 max_output_tokens: int, timeout_seconds: float | None = None) -> ModelResponse: ...


@dataclass(frozen=True)
class RunLimits:
    max_steps: int = 30
    max_tool_calls: int = 25
    max_output_tokens: int = 16_000
    wall_time_seconds: float = 300.0
    max_context_chars: int = 64_000


@dataclass(frozen=True)
class ResearchRequest:
    task_id: str
    question: str
    resources: Mapping[str, Any] = field(default_factory=dict)
    allowed_types: frozenset[str] | None = None
    allowed_capabilities: frozenset[str] | None = None
    permitted_tools: frozenset[str] | None = None
    limits: RunLimits = field(default_factory=RunLimits)


@dataclass(frozen=True)
class EpisodeResult:
    episode_id: str
    task_id: str
    submission: Any
    termination_reason: str
    trajectory_path: str
    metrics_path: str
    metrics: dict[str, Any]


class ModelError(RuntimeError):
    """Provider/transport failure whose consumed usage is unknown."""


class ModelActionError(ModelError):
    """A policy-authored invalid action with preserved provider usage."""

    def __init__(self, message: str, usage: Usage, termination_reason: str = "agent_error", *,
                 conditioning_token_ids=None, token_ids=None, logprobs=None):
        if termination_reason not in {"agent_error", "budget_exhausted"}:
            raise ValueError("invalid model action termination reason")
        super().__init__(message)
        self.usage = usage
        self.termination_reason = termination_reason
        self.conditioning_token_ids = conditioning_token_ids
        self.token_ids = token_ids
        self.logprobs = logprobs
