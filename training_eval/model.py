"""Resolve base models and checkpoint operations without mixing fork and resume."""
from dataclasses import asdict, dataclass
from typing import Any, Literal

from .contracts import Backend, BackendFactory, CapabilityError, CheckpointStore, ModelRef, SamplingPolicy


@dataclass(frozen=True)
class ModelBundle:
    backend: Backend
    policy: SamplingPolicy
    manifest: dict[str, Any] | None = None


class ModelFactory:
    def __init__(self, backends: BackendFactory, checkpoints: CheckpointStore):
        self.backends, self.checkpoints = backends, checkpoints

    def resolve(self, ref: ModelRef, purpose: Literal["train", "evaluate", "fork", "resume"],
                requirements=frozenset()) -> ModelBundle:
        if purpose not in {"train", "evaluate", "fork", "resume"}:
            raise ValueError(f"Unknown model purpose: {purpose}")
        if purpose in {"fork", "resume"} and not ref.checkpoint:
            raise ValueError(f"{purpose} requires a checkpoint")
        if purpose == "train" and ref.checkpoint:
            raise ValueError("Choose fork or resume explicitly for a checkpoint")
        manifest = self.checkpoints.read(ref.checkpoint) if ref.checkpoint else None
        base_model = manifest["identity"]["base_model"] if manifest else ref.base_model
        backend = self.backends(base_model, training=purpose != "evaluate")
        needed = set(requirements) | {"sampling"}
        if purpose == "resume":
            needed.add("resume")
        missing = needed - backend.capabilities
        if missing:
            raise CapabilityError(f"Missing backend capabilities: {sorted(missing)}")
        if manifest:
            if manifest["identity"] != asdict(backend.identity):
                raise CapabilityError("Checkpoint backend/model/tokenizer/renderer/adaptation identity mismatch")
            artifacts = manifest["artifacts"]
            if purpose == "evaluate":
                policy = backend.load_sampling(artifacts["sampling"])
            else:
                backend.load_training(artifacts["training"], optimizer=purpose == "resume")
                policy = backend.snapshot(purpose)
        else:
            policy = backend.snapshot("initial")
        return ModelBundle(backend, policy, manifest)
