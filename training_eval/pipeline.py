"""Bounded, synchronous training orchestration with restartable stage state."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import random
import statistics
import uuid

from .contracts import LoopState, ModelRef, RolloutRequest
from .evaluation import Evaluator
from .rollouts import collect
from .strategies import StrategyRegistry


@dataclass(frozen=True)
class BatchContext:
    run_spec: object
    stage: object
    state: LoopState
    policy: object
    rng: random.Random


@dataclass(frozen=True)
class BatchInput:
    items: tuple
    next_cursor: int


def _tuples(value):
    return tuple(_tuples(x) for x in value) if isinstance(value, list) else value


def validate_spec(spec):
    if not isinstance(spec.run_id, str) or not spec.run_id or not spec.stages:
        raise ValueError("run_id and at least one stage are required")
    if type(spec.seed) is not int:
        raise ValueError("seed must be an integer")
    for name in ("max_tokens", "max_turns"):
        if type(getattr(spec, name)) is not int or getattr(spec, name) <= 0:
            raise ValueError(f"{name} must be a positive integer")
    for name in ("checkpoint_every", "eval_every", "group_retries"):
        if type(getattr(spec, name)) is not int or getattr(spec, name) < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    if not math.isfinite(spec.temperature) or spec.temperature < 0:
        raise ValueError("Invalid sampling temperature")
    for stage in spec.stages:
        if type(stage.steps) is not int or stage.steps <= 0 or type(stage.batch_size) is not int or stage.batch_size <= 0:
            raise ValueError("Stage steps/batch_size must be positive integers")
        if not math.isfinite(stage.learning_rate) or stage.learning_rate <= 0:
            raise ValueError("Learning rate must be positive and finite")
        if stage.strategy == "grpo" and (type(stage.group_size) is not int or stage.group_size < 2):
            raise ValueError("GRPO group_size must be at least two")


class Pipeline:
    def __init__(self, factory, checkpoints, tracker, *, sft_dataset=None, tasks=None,
                 environment=None, verifier=None, evaluation_tasks=None, registry=None,
                 input_adapters=None, input_bindings=None):
        self.factory, self.checkpoints, self.tracker = factory, checkpoints, tracker
        self.sft_dataset, self.tasks = sft_dataset, tasks
        self.environment, self.verifier = environment, verifier
        self.evaluation_tasks = evaluation_tasks
        self.registry = registry or StrategyRegistry()
        self.input_adapters = dict(input_adapters or {})
        self.input_bindings = dict(input_bindings or {})

    def _preflight(self, spec):
        validate_spec(spec)
        requirements = set()
        for stage in spec.stages:
            strategy = self.registry.get(stage.strategy)
            requirements.update(strategy.requirements)
            if strategy.input_kind == "sft":
                if self.sft_dataset is None or not len(self.sft_dataset):
                    raise ValueError("SFT requires a nonempty tokenized dataset")
                if any(self.sft_dataset.get(i).split != "train" for i in range(len(self.sft_dataset))):
                    raise ValueError("SFT data must be training split")
            elif strategy.input_kind == "rollouts":
                if self.tasks is None or not len(self.tasks) or self.environment is None or self.verifier is None:
                    raise ValueError("RL requires tasks and external environment/verifier adapters")
                if any(self.tasks.get(i).split != "train" for i in range(len(self.tasks))):
                    raise ValueError("RL data must be training split")
            elif strategy.input_kind not in self.input_adapters or strategy.input_kind not in self.input_bindings:
                raise ValueError("Custom input kinds require an adapter and immutable input binding")
        if spec.eval_every and (self.evaluation_tasks is None or self.environment is None or self.verifier is None):
            raise ValueError("Periodic eval requires held-out tasks/environment/verifier; use eval_every=0 to disable")
        if self.evaluation_tasks is not None:
            if not len(self.evaluation_tasks):
                raise ValueError("Empty evaluation set")
            held_out = [self.evaluation_tasks.get(i) for i in range(len(self.evaluation_tasks))]
            if any(t.split not in {"dev", "development"} for t in held_out):
                raise ValueError("Periodic evaluation accepts development tasks only")
            if self.tasks is not None:
                train = [self.tasks.get(i) for i in range(len(self.tasks))]
                if ({t.family_id for t in train} & {t.family_id for t in held_out}
                        or {t.task_id for t in train} & {t.task_id for t in held_out}):
                    raise ValueError("Train/evaluation task or family leakage")
            if self.sft_dataset is not None:
                eval_ids = {t.task_id for t in held_out}
                eval_families = {t.family_id for t in held_out}
                for i in range(len(self.sft_dataset)):
                    example = self.sft_dataset.get(i)
                    if (example.example_id in eval_ids or example.task_id in eval_ids
                            or example.family_id in eval_families):
                        raise ValueError("SFT/evaluation task or family leakage")
        return frozenset(requirements)

    def _bindings(self, spec):
        strategies = {}
        for stage in spec.stages:
            strategy = self.registry.get(stage.strategy)
            strategies[stage.strategy] = {
                "implementation": f"{type(strategy).__module__}.{type(strategy).__qualname__}",
                "configuration": dict(strategy.configuration())}
        return {"sft": getattr(self.sft_dataset, "identity", None),
                "tasks": getattr(self.tasks, "identity", None),
                "evaluation_tasks": getattr(self.evaluation_tasks, "identity", None),
                "environment": getattr(self.environment, "version", None),
                "verifier": getattr(self.verifier, "version", None),
                "custom_inputs": self.input_bindings, "strategies": strategies}

    def _sft(self, ctx):
        examples = tuple(self.sft_dataset.get((ctx.state.data_cursor + i) % len(self.sft_dataset))
                         for i in range(ctx.stage.batch_size))
        return BatchInput(examples, ctx.state.data_cursor + ctx.stage.batch_size)

    def _rollouts(self, ctx, backend):
        groups, excluded, zero_variance, rewards = [], 0, 0, []
        for offset in range(ctx.stage.batch_size):
            task = self.tasks.get((ctx.state.data_cursor + offset) % len(self.tasks))
            for attempt in range(ctx.run_spec.group_retries + 1):
                group_id = uuid.uuid4().hex
                members = []
                for member in range(ctx.stage.group_size):
                    request = RolloutRequest(ctx.run_spec.run_id, ctx.state.stage, group_id,
                                             f"{group_id}-{member}", ctx.rng.randrange(2**31),
                                             ctx.run_spec.max_tokens, ctx.run_spec.max_turns,
                                             ctx.run_spec.temperature)
                    scored = collect(task, ctx.policy, request, self.environment, self.verifier,
                                     self.tracker, role="training", context_limit=backend.context_limit,
                                     step=ctx.state.optimizer_step)
                    members.append(scored)
                if all(s is not None and s.verification.status == "resolved" for s in members):
                    groups.append(tuple(members))
                    rs = [s.verification.reward for s in members]
                    rewards.extend(rs)
                    zero_variance += len(set(rs)) == 1
                    break
                self.tracker.log("group_excluded", {"group_id": group_id, "task_id": task.task_id,
                                 "retry": attempt, "episodes": len(members)}, step=ctx.state.optimizer_step)
                excluded += 1
            else:
                self.tracker.log("group_quarantined", {"task_id": task.task_id}, step=ctx.state.optimizer_step)
        self.tracker.log("metrics", {"train/excluded_groups": excluded,
                         "train/zero_variance_groups": zero_variance,
                         "train/resolved_groups": len(groups),
                         **({"train/reward": statistics.fmean(rewards)} if rewards else {})},
                         step=ctx.state.optimizer_step)
        return BatchInput(tuple(groups), ctx.state.data_cursor + ctx.stage.batch_size)

    def run(self, spec, model, *, purpose="fork", stop_after_batches=None):
        """Each stage.steps is an attempted-batch cap, including zero-signal batches.

        stop_after_batches is a graceful interruption hook; changing it does not
        change the resumable experiment. Resume restores the same immutable spec.
        """
        requirements = self._preflight(spec)
        if purpose not in {"fork", "resume"}:
            raise ValueError("Training purpose must be fork or resume")
        if stop_after_batches is not None and (type(stop_after_batches) is not int or stop_after_batches <= 0):
            raise ValueError("stop_after_batches must be positive")
        import json
        bindings = json.loads(json.dumps(self._bindings(spec), allow_nan=False))
        manifest = self.checkpoints.read(model.checkpoint) if model.checkpoint else None
        if purpose == "fork" and manifest and manifest["run_spec"]["run_id"] == spec.run_id:
            raise ValueError("A training fork requires a new run_id; use resume to continue this run")
        if purpose == "resume":
            if (manifest is None or manifest["run_spec"] != json.loads(json.dumps(spec.to_dict()))
                    or manifest["bindings"] != bindings):
                raise ValueError("Resume requires unchanged run configuration and input bindings; use fork")
            saved = manifest["state"]
            required = {"stage", "stage_attempt", "optimizer_step", "attempted_batches",
                        "data_cursor", "random_state", "strategy_state"}
            if (not required.issubset(saved) or saved["random_state"] is None
                    or any(type(saved[key]) is not int or saved[key] < 0
                           for key in required - {"random_state", "strategy_state"})
                    or saved["stage"] > len(spec.stages)
                    or saved["optimizer_step"] > saved["attempted_batches"]):
                raise ValueError("Invalid checkpoint loop state")
            index = saved["stage"]
            maximum = spec.stages[index].steps if index < len(spec.stages) else 0
            if (saved["stage_attempt"] > maximum or
                    sum(stage.steps for stage in spec.stages[:index]) + saved["stage_attempt"] != saved["attempted_batches"]):
                raise ValueError("Checkpoint counters do not match stage progress")
        bundle = self.factory.resolve(model, "train" if model.base_model else purpose, requirements=requirements)
        backend, policy = bundle.backend, bundle.policy
        if self.sft_dataset is not None and (self.sft_dataset.tokenizer != backend.identity.tokenizer
                or self.sft_dataset.renderer != backend.identity.renderer):
            raise ValueError("SFT dataset tokenizer/renderer does not match the model")
        state = LoopState(**manifest["state"]) if purpose == "resume" else LoopState()
        if not 0 <= state.stage <= len(spec.stages):
            raise ValueError("Invalid checkpoint stage")
        rng = random.Random(spec.seed)
        if state.random_state is not None:
            rng.setstate(_tuples(state.random_state))
        parent, latest = model.checkpoint, model.checkpoint
        evaluator = (Evaluator(self.evaluation_tasks, self.environment, self.verifier, self.tracker,
                     run_id=spec.run_id, max_tokens=spec.max_tokens, max_turns=spec.max_turns,
                     temperature=spec.temperature, seed=spec.seed, context_limit=backend.context_limit)
                     if spec.eval_every else None)
        self.tracker.log("run_start", {"spec": spec.to_dict(), "identity": asdict(backend.identity),
                         "bindings": bindings, "purpose": purpose}, step=state.optimizer_step)
        executed = 0

        def save():
            nonlocal parent, latest
            state.random_state = rng.getstate()
            latest = self.checkpoints.commit(backend, run_spec=spec, state=state,
                                             bindings=bindings, parent=parent)
            parent = latest
            self.tracker.log("checkpoint", {"reference": latest}, step=state.optimizer_step)
            return latest

        try:
            if purpose != "resume":
                # A recoverable starting boundary exists even if the very first
                # remote optimizer request has an ambiguous outcome.
                save()
                if evaluator:
                    snapshot = backend.load_sampling(self.checkpoints.read(latest)["artifacts"]["sampling"])
                    evaluator.evaluate(snapshot, checkpoint=latest, step=state.optimizer_step)
            while state.stage < len(spec.stages):
                stage = spec.stages[state.stage]
                strategy = self.registry.get(stage.strategy)
                strategy.load_state_dict(state.strategy_state)
                while state.stage_attempt < stage.steps:
                    ctx = BatchContext(spec, stage, state, policy, rng)
                    if strategy.input_kind == "sft":
                        inputs = self._sft(ctx)
                    elif strategy.input_kind == "rollouts":
                        inputs = self._rollouts(ctx, backend)
                    else:
                        inputs = self.input_adapters[strategy.input_kind](ctx)
                    batch = strategy.build_batch(inputs.items, context_limit=backend.context_limit)
                    metrics = backend.update(batch, stage.learning_rate) if batch else {}
                    if batch:
                        state.optimizer_step += 1
                        policy = backend.snapshot(f"policy-{uuid.uuid4().hex}")
                    state.data_cursor = inputs.next_cursor
                    state.stage_attempt += 1
                    state.attempted_batches += 1
                    executed += 1
                    state.strategy_state = dict(strategy.state_dict())
                    self.tracker.log("metrics", {**metrics, "train/optimizer_step": state.optimizer_step,
                                     "train/attempted_batches": state.attempted_batches,
                                     "train/skipped_update": int(batch is None),
                                     "train/stage": state.stage}, step=state.optimizer_step)
                    evaluate_now = bool(batch and evaluator and state.optimizer_step % spec.eval_every == 0)
                    if evaluate_now or (spec.checkpoint_every and state.attempted_batches % spec.checkpoint_every == 0):
                        save()
                    if evaluate_now:
                        # Load exactly the registered snapshot, not a mutable training handle.
                        snapshot = backend.load_sampling(self.checkpoints.read(latest)["artifacts"]["sampling"])
                        evaluator.evaluate(snapshot, checkpoint=latest, step=state.optimizer_step)
                    if stop_after_batches is not None and executed >= stop_after_batches:
                        save()
                        self.tracker.log("run_paused", {"checkpoint": latest}, step=state.optimizer_step)
                        return {"status": "paused", "checkpoint": latest, "state": asdict(state)}
                # Publish completed-stage weights before constructing a fresh
                # optimizer, then publish the new-stage boundary for recovery.
                save()
                state.stage += 1
                state.stage_attempt, state.data_cursor, state.strategy_state = 0, 0, {}
                if state.stage < len(spec.stages):
                    # New post-training stages start from weights with a fresh optimizer.
                    bundle = self.factory.resolve(ModelRef(checkpoint=latest), "fork", requirements=requirements)
                    backend, policy = bundle.backend, bundle.policy
                save()
            self.tracker.log("run_complete", {"checkpoint": latest}, step=state.optimizer_step)
            return {"status": "complete", "checkpoint": latest, "state": asdict(state)}
        except Exception as exc:
            self.tracker.log("run_failed", {"error": str(exc), "last_checkpoint": latest}, step=state.optimizer_step)
            raise
