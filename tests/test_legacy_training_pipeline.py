import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from training_eval.backends import FakeBackendFactory
from training_eval.checkpoints import LocalCheckpointStore
from training_eval.contracts import (InfrastructureError, ModelRef, RunSpec, StageSpec,
                                     Task, Verification, SFTExample)
from training_eval.data import InMemoryTaskSource, InMemorySFTDataset
from training_eval.evaluation import Evaluator
from training_eval.mocks import demo_inputs, MockEnvironment, MockVerifier
from training_eval.model import ModelFactory
from training_eval.pipeline import Pipeline, BatchInput
from training_eval.strategies import SFTStrategy, StrategyRegistry
from training_eval.tracking import JsonTracker


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.spec = RunSpec("test-run", (StageSpec("sft", 2, 0.1), StageSpec("grpo", 3, 0.1, 2, 4)),
                            seed=7, max_tokens=8, max_turns=2)

    def pipeline(self, name, inputs=None, **kwargs):
        root = self.root / name
        store = LocalCheckpointStore(root / "checkpoints")
        factory = ModelFactory(FakeBackendFactory(root / "backend"), store)
        tracker = JsonTracker(root / "logs", "test-run")
        pipeline = Pipeline(factory, store, tracker, **(inputs if inputs is not None else demo_inputs()), **kwargs)
        return pipeline

    def backend_state(self, pipeline, checkpoint):
        manifest = pipeline.checkpoints.read(checkpoint)
        return json.loads(Path(manifest["artifacts"]["training"]).read_text())

    def test_end_to_end_and_exact_local_resume(self):
        full = self.pipeline("full")
        result = full.run(self.spec, ModelRef(base_model="mock"))
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["state"]["attempted_batches"], 5)
        partial = self.pipeline("partial")
        paused = partial.run(self.spec, ModelRef(base_model="mock"), stop_after_batches=3)
        self.assertEqual(paused["status"], "paused")
        resumed = self.pipeline("resumed")
        completed = resumed.run(self.spec, ModelRef(checkpoint=paused["checkpoint"]), purpose="resume")
        self.assertEqual(self.backend_state(full, result["checkpoint"]),
                         self.backend_state(resumed, completed["checkpoint"]))
        self.assertEqual(completed["state"]["optimizer_step"], result["state"]["optimizer_step"])

    def test_stage_boundary_checkpoint_has_fresh_optimizer(self):
        p = self.pipeline("boundaries")
        p.run(self.spec, ModelRef(base_model="mock"))
        boundary = next(path for path in p.checkpoints.directory.glob('*.json')
                        if (d := json.loads(path.read_text()))["state"]["stage"] == 1
                        and d["state"]["stage_attempt"] == 0)
        self.assertEqual(self.backend_state(p, str(boundary))["optimizer_steps"], 0)
        resumed = self.pipeline("boundary-resume")
        outcome = resumed.run(self.spec, ModelRef(checkpoint=str(boundary)), purpose="resume")
        self.assertEqual(outcome["status"], "complete")

    def test_fork_resets_optimizer_and_records_lineage(self):
        p = self.pipeline("fork")
        paused = p.run(self.spec, ModelRef(base_model="mock"), stop_after_batches=1)
        fork = p.factory.resolve(ModelRef(checkpoint=paused["checkpoint"]), "fork")
        resume = p.factory.resolve(ModelRef(checkpoint=paused["checkpoint"]), "resume")
        self.assertEqual(fork.backend.optimizer_steps, 0)
        self.assertEqual(resume.backend.optimizer_steps, 1)
        self.assertEqual(fork.backend.weight, resume.backend.weight)
        self.assertEqual(fork.backend.momentum, 0)

    def test_resume_rejects_config_and_input_changes_before_loading(self):
        p = self.pipeline("mismatch")
        paused = p.run(self.spec, ModelRef(base_model="mock"), stop_after_batches=1)
        with self.assertRaisesRegex(ValueError, "unchanged"):
            p.run(replace(self.spec, temperature=0.5), ModelRef(checkpoint=paused["checkpoint"]), purpose="resume")
        inputs = demo_inputs()
        inputs["environment"].version = "different-environment"
        changed = self.pipeline("changed-env", inputs)
        with self.assertRaisesRegex(ValueError, "unchanged"):
            changed.run(self.spec, ModelRef(checkpoint=paused["checkpoint"]), purpose="resume")

    def test_direct_grpo_and_zero_signal_is_bounded(self):
        class ConstantVerifier(MockVerifier):
            def verify(self, *args, **kwargs):
                return Verification("resolved", 1.0)
        inputs = demo_inputs()
        inputs["verifier"] = ConstantVerifier()
        p = self.pipeline("zero", inputs)
        spec = replace(self.spec, stages=(StageSpec("grpo", 2, 0.1),))
        result = p.run(spec, ModelRef(base_model="mock"))
        self.assertEqual(result["state"]["optimizer_step"], 0)
        self.assertEqual(result["state"]["attempted_batches"], 2)
        self.assertTrue(Path(result["checkpoint"]).exists())

    def test_unresolved_group_retries_whole_group_then_quarantines(self):
        class Broken(MockEnvironment):
            calls = 0
            def run(self, *args):
                self.calls += 1
                raise InfrastructureError("offline")
        inputs = demo_inputs()
        env = inputs["environment"] = Broken()
        p = self.pipeline("failure", inputs)
        spec = replace(self.spec, stages=(StageSpec("grpo", 1, 0.1, group_size=3),), eval_every=0)
        result = p.run(spec, ModelRef(base_model="mock"))
        self.assertEqual(env.calls, 6)
        self.assertEqual(result["state"]["optimizer_step"], 0)
        events = [json.loads(line) for line in p.tracker.events_path.read_text().splitlines()]
        self.assertEqual(sum(e["kind"] == "group_quarantined" for e in events), 1)

    def test_periodic_eval_loads_committed_snapshot_and_preserves_coverage(self):
        p = self.pipeline("eval")
        result = p.run(self.spec, ModelRef(base_model="mock"))
        events = [json.loads(line) for line in p.tracker.events_path.read_text().splitlines()]
        reports = [e["payload"] for e in events if e["kind"] == "evaluation"]
        self.assertGreater(len(reports), 1)
        for report in reports[1:]:
            manifest = p.checkpoints.read(report["checkpoint"])
            self.assertEqual(report["policy_version"], manifest["artifacts"]["sampling"])
            self.assertEqual(report["coverage"], 1)
        bundle = p.factory.resolve(ModelRef(checkpoint=result["checkpoint"]), "evaluate")
        self.assertFalse(bundle.backend.training)
        class Missing(MockVerifier):
            def verify(self, *args, **kwargs):
                raise InfrastructureError("judge offline")
        report = Evaluator(p.evaluation_tasks, p.environment, Missing(), p.tracker).evaluate(bundle.policy)
        self.assertIsNone(report["mean_reward"])
        self.assertEqual(report["unresolved"], 1)
        self.assertEqual(report["coverage"], 0)

    def test_split_protection_and_final_test_opt_in(self):
        inputs = demo_inputs()
        inputs["evaluation_tasks"] = InMemoryTaskSource([Task("different", "development", "train-family", {})])
        with self.assertRaisesRegex(ValueError, "leakage"):
            self.pipeline("leak", inputs).run(self.spec, ModelRef(base_model="mock"))
        p = self.pipeline("final")
        tasks = InMemoryTaskSource([Task("final", "final_test", "unseen", {})])
        evaluator = Evaluator(tasks, p.environment, p.verifier, p.tracker)
        policy = p.factory.resolve(ModelRef(base_model="mock"), "evaluate").policy
        with self.assertRaisesRegex(ValueError, "split"):
            evaluator.evaluate(policy)
        self.assertEqual(evaluator.evaluate(policy, final_test=True)["resolved"], 1)

    def test_sft_only_does_not_require_environment(self):
        p = self.pipeline("sft-only", {"sft_dataset": demo_inputs()["sft_dataset"]})
        spec = replace(self.spec, stages=(StageSpec("sft", 1, 0.1),), eval_every=0)
        self.assertEqual(p.run(spec, ModelRef(base_model="mock"))["state"]["optimizer_step"], 1)

    def test_new_strategy_and_input_adapter_without_orchestrator_changes(self):
        class NewStrategy(SFTStrategy):
            name = "new-objective"
            input_kind = "reviewed-pairs-fixture"
        source = demo_inputs()["sft_dataset"]
        registry = StrategyRegistry([NewStrategy()])
        p = self.pipeline("plugin", {}, registry=registry,
                          input_adapters={"reviewed-pairs-fixture": lambda ctx: BatchInput((source.get(0),), ctx.state.data_cursor + 1)},
                          input_bindings={"reviewed-pairs-fixture": source.identity})
        spec = replace(self.spec, stages=(StageSpec("new-objective", 1, 0.1),), eval_every=0)
        self.assertEqual(p.run(spec, ModelRef(base_model="mock"))["status"], "complete")

    def test_sampler_mismatch_fails_without_training(self):
        class WrongPolicy(MockEnvironment):
            def run(self, *args):
                return replace(super().run(*args), policy_version="wrong")
        inputs = demo_inputs()
        inputs["environment"] = WrongPolicy()
        p = self.pipeline("bad-policy", inputs)
        spec = replace(self.spec, stages=(StageSpec("grpo", 1, 0.1),), eval_every=0)
        with self.assertRaisesRegex(ValueError, "identity"):
            p.run(spec, ModelRef(base_model="mock"))
        checkpoints = list(p.checkpoints.directory.glob('*.json'))
        self.assertEqual(len(checkpoints), 1)  # Recoverable bootstrap, no trained checkpoint.
        self.assertEqual(p.checkpoints.read(str(checkpoints[0]))["state"]["optimizer_step"], 0)

    def test_standalone_evaluator_rejects_invalid_budget(self):
        p = self.pipeline("budget")
        with self.assertRaisesRegex(ValueError, "budgets"):
            Evaluator(p.evaluation_tasks, p.environment, p.verifier, p.tracker, max_turns=0)

    def test_sft_lineage_rejects_known_evaluation_overlap(self):
        inputs = demo_inputs()
        inputs["sft_dataset"] = InMemorySFTDataset(
            [SFTExample("sft-example", (10, 49), (0, 1), task_id="dev-1", family_id="dev-family")],
            tokenizer="fake-tokenizer-v1", renderer="tokens-v1")
        with self.assertRaisesRegex(ValueError, "SFT/evaluation"):
            self.pipeline("sft-leak", inputs).run(self.spec, ModelRef(base_model="mock"))

    def test_custom_strategy_configuration_is_bound_for_resume(self):
        class ConfiguredStrategy(SFTStrategy):
            def __init__(self, coefficient):
                self.coefficient = coefficient
            def configuration(self):
                return {**super().configuration(), "coefficient": self.coefficient}
        p = self.pipeline("strategy-config", registry=StrategyRegistry([ConfiguredStrategy(1)]))
        spec = replace(self.spec, stages=(StageSpec("sft", 2, 0.1),))
        paused = p.run(spec, ModelRef(base_model="mock"), stop_after_batches=1)
        changed = self.pipeline("changed-strategy", registry=StrategyRegistry([ConfiguredStrategy(2)]))
        with self.assertRaisesRegex(ValueError, "unchanged"):
            changed.run(spec, ModelRef(checkpoint=paused["checkpoint"]), purpose="resume")

    def test_pipeline_fork_requires_new_run_id(self):
        p = self.pipeline("fork-id")
        paused = p.run(self.spec, ModelRef(base_model="mock"), stop_after_batches=1)
        with self.assertRaisesRegex(ValueError, "new run_id"):
            p.run(self.spec, ModelRef(checkpoint=paused["checkpoint"]), purpose="fork")

    def test_incomplete_resume_state_is_rejected(self):
        p = self.pipeline("bad-state")
        paused = p.run(self.spec, ModelRef(base_model="mock"), stop_after_batches=1)
        manifest = p.checkpoints.read(paused["checkpoint"])
        del manifest["state"]["random_state"]
        damaged_copy = self.root / 'incomplete-manifest.json'
        damaged_copy.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "loop state"):
            p.run(self.spec, ModelRef(checkpoint=str(damaged_copy)), purpose="resume")


if __name__ == "__main__":
    unittest.main()
