import json
import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from training_eval.contracts import (CapabilityError, Sample, ScoredTrajectory,
                                     SFTExample, Task, Trajectory, Verification)
from training_eval.data import InMemorySFTDataset, InMemoryTaskSource, JsonlSFTDataset
from training_eval.strategies import GRPOStrategy, SFTStrategy, StrategyRegistry


def scored(episode, reward, turns=None, **kwargs):
    turns = turns or (Sample((1, 2), (3, 4), (-0.2, -0.3)),)
    return ScoredTrajectory(Trajectory("task", episode, "policy", "env", turns, "", **kwargs),
                            Verification("resolved", reward))


class StrategyTests(unittest.TestCase):
    def test_sft_selected_token_mean_and_shift(self):
        examples = [SFTExample("a", (1, 2, 3), (0, 1, 0)),
                    SFTExample("b", (4, 5, 6), (0, 1, 1))]
        batch = SFTStrategy().build_batch(examples, context_limit=3)
        self.assertEqual(batch.rows[0].input_tokens, (1, 2))
        self.assertEqual(batch.rows[0].target_tokens, (2, 3))
        self.assertEqual(batch.rows[0].weights, (1 / 3, 0))
        self.assertAlmostEqual(sum(sum(r.weights) for r in batch.rows), 1)

    def test_sft_rejects_bad_masks_tokens_and_context(self):
        for tokens, mask in [((1, 2), (1, 1)), ((1, 2), (0, math.nan)),
                             ((1, 2), (0, .5)), ((1, -2), (0, 1)), ((1, 2), (0, 0))]:
            with self.subTest(tokens=tokens, mask=mask), self.assertRaises(ValueError):
                SFTStrategy().build_batch([SFTExample("a", tokens, mask)], context_limit=3)
        with self.assertRaises(ValueError):
            SFTStrategy().build_batch([SFTExample("a", (1, 2, 3), (0, 1, 1))], context_limit=2)

    def test_grpo_context_bound_counts_full_prompt_and_generation(self):
        group = [scored("a", 0), scored("b", 1)]
        with self.assertRaises(ValueError):
            GRPOStrategy().build_batch([group], context_limit=3)
        self.assertIsNotNone(GRPOStrategy().build_batch([group], context_limit=4))

    def test_grpo_multiturn_uses_exact_context_without_training_context(self):
        turns = (Sample((1, 2), (3, 4), (-.2, -.3)),
                 Sample((8, 9, 10, 11), (12,), (-.4,)))
        batch = GRPOStrategy().build_batch([[scored("a", 0, turns), scored("b", 2)]], context_limit=8)
        self.assertEqual(len(batch.rows), 3)
        self.assertEqual(batch.rows[1].input_tokens, (8, 9, 10, 11))
        self.assertEqual(batch.rows[1].target_tokens, (9, 10, 11, 12))
        self.assertEqual(batch.rows[1].old_logprobs, (0, 0, 0, -.4))
        self.assertEqual(batch.rows[1].advantages, (0, 0, 0, -1 / 6))
        self.assertAlmostEqual(sum(batch.rows[0].weights) + sum(batch.rows[1].weights), .5)
        self.assertAlmostEqual(sum(batch.rows[2].weights), .5)
        self.assertAlmostEqual(sum(sum(r.advantages) for r in batch.rows), 0)

    def test_equal_groups_skip_and_do_not_dilute_loss(self):
        groups = [[scored("a", 1), scored("b", 1)], [scored("c", 0), scored("d", 2)]]
        self.assertIsNone(GRPOStrategy().build_batch(groups[:1], context_limit=10))
        batch = GRPOStrategy().build_batch(groups, context_limit=10)
        self.assertEqual(len(batch.rows), 2)
        self.assertAlmostEqual(sum(sum(r.weights) for r in batch.rows), 1)

    def test_invalid_groups(self):
        good = scored("a", 0)
        other = scored("b", 1)
        for group in [[good], [good, good], [good, scored("b", math.inf)],
                      [good, replace(other, verification=Verification("unresolved", None))],
                      [good, replace(other, trajectory=replace(other.trajectory, policy_version="stale"))],
                      [good, replace(other, trajectory=replace(other.trajectory, task_id="other"))],
                      [good, scored("b", 1, (Sample((1,), (2,), (math.nan,)),))],
                      [good, scored("b", 1, (Sample((1,), (2,), ()),))]]:
            with self.subTest(group=group), self.assertRaises(ValueError):
                GRPOStrategy().build_batch([group], context_limit=10)

    def test_extreme_finite_rewards(self):
        batch = GRPOStrategy().build_batch([[scored("a", -1e308), scored("b", 1e308)]], context_limit=10)
        self.assertTrue(all(math.isfinite(a) for r in batch.rows for a in r.advantages))

    def test_registry_and_state(self):
        registry = StrategyRegistry()
        self.assertIsInstance(registry.get("sft", frozenset({"cross_entropy"})), SFTStrategy)
        for name, capabilities in [("ppo", None), ("grpo", frozenset({"cross_entropy"}))]:
            with self.assertRaises(CapabilityError):
                registry.get(name, capabilities)
        with self.assertRaises(ValueError):
            registry.register(SFTStrategy())
        with self.assertRaises(ValueError):
            SFTStrategy().load_state_dict({"unknown": 1})

    def test_strategy_configuration_is_json_stable_and_detached(self):
        for strategy in (SFTStrategy(), GRPOStrategy()):
            configuration = strategy.configuration()
            self.assertEqual(json.loads(json.dumps(configuration, allow_nan=False)), configuration)
            self.assertEqual(configuration["objective"], next(iter(strategy.requirements)))
            configuration["version"] = "caller mutation"
            self.assertEqual(strategy.configuration()["version"], 1)
        self.assertEqual(GRPOStrategy().configuration()["advantage_normalization"],
                         "group_population_standard_deviation")


class DatasetTests(unittest.TestCase):
    def test_sft_leakage_metadata_survives_adapters_and_changes_identity(self):
        example = SFTExample("one", (1, 2), (0, 1), task_id="task", family_id="family")
        dataset = InMemorySFTDataset([example], tokenizer="tok", renderer="render")
        self.assertEqual(dataset.get(0), example)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.jsonl"
            path.write_text(json.dumps({"example_id": "one", "tokens": [1, 2], "loss_mask": [0, 1],
                                        "task_id": "task", "family_id": "family"}) + "\n")
            loaded = JsonlSFTDataset(path, tokenizer="tok", renderer="render")
        self.assertEqual(dataset.identity, loaded.identity)
        changed = InMemorySFTDataset([replace(example, family_id="other")], tokenizer="tok", renderer="render")
        self.assertNotEqual(dataset.identity, changed.identity)
        with self.assertRaisesRegex(ValueError, "task_id/family_id"):
            InMemorySFTDataset([replace(example, task_id="")], tokenizer="tok", renderer="render")

    def test_jsonl_and_memory_have_identical_content_identity(self):
        example = SFTExample("one", (1, 2), (0, 1))
        dataset = InMemorySFTDataset([example], tokenizer="tok", renderer="render")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.jsonl"
            path.write_text(json.dumps({"example_id": "one", "tokens": [1, 2], "loss_mask": [0, 1]}) + "\n")
            loaded = JsonlSFTDataset(path, tokenizer="tok", renderer="render")
        self.assertEqual(dataset.identity, loaded.identity)
        self.assertEqual(loaded.get(0), example)
        self.assertNotEqual(dataset.identity, InMemorySFTDataset([example], tokenizer="other", renderer="render").identity)

    def test_task_source_identity_and_payload_isolation(self):
        payload = {"question": ["q"]}
        source = InMemoryTaskSource([Task("one", "train", "family", payload)])
        original = source.identity
        payload["question"].append("caller mutation")
        source.get(0).payload["question"].append("consumer mutation")
        self.assertEqual(source.get(0).payload, {"question": ["q"]})
        self.assertEqual(source.identity, original)
        with self.assertRaises(ValueError):
            InMemoryTaskSource([Task("one", "train", "family", {}), Task("one", "eval", "family", {})])


if __name__ == "__main__":
    unittest.main()
