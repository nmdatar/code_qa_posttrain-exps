import unittest
from qa_eval.rl import prepare_group


class RLContractTests(unittest.TestCase):
    def record(self, reward, tier="accepted"):
        return {"judge_role": "training", "split": "train", "task_hash": "a", "experiment_hash": "b",
                "reward_version": "1.0", "tier": tier, "reward": reward}

    def test_unresolved_excludes_whole_group(self):
        result = prepare_group([self.record(.9), self.record(None, "unresolved")])
        self.assertIsNone(result["rewards"])
        self.assertEqual(result["excluded_episodes"], 2)

    def test_bands_survive_relative_advantage(self):
        result = prepare_group([self.record(0, "failed"), self.record(.2, "partial"), self.record(.9), self.record(1)])
        self.assertEqual(result["advantages"], sorted(result["advantages"]))

    def test_equal_rewards_do_not_invent_signal(self):
        self.assertEqual(prepare_group([self.record(.9), self.record(.9)])["advantages"], [0, 0])

    def test_eval_leakage_rejected(self):
        r = self.record(.9)
        r["split"] = "final_test"
        with self.assertRaises(ValueError):
            prepare_group([r])
