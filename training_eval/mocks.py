"""Explicitly synthetic adapters for offline smoke tests, not a repo environment."""
from .contracts import SFTExample, Task, Trajectory, Verification
from .data import InMemorySFTDataset, InMemoryTaskSource


class MockEnvironment:
    version = "mock-two-turn-v1"

    def run(self, task, policy, request):
        prompt = (10, int(task.payload.get("prompt_id", 11)))
        turns = []
        observations = []
        for turn in range(min(2, request.max_turns)):
            sample = policy.sample(prompt, max_tokens=request.max_tokens,
                                   temperature=request.temperature, seed=request.seed + turn)
            turns.append(sample)
            if turn == 0 and request.max_turns > 1:
                observations.append("synthetic tool observation")
                prompt = prompt + sample.tokens + (20, 21)
        return Trajectory(task.task_id, request.episode_id, policy.version, self.version,
                          tuple(turns), turns[-1].text, metrics={
                              "output_tokens": sum(len(t.tokens) for t in turns),
                              "tool_calls": len(observations), "cost": None},
                          observations=tuple(observations))


class MockVerifier:
    version = "mock-binary-reward-v1"

    def verify(self, task, trajectory, *, role):
        reward = float(trajectory.answer == "1")
        return Verification("resolved", reward, {"synthetic_success": reward},
                            "Synthetic fixture: reward one for output 1")


def demo_inputs(config=None):
    return {
        "sft_dataset": InMemorySFTDataset([SFTExample("demo-sft", (10, 49), (0, 1))],
                                         tokenizer="fake-tokenizer-v1", renderer="tokens-v1"),
        "tasks": InMemoryTaskSource([Task("train-1", "train", "train-family", {"prompt_id": 11})]),
        "evaluation_tasks": InMemoryTaskSource([Task("dev-1", "development", "dev-family", {"prompt_id": 12})]),
        "environment": MockEnvironment(), "verifier": MockVerifier(),
    }
