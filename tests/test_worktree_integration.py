"""Regression checks for interfaces that collided across completed worktrees."""
import unittest
from unittest.mock import patch
from agent_harness.cli import main
from agent_harness.runner import AgentRunner, run_episode
from agent_harness.training_runner import run_episode as training_run_episode


class WorktreeIntegrationTests(unittest.TestCase):
    def test_training_entrypoint_preserved_alongside_research_runner(self):
        self.assertIs(run_episode, training_run_episode)
        self.assertTrue(callable(AgentRunner))

    def test_legacy_sandbox_commands_still_dispatch(self):
        for args in (["validate", "--manifest", "env.json"],
                     ["run", "--manifest", "env.json", "--", "python3", "-V"],
                     ["run", "--manifest=env.json", "--", "python3", "-V"]):
            with self.subTest(args=args), patch('agent_harness.sandbox_cli.main', return_value=0) as sandbox:
                self.assertEqual(main(args), 0)
                sandbox.assert_called_once_with(args)
