"""Tests for the parts of rescore.py that need no verifiers: the summary and the failure rate."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import rescore  # noqa: E402

ROWS = [
    {"env": "test", "status": "scored", "reward": 1.0},
    {"env": "test", "status": "scored", "reward": 0.5},
    {"env": "test", "status": "error", "reward": None},
    {"env": "test_unseen", "status": "no_completion", "reward": None},
]


class Summary(unittest.TestCase):
    def test_counts_by_status_and_mean_of_scored_rows(self):
        summary = rescore.summarize(ROWS)
        self.assertEqual(summary["test"], {"rollouts": 3, "scored": 2, "error": 1, "no_completion": 0,
                                           "mean_reward": 0.75})
        self.assertEqual(summary["test_unseen"], {"rollouts": 1, "scored": 0, "error": 0, "no_completion": 1,
                                                  "mean_reward": None})

    def test_failure_rate_counts_every_row_not_scored(self):
        self.assertEqual(rescore.failure_rate(ROWS), 0.5)
        self.assertEqual(rescore.failure_rate([]), 1.0)


class ScoringFailureOnly(unittest.TestCase):
    def test_scoring_errors_are_cleared_for_rescoring(self):
        render = ("TaskError", "task SceneTask scoring: RuntimeError: render script failed: no frames")
        timed_out = ("TaskError", "task SceneTask scoring timed out")
        self.assertTrue(rescore.scoring_failure_only([render], "task_error"))
        self.assertTrue(rescore.scoring_failure_only([render, timed_out], "task_error"))
        self.assertTrue(rescore.scoring_failure_only([], "scoring_timeout"))

    def test_generation_errors_mean_no_completion(self):
        self.assertFalse(rescore.scoring_failure_only([("ProviderError", "400 context length exceeded")], None))
        self.assertFalse(rescore.scoring_failure_only([("HarnessError", "exit 1")], None))
        self.assertFalse(rescore.scoring_failure_only([("TaskError", "task SceneTask setup: boom")], None))
        self.assertFalse(rescore.scoring_failure_only(
            [("TaskError", "task SceneTask scoring: x"), ("ProviderError", "timeout")], None))
        self.assertFalse(rescore.scoring_failure_only([], "agent_timeout"))
        self.assertFalse(rescore.scoring_failure_only([], None))


if __name__ == "__main__":
    unittest.main()
