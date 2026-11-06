"""Tests for select_checkpoint.py on small fake run directories."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import select_checkpoint  # noqa: E402
from test_guard import eval_row, write_adapter, write_metrics, write_stream  # noqa: E402


def add_checkpoints(run: Path, steps, adapters: bool = True) -> None:
    for step in steps:
        meta = run / "checkpoints" / f"step_{step}" / "trainer" / ".metadata"
        meta.parent.mkdir(parents=True, exist_ok=True)
        meta.write_bytes(b"")
        if adapters:
            write_adapter(select_checkpoint.adapter_dir(run, step))


class Choose(unittest.TestCase):
    def test_highest_score_wins(self):
        self.assertEqual(select_checkpoint.choose({0: 0.9, 15: 0.4, 30: 0.6, 45: 0.5}, {15, 30, 45}), 30)

    def test_tie_goes_to_earlier_step(self):
        self.assertEqual(select_checkpoint.choose({15: 0.6, 30: 0.6}, {15, 30}), 15)

    def test_base_model_is_never_chosen(self):
        self.assertIsNone(select_checkpoint.choose({0: 0.9}, {0}))

    def test_step_without_checkpoint_is_skipped(self):
        self.assertEqual(select_checkpoint.choose({15: 0.4, 30: 0.9}, {15}), 15)


class Select(unittest.TestCase):
    def test_select_from_run_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            write_metrics(run, [
                eval_row(0, 0.30, 0.0, 1.0),
                {"step": 3, "train/agg/all/has_error/mean": 0.0},
                eval_row(15, 0.45, 0.0, 1.0, agent="other-name"),
                eval_row(30, 0.52, 0.0, 1.0),
                eval_row(30, 0.10, 0.0, 1.0, env="other-env"),
                eval_row(45, 0.70, 0.0, 1.0),
            ])
            add_checkpoints(run, [15, 30])  # step 45 was evaluated but has no checkpoint
            result = select_checkpoint.select(run, "val")
            self.assertEqual(result["chosen_step"], 30)
            self.assertEqual(result["baseline"], 0.30)
            self.assertTrue(result["beats_baseline"])
            self.assertEqual(result["with_checkpoint"], [15, 30])
            self.assertEqual(Path(result["checkpoint"]), run / "checkpoints" / "step_30")
            self.assertEqual(Path(result["adapter"]), run / "adapters" / "step_30")

    def test_step_without_adapter_copy_is_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            write_metrics(run, [eval_row(0, 0.3, 0.0, 1.0), eval_row(15, 0.4, 0.0, 1.0), eval_row(30, 0.9, 0.0, 1.0)])
            add_checkpoints(run, [15])
            add_checkpoints(run, [30], adapters=False)
            self.assertEqual(select_checkpoint.select(run, "val")["chosen_step"], 15)

    def test_main_writes_output_and_fails_without_checkpoints(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            write_metrics(run, [eval_row(0, 0.3, 0.0, 1.0), eval_row(15, 0.4, 0.0, 1.0)])
            out = run / "selection.json"
            self.assertEqual(select_checkpoint.main(["--run-dir", tmp, "--out", str(out)]), 1)
            self.assertIsNone(json.loads(out.read_text())["chosen_step"])
            add_checkpoints(run, [15])
            self.assertEqual(select_checkpoint.main(["--run-dir", tmp, "--out", str(out)]), 0)
            self.assertEqual(json.loads(out.read_text())["chosen_step"], 15)


def eval_episode(step: int, start: int, end: int, env: str = "val") -> dict:
    """The fields of an eval episode record that policy_spans reads (verifiers trace.py, EvalWorkInfo)."""
    return {"id": f"e{step}-{start}-{end}", "env": {"name": env},
            "run": {"type": "train", "id": "r", "work": {"type": "eval", "step": step,
                                                          "policy": {"start": start, "end": end}}},
            "traces": []}


class PolicySpans(unittest.TestCase):
    def test_spans_per_eval_step(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            train = {"id": "t", "env": {"name": "train"},
                     "run": {"type": "train", "id": "r", "work": {"type": "train", "step": 3}}, "traces": []}
            write_stream(run, [eval_episode(0, 0, 0), eval_episode(15, 15, 15), eval_episode(15, 15, 17),
                               eval_episode(15, 14, 14, env="other"), train])
            spans = select_checkpoint.policy_spans(run, "val")
            self.assertEqual(spans[0], {"episodes": 1, "min_start": 0, "max_end": 0})
            self.assertEqual(spans[15], {"episodes": 2, "min_start": 15, "max_end": 17})
            self.assertNotIn(3, spans)

    def test_spans_recorded_in_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            write_metrics(run, [eval_row(0, 0.3, 0.0, 1.0), eval_row(15, 0.4, 0.0, 1.0)])
            add_checkpoints(run, [15])
            write_stream(run, [eval_episode(15, 15, 16)])
            result = select_checkpoint.select(run, "val")
            self.assertEqual(result["policy_spans"]["15"], {"episodes": 1, "min_start": 15, "max_end": 16})
            self.assertEqual(result["chosen_policy_span"]["drift"], 1)

    def test_span_error_is_recorded_and_does_not_block_the_choice(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            write_metrics(run, [eval_row(0, 0.3, 0.0, 1.0), eval_row(15, 0.4, 0.0, 1.0)])
            add_checkpoints(run, [15])
            original = select_checkpoint.policy_spans

            def broken(*_):
                raise OSError("unreadable chunk")

            select_checkpoint.policy_spans = broken
            try:
                result = select_checkpoint.select(run, "val")
            finally:
                select_checkpoint.policy_spans = original
            self.assertEqual(result["chosen_step"], 15)
            self.assertEqual(result["policy_spans"], {})
            self.assertIn("unreadable chunk", result["policy_spans_error"])

    def test_spans_can_be_skipped(self):
        import os

        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            write_metrics(run, [eval_row(0, 0.3, 0.0, 1.0), eval_row(15, 0.4, 0.0, 1.0)])
            add_checkpoints(run, [15])
            write_stream(run, [eval_episode(15, 15, 16)])
            os.environ["SELECT_SKIP_SPANS"] = "1"
            try:
                result = select_checkpoint.select(run, "val")
            finally:
                del os.environ["SELECT_SKIP_SPANS"]
            self.assertEqual(result["chosen_step"], 15)
            self.assertTrue(result["policy_spans_skipped"])
            self.assertEqual(result["policy_spans"], {})


if __name__ == "__main__":
    unittest.main()
