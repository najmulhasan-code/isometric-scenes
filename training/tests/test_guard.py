"""Tests for guard.py on small fake run directories.

Run with `python -m pytest training/tests` or
`python -m unittest discover -s training/tests`.
"""

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import guard  # noqa: E402

LN_VOCAB = math.log(guard.VOCAB_SIZE)


def write_metrics(run_dir: Path, rows: list[dict]) -> None:
    path = run_dir / "monitors" / "file" / "metrics.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def write_stream(run_dir: Path, records: list[dict], name: str = "00000.jsonl") -> Path:
    stream = run_dir / "monitors" / "file" / "traces" / "stream"
    stream.mkdir(parents=True, exist_ok=True)
    path = stream / name
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    return path


def eval_row(step: int, mean: float, low: float, high: float, env: str = "val", agent: str = "agent",
             subset: str = "effective", error: float = 0.0) -> dict:
    prefix = f"eval/{env}/{subset}/{agent}/reward"
    return {"step": step, "time": 0.0, f"{prefix}/mean": mean, f"{prefix}/min": low, f"{prefix}/max": high,
            f"eval/{env}/all/has_error/mean": error, f"eval/{env}/policy_version": float(step)}


def trainer_row(step: int, loss: float = 0.01, entropy: float = 0.8, mismatch: float = 0.001) -> dict:
    return {"step": step, "time": 0.0, "loss/mean": loss, "entropy/all/mean": entropy,
            "mismatch_kl/all/mean": mismatch}


def grad_row(step: int, grad: float = 0.5) -> dict:
    return {"step": step, "time": 0.0, "optim/lr": 3e-6, "optim/grad_norm": grad}


def train_row(step: int, error: float = 0.0) -> dict:
    return {"step": step, "time": 0.0, "train/agg/all/has_error/mean": error,
            "train/agg/all/agent/reward/mean": 0.5}


def episode(message: dict, judge_reply: str = '{"scene": true}') -> dict:
    """An episode record: the policy's reply in the nodes, a judge call under info."""
    return {"id": "e1", "traces": [{
        "nodes": [{"messages": [{"role": "system", "content": "draw"}, {"role": "user", "content": "a shop"},
                                message]}],
        "info": {"judge_calls": [{"response": {"message": {"role": "assistant", "content": judge_reply}}}]},
    }]}


class StopRules(unittest.TestCase):
    def test_healthy_run_has_no_stop_reason(self):
        rows = [eval_row(0, 0.4, 0.0, 1.0), train_row(1), trainer_row(1), grad_row(1)]
        self.assertIsNone(guard.stop_reason(rows))

    def test_equal_baseline_rewards_stop(self):
        rule, reason = guard.stop_reason([eval_row(0, 0.0, 0.0, 0.0)])
        self.assertEqual(rule, "no_signal")
        self.assertIn("no learning signal", reason)

    def test_equal_rewards_later_do_not_stop(self):
        self.assertIsNone(guard.baseline_without_signal([eval_row(0, 0.3, 0.0, 1.0), eval_row(15, 1.0, 1.0, 1.0)]))

    def test_agent_segment_is_matched_by_pattern(self):
        self.assertIsNotNone(guard.baseline_without_signal([eval_row(0, 0.5, 0.5, 0.5, agent="policy-seat")]))

    def test_all_subset_used_when_effective_missing(self):
        self.assertIsNotNone(guard.baseline_without_signal([eval_row(0, 0.2, 0.2, 0.2, subset="all")]))

    def test_effective_subset_preferred(self):
        row = eval_row(0, 0.1, 0.1, 0.1, subset="all")
        row.update(eval_row(0, 0.5, 0.0, 1.0))
        self.assertIsNone(guard.baseline_without_signal([row]))

    def test_three_consecutive_high_error_steps_stop(self):
        rows = [train_row(1, 0.2), train_row(2, 0.11), train_row(3, 0.15)]
        rule, reason = guard.stop_reason(rows)
        self.assertEqual(rule, "error_rate")
        self.assertIn("steps 1-3", reason)

    def test_two_high_error_steps_do_not_stop(self):
        self.assertIsNone(guard.error_rate_too_high([train_row(1, 0.5), train_row(2, 0.5), train_row(3, 0.0)]))

    def test_interrupted_streak_does_not_stop(self):
        rows = [train_row(1, 0.5), train_row(2, 0.5), train_row(3, 0.05), train_row(4, 0.5), train_row(5, 0.5)]
        self.assertIsNone(guard.error_rate_too_high(rows))

    def test_gap_in_steps_breaks_the_streak(self):
        self.assertIsNone(guard.error_rate_too_high([train_row(1, 0.5), train_row(2, 0.5), train_row(4, 0.5)]))

    def test_error_rate_at_limit_does_not_count(self):
        self.assertIsNone(guard.error_rate_too_high([train_row(s, 0.10) for s in range(1, 6)]))

    def test_uniform_entropy_at_step_one_stops(self):
        rule, reason = guard.stop_reason([trainer_row(1, entropy=LN_VOCAB - 0.01)])
        self.assertEqual(rule, "first_step")
        self.assertIn("lm_head", reason)

    def test_uniform_loss_at_step_one_stops(self):
        self.assertIn("lm_head", guard.first_step_broken([trainer_row(1, loss=LN_VOCAB)]))

    def test_zero_grad_norm_at_step_one_stops(self):
        self.assertIn("gradient norm is 0", guard.first_step_broken([grad_row(1, 0.0)]))

    def test_zero_grad_norm_later_is_not_a_step_one_failure(self):
        self.assertIsNone(guard.first_step_broken([grad_row(1), grad_row(7, 0.0)]))

    def test_wall_clock_cap(self):
        self.assertIsNone(guard.over_time(0.0, 20, now=19.9 * 3600))
        self.assertIn("20 h", guard.over_time(0.0, 20, now=20 * 3600))

    def test_latest_step_is_the_highest_trainer_step(self):
        rows = [trainer_row(1), trainer_row(3), train_row(9), grad_row(5)]
        self.assertEqual(guard.latest_step(rows), 3)
        self.assertIsNone(guard.latest_step([train_row(1)]))

    def test_stall_after_the_limit_without_progress(self):
        self.assertEqual(guard.STALL_MINUTES, 120.0)
        self.assertIsNone(guard.stall_reason(0.0, 120, now=119 * 60))
        self.assertIn("no progress", guard.stall_reason(0.0, 120, now=120 * 60))

    def test_progress_marker_moves_with_steps_rows_and_traces(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            self.assertEqual(guard.progress_marker(run, []), (None, 0, 0))
            base = [eval_row(0, 0.3, 0.0, 1.0)]
            write_metrics(run, base)
            first = guard.progress_marker(run, guard.read_rows(guard.metrics_files(run)))
            # A validation epoch finishing: a new step-keyed row, no new trainer step.
            write_metrics(run, base + [eval_row(0, 0.3, 0.0, 1.0, env="val2")])
            second = guard.progress_marker(run, guard.read_rows(guard.metrics_files(run)))
            self.assertNotEqual(first, second)
            # Rollouts landing in the trace stream.
            write_stream(run, [{"n": 1}])
            third = guard.progress_marker(run, guard.read_rows(guard.metrics_files(run)))
            self.assertNotEqual(second, third)
            # A new trainer step.
            write_metrics(run, base + [eval_row(0, 0.3, 0.0, 1.0, env="val2"), trainer_row(1)])
            self.assertEqual(guard.progress_marker(run, guard.read_rows(guard.metrics_files(run)))[0], 1)

    def test_time_keyed_rows_alone_are_not_progress(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            rows = [eval_row(0, 0.3, 0.0, 1.0), trainer_row(1)]
            write_metrics(run, rows)
            before = guard.progress_marker(run, guard.read_rows(guard.metrics_files(run)))
            # A hung pipeline: the orchestrator keeps appending timer rows with step null.
            timer = [{"step": None, "time": float(t), "inference/kv_usage": 0.1} for t in range(50)]
            write_metrics(run, rows + timer)
            after = guard.progress_marker(run, guard.read_rows(guard.metrics_files(run)))
            self.assertEqual(before, after)
            self.assertIsNotNone(guard.stall_reason(0.0, guard.STALL_MINUTES, now=guard.STALL_MINUTES * 60))

    def test_stop_file_is_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "GUARD_STOP"
            guard.write_stop(path, "time_cap", "wall-clock cap of 20 h reached")
            self.assertEqual(json.loads(path.read_text())["rule"], "time_cap")


class Reading(unittest.TestCase):
    def test_partial_last_line_and_time_rows_are_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            write_metrics(run, [train_row(1, 0.2), {"step": None, "inference/x": 1}])
            path = run / "monitors" / "file" / "metrics.jsonl"
            with open(path, "a", encoding="utf-8") as f:
                f.write('{"step": 2, "train/agg/all/has_err')
            rows = guard.read_rows(guard.metrics_files(run))
            self.assertEqual(guard.values(rows, guard.TRAIN_ERROR_KEY), {1: 0.2})

    def test_no_metrics_yet(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(guard.read_rows(guard.metrics_files(Path(tmp))), [])

    def test_stream_chunks_in_order_skipping_tmp_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            write_stream(run, [{"n": 2}], "00001.jsonl")
            write_stream(run, [{"n": 0}, {"n": 1}], "00000.jsonl")
            (write_stream(run, [], "00002.jsonl.zst.tmp")).write_bytes(b"partial")
            lines = [json.loads(line) for line in guard.stream_lines(run)]
            self.assertEqual(lines, [{"n": 0}, {"n": 1}, {"n": 2}])

    def test_missing_stream_yields_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(list(guard.stream_lines(Path(tmp))), [])


def write_adapter(directory: Path, finished: bool = False) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "adapter_model.safetensors").write_bytes(b"weights")
    (directory / "adapter_config.json").write_text('{"r": 32}', encoding="utf-8")
    if finished:
        (directory / ".finished").touch()


def make_smoke_run(run: Path, *, error: float = 0.0, eval_error: float = 0.0, grad: float = 0.4,
                   checkpoints=(1, 2), reply: dict | None = None) -> None:
    write_metrics(run, [
        eval_row(0, 0.3, 0.0, 1.0, error=eval_error), train_row(1, error), trainer_row(1), grad_row(1, grad),
        eval_row(1, 0.4, 0.0, 1.0), train_row(2), trainer_row(2), grad_row(2, grad), eval_row(2, 0.4, 0.0, 1.0),
    ])
    for step in checkpoints:
        meta = run / "checkpoints" / f"step_{step}" / "trainer" / ".metadata"
        meta.parent.mkdir(parents=True, exist_ok=True)
        meta.write_bytes(b"")
        write_adapter(guard.adapter_dir(run, step))
    message = reply if reply is not None else {"role": "assistant", "content": "<svg></svg>", "reasoning_content": None}
    write_stream(run, [episode(message)])


class Smoke(unittest.TestCase):
    def run_smoke(self, **kwargs) -> list[str]:
        with tempfile.TemporaryDirectory() as tmp:
            make_smoke_run(Path(tmp), **kwargs)
            return guard.smoke(Path(tmp), steps=2)

    def test_clean_smoke_passes(self):
        self.assertEqual(self.run_smoke(), [])

    def test_small_error_rate_is_tolerated(self):
        # 8% on one of two steps: 4% overall, under the 5% tolerance.
        self.assertEqual(self.run_smoke(error=0.08), [])

    def test_training_errors_above_tolerance_fail(self):
        self.assertTrue(any("training error rate" in f for f in self.run_smoke(error=0.25)))

    def test_validation_errors_above_tolerance_fail(self):
        self.assertTrue(any("validation error rate" in f for f in self.run_smoke(eval_error=0.5)))

    def test_zero_gradients_fail(self):
        self.assertTrue(any("gradient norm" in f for f in self.run_smoke(grad=0.0)))

    def test_missing_checkpoint_fails(self):
        failures = self.run_smoke(checkpoints=(1,))
        self.assertIn("no checkpoint for step 2", failures)
        self.assertIn("no adapter copy for step 2", failures)

    def test_opening_think_tag_fails(self):
        failures = self.run_smoke(reply={"role": "assistant", "content": "<think>hmm</think><svg/>"})
        self.assertTrue(any("thinking is on" in f for f in failures))

    def test_closing_think_tag_alone_fails(self):
        failures = self.run_smoke(reply={"role": "assistant", "content": "plan</think><svg/>"})
        self.assertTrue(any("thinking is on" in f for f in failures))

    def test_reasoning_content_fails(self):
        failures = self.run_smoke(reply={"role": "assistant", "content": "<svg/>", "reasoning_content": "plan"})
        self.assertTrue(any("thinking is on" in f for f in failures))

    def test_one_stray_tag_among_many_completions_passes(self):
        # Thinking off, but the model wrote one "</think>" in one of 20 answers.
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            replies = [{"role": "assistant", "content": "<svg/>"}] * 19 + [{"role": "assistant", "content": "x</think><svg/>"}]
            write_stream(run, [episode(r) for r in replies])
            self.assertEqual(guard.check_thinking_off(run), [])

    def test_many_tagged_completions_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            replies = [{"role": "assistant", "content": "<svg/>"}] * 17 + [{"role": "assistant", "content": "x</think><svg/>"}] * 3
            write_stream(run, [episode(r) for r in replies])
            self.assertTrue(any("thinking is on" in f for f in guard.check_thinking_off(run)))

    def test_judge_replies_under_info_are_not_checked(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            write_stream(run, [episode({"role": "assistant", "content": "<svg/>"}, judge_reply="<think>x</think>")])
            self.assertEqual(guard.check_thinking_off(run), [])

    def test_no_completion_fails(self):
        failures = self.run_smoke(reply={"role": "assistant", "content": ""})
        self.assertIn("no sampled completion found in the trace stream", failures)

    def test_no_metrics_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertTrue(guard.smoke(Path(tmp), steps=2))


class Adapters(unittest.TestCase):
    def test_finished_validation_broadcasts_are_copied_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            for step in (14, 15, 16):
                write_adapter(run / "broadcasts" / f"step_{step}", finished=True)
            write_adapter(run / "broadcasts" / "step_30")  # still being written
            self.assertEqual(guard.snapshot_adapters(run, 15), [15])
            copied = guard.adapter_dir(run, 15)
            self.assertEqual(sorted(p.name for p in copied.iterdir()), ["adapter_config.json", "adapter_model.safetensors"])
            self.assertEqual(guard.snapshot_adapters(run, 15), [])  # already copied
            (run / "broadcasts" / "step_30" / ".finished").touch()
            self.assertEqual(guard.snapshot_adapters(run, 15), [30])
            self.assertEqual(guard.check_adapters(run, [15, 30, 45]), ["no adapter copy for step 45"])

    def test_no_broadcasts_yet(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(guard.snapshot_adapters(Path(tmp), 15), [])


class Projection(unittest.TestCase):
    @staticmethod
    def rows(steps, seconds=60.0):
        return [dict(trainer_row(step), time=step * seconds) for step in steps]

    def test_nothing_before_step_ten(self):
        self.assertIsNone(guard.hours_left(self.rows(range(1, 10)), 150))

    def test_step_ten_uses_steps_one_to_ten(self):
        step, hours = guard.hours_left(self.rows(range(1, 11)), 150)
        self.assertEqual(step, 10)
        self.assertAlmostEqual(hours, 60.0 * 140 / 3600)

    def test_mean_of_the_last_ten_intervals(self):
        # Steps 1-10 at 60 s each, steps 11-20 at 120 s each: only the last ten count.
        rows = self.rows(range(1, 11)) + [dict(trainer_row(s), time=600.0 + (s - 10) * 120.0) for s in range(11, 21)]
        step, hours = guard.hours_left(rows, 150)
        self.assertEqual(step, 20)
        self.assertAlmostEqual(hours, 120.0 * 130 / 3600)

    def test_done_means_zero_hours_left(self):
        self.assertEqual(guard.hours_left(self.rows(range(1, 151)), 150)[1], 0.0)

    def test_projection_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "projected_hours"
            line = guard.write_projection(path, 20, 2.1666, 150)
            self.assertEqual(path.read_text().splitlines(), ["2.17", line])
            self.assertIn("step 20 of 150", line)


class Mismatch(unittest.TestCase):
    def test_warnings_only_above_the_level(self):
        rows = [trainer_row(1, mismatch=0.019), trainer_row(2, mismatch=0.5)]
        self.assertEqual(list(guard.mismatch_warnings(rows)), [2])

    def test_watch_warnings_start_at_step_fifteen(self):
        rows = [trainer_row(14, mismatch=0.5), trainer_row(15, mismatch=0.5)]
        self.assertEqual(list(guard.mismatch_warnings(rows, guard.MISMATCH_WATCH_FROM_STEP)), [15])

    def test_high_mismatch_does_not_fail_the_smoke(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            make_smoke_run(run)
            write_metrics(run, guard.read_rows(guard.metrics_files(run)) + [trainer_row(2, mismatch=0.5)])
            self.assertEqual(guard.smoke(run, 2), [])

    def test_high_mismatch_in_the_full_run_never_stops_it(self):
        self.assertIsNone(guard.stop_reason([trainer_row(1), trainer_row(15, mismatch=5.0)]))


class LoraLoading(unittest.TestCase):
    ERROR = ("ValueError: While loading /job/runs/smoke/broadcasts/step_1, expected target modules in "
             "{'q_proj', 'o_proj'} but received ['model.layers.0.linear_attn.in_proj_b']. "
             "Please verify that the loaded LoRA module is correct")

    def write_log(self, run: Path, text: str) -> None:
        log = run / "logs" / "attempt_1" / "inference.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(text, encoding="utf-8")

    def test_target_module_error_fails_the_smoke(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            make_smoke_run(run)
            self.write_log(run, "INFO started\n" + self.ERROR + "\n")
            failures = guard.smoke(run, 2)
            self.assertTrue(any("vLLM rejected LoRA target modules" in f for f in failures))

    def test_clean_logs_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            self.write_log(run, "INFO loaded LoRA adapter\n")
            self.assertEqual(guard.check_lora_module_errors(run), [])

    def test_version_not_applied_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            self.write_log(run, "WARNING Inference did not apply policy v2 within 1200s - proceeding anyway\n")
            self.assertTrue(guard.check_adapter_loaded(run, [], 2))

    def test_final_eval_on_an_older_version_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            row = eval_row(2, 0.4, 0.0, 1.0)
            row["eval/val/policy_version"] = 1.0
            self.assertTrue(guard.check_adapter_loaded(Path(tmp), [row], 2))
            self.assertEqual(guard.check_adapter_loaded(Path(tmp), [eval_row(2, 0.4, 0.0, 1.0)], 2), [])


class StallClock(unittest.TestCase):
    def test_progress_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / guard.PROGRESS_FILE
            guard.save_progress(path, (3, 10, 500), 1234.5)
            self.assertEqual(guard.load_progress(path), ((3, 10, 500), 1234.5))

    def test_missing_or_broken_progress_starts_fresh(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / guard.PROGRESS_FILE
            marker, when = guard.load_progress(path)
            self.assertIsNone(marker)
            path.write_text("not json", encoding="utf-8")
            self.assertIsNone(guard.load_progress(path)[0])

    def test_restarted_guard_keeps_the_stall_clock(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            write_metrics(run, [trainer_row(1)])
            marker = guard.progress_marker(run, guard.read_rows(guard.metrics_files(run)))
            # An earlier guard saw this same progress three hours ago.
            guard.save_progress(run / guard.PROGRESS_FILE, marker, guard.time.time() - 3 * 3600)
            original_alive, original_stop = guard.alive, guard.stop
            guard.alive, guard.stop = (lambda _pgid: True), (lambda _pgid, grace=60.0: None)
            try:
                result = guard.watch(run, 0, 20, run / "GUARD_STOP", poll=0)
            finally:
                guard.alive, guard.stop = original_alive, original_stop
            self.assertEqual(result, 3)
            self.assertEqual(json.loads((run / "GUARD_STOP").read_text())["rule"], "stalled")


class Robustness(unittest.TestCase):
    def test_progress_marker_tolerates_a_vanished_chunk(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            gone = write_stream(run, [{"n": 1}])
            original = guard.stream_chunks
            guard.stream_chunks = lambda _run: [gone]
            try:
                gone.unlink()  # sealed and renamed between the listing and the stat
                self.assertEqual(guard.progress_marker(run, [])[2], 0)
            finally:
                guard.stream_chunks = original

    def test_watch_survives_an_error_in_one_poll(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            calls = {"alive": 0, "read": 0}
            original_alive, original_read = guard.alive, guard.read_rows

            def alive(_pgid):
                calls["alive"] += 1
                return calls["alive"] <= 2

            def read_rows(paths):
                calls["read"] += 1
                if calls["read"] == 1:
                    raise RuntimeError("file changed while reading")
                return original_read(paths)

            guard.alive, guard.read_rows = alive, read_rows
            try:
                result = guard.watch(run, 0, 20, run / "GUARD_STOP", poll=0)
            finally:
                guard.alive, guard.read_rows = original_alive, original_read
            self.assertEqual(result, 0)
            self.assertGreaterEqual(calls["read"], 2)  # the second poll ran
            self.assertFalse((run / "GUARD_STOP").exists())

    def test_two_copiers_do_not_clash(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            write_adapter(run / "broadcasts" / "step_15", finished=True)
            self.assertEqual(guard.snapshot_adapters(run, 15), [15])
            self.assertEqual(guard.snapshot_adapters(run, 15), [])
            leftovers = [p.name for p in (run / "adapters").iterdir() if "partial" in p.name]
            self.assertEqual(leftovers, [])

    def test_copied_adapter_survives_the_broadcast_cleanup(self):
        import shutil

        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            write_adapter(run / "broadcasts" / "step_15", finished=True)
            guard.snapshot_adapters(run, 15)
            shutil.rmtree(run / "broadcasts")
            self.assertEqual((guard.adapter_dir(run, 15) / "adapter_model.safetensors").read_bytes(), b"weights")

    def test_copier_loop_ends_with_the_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            write_adapter(run / "broadcasts" / "step_15", finished=True)
            original = guard.alive
            guard.alive = lambda _pgid: False
            try:
                guard.copy_adapters_until(run, 15, pgid=1, every=0)
            finally:
                guard.alive = original
            self.assertTrue(guard.adapter_dir(run, 15).exists())


class Main(unittest.TestCase):
    def test_smoke_command_exit_codes(self):
        with tempfile.TemporaryDirectory() as tmp:
            make_smoke_run(Path(tmp))
            self.assertEqual(guard.main(["smoke", "--run-dir", tmp]), 0)
        with tempfile.TemporaryDirectory() as tmp:
            make_smoke_run(Path(tmp), checkpoints=())
            self.assertEqual(guard.main(["smoke", "--run-dir", tmp]), 1)

    def test_thinking_command_exit_codes(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_stream(Path(tmp), [episode({"role": "assistant", "content": "<svg/>"})])
            self.assertEqual(guard.main(["thinking", "--run-dir", tmp]), 0)
        with tempfile.TemporaryDirectory() as tmp:
            write_stream(Path(tmp), [episode({"role": "assistant", "content": "<think></think><svg/>"})])
            self.assertEqual(guard.main(["thinking", "--run-dir", tmp]), 1)


if __name__ == "__main__":
    unittest.main()
