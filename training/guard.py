"""Checks on a prime-rl run, read from the files it writes. Standard library only
(plus pyzstd, a prime-rl dependency, for sealed trace chunks).

Commands:

  guard.py smoke --run-dir DIR [--steps N] [--max-error-rate R]
      After the smoke run: errored rollouts at most R (default 5%), nonzero
      gradients, no LoRA target-module error in the inference logs, the last
      step's adapter loaded by inference, the checkpoints and adapter copies
      written, thinking off in the sampled completions. Exit 0 if all pass, 1
      otherwise. A trainer-inference mismatch KL above 0.02 is a warning only.

  guard.py adapters --run-dir DIR --interval N [--every S --pgid G]
      Copy the LoRA adapter of every finished weight broadcast whose step is a
      multiple of N into <run>/adapters/step_N/ (see snapshot_adapters). With
      --every, repeat every S seconds while process group G is alive.

  guard.py thinking --run-dir DIR
      Thinking off in every sampled completion of a run (train, validation or
      a standalone eval). Exit 0 if so, 1 otherwise.

  guard.py watch --run-dir DIR --pgid N [--max-hours H] [--stall-minutes M]
                 [--stop-file F]
      Next to the full run. When a stop rule fires, stops the run (SIGTERM,
      then SIGKILL, to process group N), writes {"rule", "reason"} as JSON to
      the stop file and exits 3. Exits 0 when the run ends on its own. Also
      copies the adapter of every validation step (--adapter-interval), and
      every 10 steps logs the projected hours left to --total-steps and writes
      them to --projection-file (a projection only; it never stops the run).

Stop rules (`rule` in the stop file):
  no_signal    the step-0 validation rewards are all equal;
  error_rate   more than 10% of rollouts errored on 3 consecutive training steps;
  first_step   at step 1 the entropy or loss is near ln(248320), the sign of an
               untrained lm_head (prime-rl issue #3700), or the gradient norm is 0;
  stalled      no progress for 120 minutes, from the start of the watch or the
               last progress. Progress is a new trainer step, a new step-keyed
               metrics row, or growth of the trace stream (which receives only
               completed episodes), so startup, validation epochs and checkpoint
               writes do not count as a stall. Time-keyed rows (step null), which
               the orchestrator writes on a timer even when the pipeline is hung,
               do not count;
  time_cap     the wall-clock cap is reached. This is a normal end of training.

Metric names, as prime-rl 35fcb52 writes them to
<run>/monitors/file/metrics.jsonl (one JSON object per line, flat keys, with
"step"):
  orchestrator/metrics.py  train/agg/all/has_error/mean
                           eval/<env>/all/has_error/mean
                           eval/<env>/<all|effective>/<agent>/reward/<mean|min|max>
  trainer/rl/train.py      loss/mean, entropy/all/mean, optim/grad_norm
The <agent> segment is matched with a pattern, so its exact name does not matter.
"""

import argparse
import json
import math
import os
import re
import shutil
import signal
import sys
import time
import traceback
from pathlib import Path

VOCAB_SIZE = 248320
"""Qwen3.5 vocabulary: a uniform next-token distribution has entropy ln(VOCAB_SIZE)."""
UNIFORM_TOLERANCE = 1.0
"""How close to ln(VOCAB_SIZE), in nats, counts as uniform."""
MAX_ERROR_RATE = 0.10
ERROR_STEPS = 3
"""Consecutive training steps above MAX_ERROR_RATE that stop the run."""
SMOKE_MAX_ERROR_RATE = 0.05
MISMATCH_KEY = "mismatch_kl/all/mean"
"""Trainer-inference KL on the sampled tokens (trainer/rl/train.py:615-616, logged as
tensor stats; trainer/rl/loss.py:154-158)."""
MISMATCH_WARN_KL = 0.02
"""Warning level, never a stop. prime-rl documents no normal range for it
(prime-rl docs/training.md:134-135 at commit 35fcb52 only call a sustained, growing mean a
warning sign),
so 0.02 is a chosen level."""
MISMATCH_WATCH_FROM_STEP = 15
LORA_MODULE_ERROR = re.compile(r"While loading .*, expected target modules in .* but received ")
"""vLLM's error for adapter modules it does not support (inference/patches.py:886-891)."""
STALL_MINUTES = 120.0

TRAIN_ERROR_KEY = "train/agg/all/has_error/mean"
LOSS_KEY = "loss/mean"
ENTROPY_KEY = "entropy/all/mean"
GRAD_NORM_KEY = "optim/grad_norm"
EVAL_REWARD = re.compile(r"^eval/(?P<env>[^/]+)/(?P<subset>all|effective)/(?P<agent>[^/]+)/reward/(?P<stat>mean|min|max)$")
EVAL_ERROR = re.compile(r"^eval/(?P<env>[^/]+)/all/has_error/mean$")
CHUNK = re.compile(r"^(?P<number>\d+)\.jsonl(?P<sealed>\.zst)?$")
"""Trace-stream chunk names (monitors/file/traces/chunks.py:26-36); anything else,
such as a `.tmp` file left by sealing, is not a chunk."""


# Reading


def metrics_files(run_dir: Path) -> list[Path]:
    """Every metrics file of the run. Trainer and orchestrator share one by default."""
    return sorted((run_dir / "monitors" / "file").glob("metrics*.jsonl"))


def read_rows(paths: list[Path]) -> list[dict]:
    """All rows that parse; a line still being written is skipped."""
    rows = []
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, dict):
                    rows.append(row)
    return rows


def values(rows: list[dict], key: str) -> dict[int, float]:
    """`key` by step; a later row for the same step wins."""
    out = {}
    for row in rows:
        if key in row and row.get("step") is not None:
            out[int(row["step"])] = float(row[key])
    return out


def eval_rewards(rows: list[dict], env: str | None = None) -> dict[str, dict[int, dict[str, float]]]:
    """Validation reward stats by env, then step: {"mean", "min", "max"}.

    Uses the `effective` subset (errored rollouts left out) when present for that step, else `all`."""
    found: dict[tuple[str, int, str], dict[str, float]] = {}
    for row in rows:
        if row.get("step") is None:
            continue
        for key, value in row.items():
            match = EVAL_REWARD.match(key)
            if match and (env is None or match["env"] == env):
                found.setdefault((match["env"], int(row["step"]), match["subset"]), {})[match["stat"]] = float(value)
    out: dict[str, dict[int, dict[str, float]]] = {}
    for (name, step, subset), stats in found.items():
        if subset == "all" and (name, step, "effective") in found:
            continue
        out.setdefault(name, {})[step] = stats
    return out


def stream_chunks(run_dir: Path) -> list[Path]:
    """The trace stream's chunks in order; a chunk is plain while live, zstd once sealed."""
    chunks: dict[int, Path] = {}
    for path in (run_dir / "monitors" / "file" / "traces" / "stream").glob("*"):
        match = CHUNK.match(path.name)
        if match and (int(match["number"]) not in chunks or not match["sealed"]):
            chunks[int(match["number"])] = path
    return [chunks[n] for n in sorted(chunks)]


def stream_lines(run_dir: Path):
    """Every record of the run's trace stream, as bytes, in order."""
    for path in stream_chunks(run_dir):
        if path.suffix == ".zst":
            import pyzstd

            opener = pyzstd.ZstdFile(path, "rb")
        else:
            opener = open(path, "rb")
        with opener as f:
            yield from (line for line in f if line.strip())


# Stop rules. Each returns a reason, or None when the rule does not fire.


def baseline_without_signal(rows: list[dict]) -> str | None:
    for env, steps in eval_rewards(rows).items():
        stats = steps.get(0)
        if stats and "min" in stats and "max" in stats and stats["min"] == stats["max"]:
            return f"step-0 validation rewards on {env} are all {stats['min']:.4f}: no learning signal"
    return None


def error_rate_too_high(rows: list[dict], limit: float = MAX_ERROR_RATE, run: int = ERROR_STEPS) -> str | None:
    streak: list[int] = []
    for step, rate in sorted(values(rows, TRAIN_ERROR_KEY).items()):
        if rate > limit and (not streak or step == streak[-1] + 1):
            streak.append(step)
        elif rate > limit:
            streak = [step]
        else:
            streak = []
        if len(streak) >= run:
            return f"error rate above {limit:.0%} on training steps {streak[0]}-{streak[-1]}"
    return None


def first_step_broken(rows: list[dict], vocab_size: int = VOCAB_SIZE) -> str | None:
    """Step 1 of the trainer: a uniform output distribution, or no gradient."""
    uniform = math.log(vocab_size)
    for key in (ENTROPY_KEY, LOSS_KEY):
        value = values(rows, key).get(1)
        if value is not None and abs(value - uniform) < UNIFORM_TOLERANCE:
            return f"step-1 {key} = {value:.4f}, near ln({vocab_size}) = {uniform:.4f}: lm_head not loaded"
    grad = values(rows, GRAD_NORM_KEY).get(1)
    if grad is not None and grad == 0.0:
        return "step-1 gradient norm is 0"
    return None


def over_time(started: float, max_hours: float, now: float | None = None) -> str | None:
    elapsed = (time.time() if now is None else now) - started
    if elapsed >= max_hours * 3600:
        return f"wall-clock cap of {max_hours:g} h reached"
    return None


def latest_step(rows: list[dict]) -> int | None:
    """The highest trainer step in the metrics (rows with the trainer's loss)."""
    steps = values(rows, LOSS_KEY)
    return max(steps) if steps else None


def progress_marker(run_dir: Path, rows: list[dict]) -> tuple:
    """Changes whenever the run makes progress: the highest trainer step, the number
    of step-keyed metrics rows, and the bytes written to the trace stream (chunks
    and index). Rows with step null are written on a timer (orchestrator/
    periodic_logger.py:50, inference_metrics.py:358) and are left out. The stream
    receives an episode only once it completes (orchestrator.py:574-585, file
    monitor log_episodes); live rollouts go to traces/live/, which is not read."""
    traces = run_dir / "monitors" / "file" / "traces"
    stream = stream_chunks(run_dir) + list(traces.glob("*.index.jsonl")) if traces.is_dir() else []
    return (
        latest_step(rows),
        sum(row.get("step") is not None for row in rows),
        sum(file_size(path) for path in stream),
    )


def file_size(path: Path) -> int:
    """Size of a file, 0 if it is gone: sealing renames trace chunks while the guard reads."""
    try:
        return path.stat().st_size
    except OSError:
        return 0


def stall_reason(last_change: float, minutes: float, now: float | None = None) -> str | None:
    """Fires when there has been no progress for `minutes` since `last_change`."""
    idle = (time.time() if now is None else now) - last_change
    if idle >= minutes * 60:
        return f"no progress (no new step, no new metrics or traces) for {idle / 60:.0f} min"
    return None


RULES = (("no_signal", baseline_without_signal), ("error_rate", error_rate_too_high), ("first_step", first_step_broken))


def stop_reason(rows: list[dict]) -> tuple[str, str] | None:
    """(rule, reason) of the first rule that fires."""
    for name, rule in RULES:
        if reason := rule(rows):
            return name, reason
    return None


# Smoke and thinking checks. Each returns a list of failures.


def check_errors(rows: list[dict], limit: float = SMOKE_MAX_ERROR_RATE) -> list[str]:
    """Mean error rate over the training steps and over the validation epochs. Steps
    and epochs of one run have equal sizes, so the mean of the rates is the overall rate."""
    failures = []
    train = list(values(rows, TRAIN_ERROR_KEY).values())
    evals = [float(v) for row in rows for k, v in row.items() if EVAL_ERROR.match(k)]
    for name, rates in (("training", train), ("validation", evals)):
        if not rates:
            failures.append(f"no {name} error-rate rows: {name} did not run")
        elif (rate := sum(rates) / len(rates)) > limit:
            failures.append(f"{name} error rate {rate:.1%}, above {limit:.0%}")
    return failures


def check_gradients(rows: list[dict], steps: int) -> list[str]:
    grads = values(rows, GRAD_NORM_KEY)
    failures = [f"no gradient norm logged for step {s}" for s in range(1, steps + 1) if s not in grads]
    failures += [f"gradient norm {g} at step {s}" for s, g in sorted(grads.items()) if not g > 0]
    if reason := first_step_broken(rows):
        failures.append(reason)
    return failures


def mismatch_warnings(rows: list[dict], from_step: int = 1, limit: float = MISMATCH_WARN_KL) -> dict[int, str]:
    """Steps from `from_step` on whose trainer-inference mismatch KL is above `limit`."""
    return {step: f"{MISMATCH_KEY} = {value:.4f} at step {step}, above {limit}"
            for step, value in sorted(values(rows, MISMATCH_KEY).items())
            if step >= from_step and not value <= limit}


def check_lora_module_errors(run_dir: Path) -> list[str]:
    """vLLM's unsupported-target-modules error in any of the run's logs
    (<run>/logs/attempt_<n>/, prime-rl docs/training.md "Log Files", commit 35fcb52)."""
    failures = []
    for path in sorted((run_dir / "logs").rglob("*.log")) if (run_dir / "logs").is_dir() else []:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if LORA_MODULE_ERROR.search(text):
            failures.append(f"vLLM rejected LoRA target modules (see {path.relative_to(run_dir)})")
    return sorted(set(failures))


def run_log_text(run_dir: Path) -> str:
    texts = []
    for path in sorted((run_dir / "logs").rglob("*.log")) if (run_dir / "logs").is_dir() else []:
        try:
            texts.append(path.read_text(encoding="utf-8", errors="replace"))
        except OSError:
            continue
    return "\n".join(texts)


def check_adapter_loaded(run_dir: Path, rows: list[dict], step: int) -> list[str]:
    """The last step's adapter reached inference. Before shutting down, the
    orchestrator waits until inference applied policy version `step`, and logs
    "Inference did not apply policy v<step> ..." if it did not
    (orchestrator.py:505-534, 785-786); for LoRA, applying a version means vLLM's
    /load_lora_adapter succeeded (transports/weights/filesystem.py:67-72). The
    final eval fires on that version (orchestrator.py:818-824), and its
    eval/<env>/policy_version is the lowest version its episodes came from
    (orchestrator.py:961-980). Fails on the warning, or on an eval of an older
    version."""
    failures = []
    if f"Inference did not apply policy v{step} " in run_log_text(run_dir):
        failures.append(f"inference did not apply policy v{step}: the step-{step} adapter did not load")
    versions = [float(row[key]) for row in rows if row.get("step") == step
                for key in row if re.match(r"^eval/[^/]+/policy_version$", key)]
    if versions and max(versions) < step:
        failures.append(f"the eval at step {step} ran on policy version {max(versions):g}, not {step}")
    return failures


def check_checkpoints(run_dir: Path, steps: list[int]) -> list[str]:
    """A DCP checkpoint is complete once its .metadata exists (tools/convert_dcp_to_bf16.py:64-70)."""
    return [
        f"no checkpoint for step {s}"
        for s in steps
        if not (run_dir / "checkpoints" / f"step_{s}" / "trainer" / ".metadata").exists()
    ]


def assistant_messages(record) -> list[dict]:
    """Every dict with role "assistant" in a parsed episode record, except under
    "info", where judge calls are kept (verifiers trace.py, record_judge_call)."""
    found = []
    stack = [record]
    while stack:
        item = stack.pop()
        if isinstance(item, dict):
            if item.get("role") == "assistant":
                found.append(item)
            stack.extend(value for key, value in item.items() if key != "info")
        elif isinstance(item, list):
            stack.extend(item)
    return found


def message_text(message: dict) -> str:
    content = message.get("content")
    if isinstance(content, list):
        return "".join(part.get("text", "") for part in content if isinstance(part, dict))
    return content or ""


THINKING_MAX_SHARE = 0.10
"""Share of completions with reasoning or a think tag above which thinking counts as on.
With thinking on, nearly every completion has them; with it off, the model can still
write a stray tag now and then (1 of about 160 in the first smoke run)."""


def check_thinking_off(run_dir: Path, max_share: float = THINKING_MAX_SHARE) -> list[str]:
    replies = flagged = 0
    for line in stream_lines(run_dir):
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        for message in assistant_messages(record):
            text = message_text(message)
            if not text and not message.get("reasoning_content"):
                continue
            replies += 1
            if message.get("reasoning_content") or "<think>" in text or "</think>" in text:
                flagged += 1
    if not replies:
        return ["no sampled completion found in the trace stream"]
    print(f"thinking check: {flagged} of {replies} completions have reasoning or a think tag")
    if flagged > max_share * replies:
        return [f"{flagged} of {replies} completions have reasoning or a think tag: thinking is on"]
    return []


def smoke(run_dir: Path, steps: int, max_error_rate: float = SMOKE_MAX_ERROR_RATE) -> list[str]:
    rows = read_rows(metrics_files(run_dir))
    if not rows:
        return [f"no metrics under {run_dir / 'monitors' / 'file'}"]
    for warning in mismatch_warnings(rows).values():
        print(f"WARNING: {warning}", flush=True)
    return (
        check_errors(rows, max_error_rate)
        + check_gradients(rows, steps)
        + check_lora_module_errors(run_dir)
        + check_adapter_loaded(run_dir, rows, steps)
        + check_checkpoints(run_dir, list(range(1, steps + 1)))
        + check_adapters(run_dir, list(range(1, steps + 1)))
        + check_thinking_off(run_dir)
    )


# Watching


def alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def stop(pgid: int, grace: float = 60.0) -> None:
    """SIGTERM to the whole process group, SIGKILL after `grace` seconds."""
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.time() + grace
    while time.time() < deadline and alive(pgid):
        time.sleep(2)
    if alive(pgid):
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass


# LoRA adapters ---------------------------------------------------------------

ADAPTER_FILES = ("adapter_model.safetensors", "adapter_config.json")
"""The PEFT files the trainer writes for vLLM in every LoRA broadcast
(transports/weights/filesystem.py:34-52, utils/weights.py:48-56)."""
FINISHED_MARKER = ".finished"
"""Written once a broadcast is complete (transports/weights/base.py:26)."""


def adapter_dir(run_dir: Path, step: int) -> Path:
    return run_dir / "adapters" / f"step_{step}"


def snapshot_adapters(run_dir: Path, interval: int) -> list[int]:
    """Copy the adapter of each finished broadcast at a multiple of `interval` to
    <run>/adapters/step_N/. The trainer broadcasts every step and keeps only the
    last two broadcasts (transports/weights/base.py:100-108), so a validation
    step's adapter must be copied within about two steps. Returns the steps copied."""
    copied = []
    broadcasts = run_dir / "broadcasts"
    for step_dir in sorted(broadcasts.glob("step_*")) if broadcasts.is_dir() else []:
        try:
            step = int(step_dir.name.removeprefix("step_"))
        except ValueError:
            continue
        target = adapter_dir(run_dir, step)
        # Per-process partial directory: two copiers may work on the same step.
        partial = target.with_name(f"{target.name}.partial-{os.getpid()}")
        try:
            if step % interval or target.exists() or not (step_dir / FINISHED_MARKER).exists():
                continue
            if not all((step_dir / name).exists() for name in ADAPTER_FILES):
                continue
            shutil.rmtree(partial, ignore_errors=True)
            partial.mkdir(parents=True)
            for name in ADAPTER_FILES:
                # A hard link survives the trainer deleting the broadcast; copy where
                # links are not possible.
                try:
                    os.link(step_dir / name, partial / name)
                except OSError:
                    shutil.copy2(step_dir / name, partial / name)
            partial.rename(target)
            copied.append(step)
        except Exception:  # cleaned while copying, or the other copier won: skip this step
            shutil.rmtree(partial, ignore_errors=True)
    return copied


def check_adapters(run_dir: Path, steps: list[int]) -> list[str]:
    return [f"no adapter copy for step {s}" for s in steps
            if not all((adapter_dir(run_dir, s) / name).exists() for name in ADAPTER_FILES)]


# Time projection --------------------------------------------------------------


def step_times(rows: list[dict]) -> dict[int, float]:
    """Wall time of each trainer step's row (rows with the trainer's loss)."""
    return {int(row["step"]): float(row["time"]) for row in rows
            if LOSS_KEY in row and row.get("step") is not None and row.get("time") is not None}


PROJECTION_WINDOW = 10
"""Steps per projection: one after every 10th step, from the mean of the last 10 intervals."""


def hours_left(rows: list[dict], total_steps: int, window: int = PROJECTION_WINDOW) -> tuple[int, float] | None:
    """(latest trainer step, projected hours to `total_steps`), from the mean step
    interval over the last `window` steps of this run. The interval includes the
    validation epochs that ran meanwhile. None before step `window`, or while a
    step at the start of the window is missing."""
    times = step_times(rows)
    if not times or max(times) < window:
        return None
    latest = max(times)
    # Steps start at 1, so at step 10 the window holds steps 1-10 (9 intervals).
    earliest = max(latest - window, min(times))
    if earliest not in times or latest == earliest:
        return None
    seconds_per_step = (times[latest] - times[earliest]) / (latest - earliest)
    return latest, seconds_per_step * max(total_steps - latest, 0) / 3600


def write_projection(path: Path, step: int, hours: float, total_steps: int) -> str:
    line = f"step {step} of {total_steps}: about {hours:.1f} h of training left"
    path.write_text(f"{hours:.2f}\n{line}\n", encoding="utf-8")
    return line


PROGRESS_FILE = "guard_progress.json"


def save_progress(path: Path, marker: tuple, when: float) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps({"marker": list(marker), "last_change": when}), encoding="utf-8")
    tmp.replace(path)


def load_progress(path: Path) -> tuple[tuple | None, float]:
    """(marker, time of last progress) from an earlier guard of this run, else (None, now)."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return tuple(data["marker"]), float(data["last_change"])
    except (OSError, ValueError, KeyError, TypeError):
        return None, time.time()


def write_stop(stop_file: Path, rule: str, reason: str) -> None:
    stop_file.write_text(json.dumps({"rule": rule, "reason": reason}) + "\n", encoding="utf-8")


def watch(run_dir: Path, pgid: int, max_hours: float, stop_file: Path, poll: float,
          stall_minutes: float = STALL_MINUTES, adapter_interval: int | None = None,
          total_steps: int | None = None, projection_file: Path | None = None,
          started: float | None = None) -> int:
    """Watch until the run ends. An error inside one poll (a file that changes while
    it is read, for instance) is logged with its traceback and the next poll
    goes on, so the guard itself never dies of it. `started` (a Unix time) keeps
    the wall-clock cap of a guard restarted by run_all.sh, and the last progress
    is kept in <run>/guard_progress.json, so a restart keeps the stall clock."""
    started = time.time() if started is None else started
    progress_file = run_dir / PROGRESS_FILE
    last_marker, last_change = load_progress(progress_file)
    projected_at = 0
    warned: set[int] = set()
    while alive(pgid):
        try:
            if adapter_interval:
                snapshot_adapters(run_dir, adapter_interval)
            rows = read_rows(metrics_files(run_dir))
            if total_steps and projection_file and (projection := hours_left(rows, total_steps)):
                step, hours = projection
                if step % PROJECTION_WINDOW == 0 and step > projected_at:
                    projected_at = step
                    print(f"guard: {write_projection(projection_file, step, hours, total_steps)}", flush=True)
            for step, warning in mismatch_warnings(rows, MISMATCH_WATCH_FROM_STEP).items():
                if step not in warned:
                    warned.add(step)
                    print(f"guard: WARNING: {warning}", flush=True)
            marker = progress_marker(run_dir, rows)
            if marker != last_marker:
                last_marker, last_change = marker, time.time()
                save_progress(progress_file, marker, last_change)
            fired = stop_reason(rows)
            if fired is None and (reason := stall_reason(last_change, stall_minutes)):
                fired = ("stalled", reason)
            if fired is None and (reason := over_time(started, max_hours)):
                fired = ("time_cap", reason)
            if fired:
                print(f"guard: stopping the run ({fired[0]}): {fired[1]}", flush=True)
                write_stop(stop_file, *fired)
                stop(pgid)
                return 3
        except Exception:
            print("guard: error in one poll; continuing", flush=True)
            traceback.print_exc()
            sys.stdout.flush()
        time.sleep(poll)
    # The run ended: a last look, so a rule that fired at the very end is still recorded.
    try:
        if adapter_interval:
            snapshot_adapters(run_dir, adapter_interval)
        fired = stop_reason(read_rows(metrics_files(run_dir)))
    except Exception:
        traceback.print_exc()
        fired = None
    if fired:
        print(f"guard: the run ended, but a stop rule fired ({fired[0]}): {fired[1]}", flush=True)
        write_stop(stop_file, *fired)
        return 3
    return 0


def copy_adapters_until(run_dir: Path, interval: int, pgid: int | None, every: float) -> None:
    """A second, independent adapter copier: one pass every `every` seconds while
    process group `pgid` is alive, then a last pass. snapshot_adapters is
    idempotent, so it can run next to the watch."""
    while True:
        running = pgid is not None and alive(pgid)
        try:
            copied = snapshot_adapters(run_dir, interval)
            if copied:
                print(f"adapters: copied steps {copied}", flush=True)
        except Exception:
            traceback.print_exc()
            sys.stdout.flush()
        if not running:
            return
        time.sleep(every)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    s = commands.add_parser("smoke")
    s.add_argument("--run-dir", type=Path, required=True)
    s.add_argument("--steps", type=int, default=2)
    s.add_argument("--max-error-rate", type=float, default=SMOKE_MAX_ERROR_RATE)
    t = commands.add_parser("thinking")
    t.add_argument("--run-dir", type=Path, required=True)
    w = commands.add_parser("watch")
    w.add_argument("--run-dir", type=Path, required=True)
    w.add_argument("--pgid", type=int, required=True)
    w.add_argument("--max-hours", type=float, default=float(os.environ.get("MAX_TRAIN_HOURS", "10")))
    w.add_argument("--adapter-interval", type=int, default=None)
    w.add_argument("--total-steps", type=int, default=None)
    w.add_argument("--projection-file", type=Path, default=None)
    w.add_argument("--started", type=float, default=None)
    w.add_argument("--stall-minutes", type=float, default=STALL_MINUTES)
    w.add_argument("--stop-file", type=Path, default=None)
    w.add_argument("--poll", type=float, default=30.0)
    a = commands.add_parser("adapters")
    a.add_argument("--run-dir", type=Path, required=True)
    a.add_argument("--interval", type=int, required=True)
    a.add_argument("--pgid", type=int, default=None)
    a.add_argument("--every", type=float, default=None)
    args = parser.parse_args(argv)

    if args.command == "adapters":
        if args.every:
            copy_adapters_until(args.run_dir, args.interval, args.pgid, args.every)
            return 0
        copied = snapshot_adapters(args.run_dir, args.interval)
        print(f"copied adapters for steps {copied}")
        return 0

    if args.command in ("smoke", "thinking"):
        if args.command == "smoke":
            failures = smoke(args.run_dir, args.steps, args.max_error_rate)
        else:
            failures = check_thinking_off(args.run_dir)
        for failure in failures:
            print(f"{args.command} check failed: {failure}")
        if not failures:
            print(f"{args.command} checks passed")
        return 1 if failures else 0
    stop_file = args.stop_file or args.run_dir / "GUARD_STOP"
    return watch(args.run_dir, args.pgid, args.max_hours, stop_file, args.poll, args.stall_minutes,
                 args.adapter_interval, args.total_steps, args.projection_file, args.started)


if __name__ == "__main__":
    sys.exit(main())
