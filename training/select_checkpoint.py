"""Choose the checkpoint with the highest validation reward. Standard library only.

  select_checkpoint.py --run-dir DIR [--env val] [--out selection.json]

Reads the validation reward of every evaluated step from
<run>/monitors/file/metrics.jsonl (keys eval/<env>/<subset>/<agent>/reward/mean,
written by prime-rl's orchestrator; see guard.py), keeps the steps that have
both a complete checkpoint, <run>/checkpoints/step_N/trainer/.metadata, and a
copy of the LoRA adapter, <run>/adapters/step_N/ (guard.snapshot_adapters), and
picks the best. Step 0 is the base model, which has neither; it is reported as
the baseline. A tie goes to the earlier step.

The eval at step N runs on the policy after N trainer steps, the same weights
the trainer saves as step_N (orchestrator/eval_source.py, trigger(step)).
Training is asynchronous, so weights can change while an eval epoch runs. For
each step, the policy versions its episodes were generated under are read from
the trace stream (run.work.policy.{start,end}, verifiers trace.py:177-207) and
recorded as `policy_spans`; they are not used to choose. If reading them fails,
the error is recorded as `policy_spans_error` and the choice is unaffected.
With SELECT_SKIP_SPANS=1 in the environment (used on a wind-down, where time is
short) the spans are not read, and `policy_spans_skipped` is true.

Prints the chosen step's adapter directory on stdout; writes all scores to --out.
Exit 0 on success, 1 if no evaluated step has a checkpoint.
"""

import argparse
import json
import os
import sys
from pathlib import Path

from guard import ADAPTER_FILES, adapter_dir, eval_rewards, metrics_files, read_rows, stream_lines


def checkpoint_dir(run_dir: Path, step: int) -> Path:
    return run_dir / "checkpoints" / f"step_{step}"


def has_checkpoint(run_dir: Path, step: int) -> bool:
    return (checkpoint_dir(run_dir, step) / "trainer" / ".metadata").exists()


def has_adapter(run_dir: Path, step: int) -> bool:
    return all((adapter_dir(run_dir, step) / name).exists() for name in ADAPTER_FILES)


def choose(scores: dict[int, float], available: set[int]) -> int | None:
    """The step with the highest score among `available`, earliest on a tie."""
    candidates = [step for step in scores if step > 0 and step in available]
    if not candidates:
        return None
    return min(candidates, key=lambda step: (-scores[step], step))


def policy_spans(run_dir: Path, env: str) -> dict[int, dict]:
    """Per eval step of `env`: episodes, lowest span start, highest span end."""
    spans: dict[int, dict] = {}
    for line in stream_lines(run_dir):
        if b'"eval"' not in line:  # skip parsing training episodes
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        work = (record.get("run") or {}).get("work") or {}
        name = (record.get("env") or {}).get("name") or (record.get("env") or {}).get("id")
        if work.get("type") != "eval" or name != env or work.get("step") is None:
            continue
        span = work.get("policy") or {}
        entry = spans.setdefault(int(work["step"]), {"episodes": 0, "min_start": None, "max_end": None})
        entry["episodes"] += 1
        if span.get("start") is not None:
            entry["min_start"] = span["start"] if entry["min_start"] is None else min(entry["min_start"], span["start"])
        if span.get("end") is not None:
            entry["max_end"] = span["end"] if entry["max_end"] is None else max(entry["max_end"], span["end"])
    return spans


def chosen_span(spans: dict[int, dict], step: int | None) -> dict | None:
    """The chosen step's span with its drift: the highest version minus the step."""
    span = spans.get(step) if step is not None else None
    if not span:
        return None
    drift = span["max_end"] - step if span.get("max_end") is not None else None
    return {**span, "step": step, "drift": drift}


def select(run_dir: Path, env: str = "val") -> dict:
    rows = read_rows(metrics_files(run_dir))
    by_step = eval_rewards(rows, env).get(env, {})
    scores = {step: stats["mean"] for step, stats in by_step.items() if "mean" in stats}
    available = {step for step in scores if has_checkpoint(run_dir, step) and has_adapter(run_dir, step)}
    chosen = choose(scores, available)
    baseline = scores.get(0)
    skip_spans = os.environ.get("SELECT_SKIP_SPANS") == "1"
    spans, spans_error = {}, None
    if not skip_spans:
        try:
            spans = policy_spans(run_dir, env)
        except Exception as error:  # recorded only; never blocks the choice
            spans_error = f"{type(error).__name__}: {error}"
    return {
        "env": env,
        "metric": f"eval/{env}/effective/<agent>/reward/mean (the all subset where effective is missing)",
        "scores": {str(step): scores[step] for step in sorted(scores)},
        "with_checkpoint": sorted(available),
        "baseline": baseline,
        "chosen_step": chosen,
        "chosen_score": scores.get(chosen) if chosen is not None else None,
        "checkpoint": str(checkpoint_dir(run_dir, chosen)) if chosen is not None else None,
        "adapter": str(adapter_dir(run_dir, chosen)) if chosen is not None else None,
        "beats_baseline": None if chosen is None or baseline is None else scores[chosen] > baseline,
        "policy_spans": {str(step): spans[step] for step in sorted(spans)},
        "policy_spans_error": spans_error,
        "policy_spans_skipped": skip_spans,
        # The policy versions the chosen step's validation episodes came from:
        # LoRA adapters are hot-swapped under live traffic (transports/weights/
        # filesystem.py:60-74), and the RL eval config has no setting that holds
        # an epoch to one version, so an epoch at step N can include later versions.
        "chosen_policy_span": chosen_span(spans, chosen),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--env", default="val")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    result = select(args.run_dir, args.env)
    if args.out:
        args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if result["chosen_step"] is None:
        print(f"no evaluated step on {args.env} has a checkpoint", file=sys.stderr)
        return 1
    print(f"chose step {result['chosen_step']} ({result['chosen_score']:.4f}; "
          f"baseline {result['baseline']})", file=sys.stderr)
    print(result["adapter"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
