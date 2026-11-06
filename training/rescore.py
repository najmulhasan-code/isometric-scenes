"""Score saved rollouts again with another judge, so every judge sees the same answers.

  rescore.py --run-dir DIR --judge MODEL --out FILE.jsonl [--max-tokens N]
             [--model-tag TAG] [--seed S] [--kind eval] [--limit N]
             [--concurrency N] [--max-error-rate R] [--thinking-on]

Reads the episodes a prime-rl run saved in its trace stream (guard.stream_lines)
with verifiers' record loader `vf.WireEpisode` (verifiers episode.py:172). Each
rollout with a completion is scored again with `Task.rescore` (verifiers
task.py:227-235) on a copy of its trace, by an isometric-scenes task whose judge
is MODEL. The policy is not called again: the stored SVG goes through the same
gates, render (frame times are seeded from the SVG text) and frame checks; only
the judge differs.

A rollout whose online scoring failed still has its completion. In verifiers,
scoring runs inside `boundary(TaskError, "task <Task> scoring")` (task.py:281,
errors.py:82-96), so the render sandbox and judge failures of this taskset are
recorded as TaskError with that message, and a scoring deadline as stop
condition "scoring_timeout" (trace.py, record_timeout). Only then is the copy's
error cleared, because `Task.rescore` refuses errored traces (task.py:229-230);
the original errors are kept in the row as `run_errors`. Any other error
(ProviderError, HarnessError, InterceptionError, EnvError, a setup TaskError,
...) means generation did not finish: the row gets status "no_completion" and is
not scored. A failed rescoring is retried 3 times.

Every row carries model tag, seed, split and prompt id (row_id, idx), so answers
can be paired across models and judges in analysis. Row status is one of
"scored", "error", "no_completion". The judge's replies are kept under
`judge_responses`; the finish reason is not part of a successful judge response
(verifiers judge.py:80-84), and appears in the error text when the reply does
not parse (judge.py:219-221).

--thinking-on records in the summary that guard.py found a completion with
thinking on in this run; the rows are scored as usual.

Writes FILE.jsonl and FILE.summary.json. Runs in prime-rl's environment, where
verifiers and isometric_scenes are installed; needs PRIME_API_KEY and
RENDER_IMAGE. Exit 0 if at most R (default 5%) of the rows are not "scored",
2 if more, 1 if there was no rollout at all.
"""

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path

from guard import stream_lines

RETRIES = 3
RETRY_DELAY = 10.0
"""Seconds before the first retry; doubled for each further retry."""


SCORING_ERROR = re.compile(r"^task \S+ scoring(: | timed out$)")
"""The messages `boundary(TaskError, f"task {name} scoring")` gives (errors.py:93-96)."""


def scoring_failure_only(errors: list[tuple[str, str]], stop_condition: str | None) -> bool:
    """True when the trace failed only while being scored: every error is the
    TaskError of the scoring boundary, or there is no error and the scoring
    deadline expired. `errors` holds (type, message) pairs."""
    if not errors:
        return stop_condition == "scoring_timeout"
    return all(kind == "TaskError" and SCORING_ERROR.match(message) for kind, message in errors)


def episode_kind(episode) -> str:
    """"train" or "eval", as prime-rl's file monitor stamps it on each trace."""
    for trace in episode.traces:
        kind = (trace.info or {}).get("kind")
        if kind in ("train", "eval"):
            return kind
    return "eval"


def summarize(rows: list[dict]) -> dict:
    """Counts by status and the mean reward of the scored rows, per env."""
    by_env: dict[str, dict] = {}
    for row in rows:
        env = by_env.setdefault(row["env"], {"rollouts": 0, "scored": 0, "error": 0, "no_completion": 0,
                                             "rewards": []})
        env["rollouts"] += 1
        env[row["status"]] += 1
        if row["status"] == "scored":
            env["rewards"].append(row["reward"])
    for env in by_env.values():
        rewards = env.pop("rewards")
        env["mean_reward"] = sum(rewards) / len(rewards) if rewards else None
    return by_env


def failure_rate(rows: list[dict]) -> float:
    return sum(row["status"] != "scored" for row in rows) / len(rows) if rows else 1.0


async def rescore_all(run_dir: Path, judge: str, max_tokens: int, kind: str | None, limit: int | None,
                      concurrency: int, tag: str | None, seed: int | None) -> list[dict]:
    import verifiers.v1 as vf
    from isometric_scenes.taskset import SceneData, SceneTask, SceneTaskConfig

    config = SceneTaskConfig(judge=vf.JudgeConfig(model=judge, sampling={"temperature": 0.0, "max_tokens": max_tokens}),
                             max_sandboxes=concurrency)
    work = []
    for line in stream_lines(run_dir):
        episode = vf.WireEpisode.model_validate_json(line)
        if kind and episode_kind(episode) != kind:
            continue
        work.extend((episode, trace) for trace in episode.traces)
    if limit is not None:
        work = work[:limit]

    slots = asyncio.Semaphore(concurrency)

    async def one(episode, trace) -> dict:
        data = SceneData.model_validate(trace.task.data.model_dump(mode="json"))
        row = {
            "model": tag, "seed": seed, "env": episode.env.name or episode.env.id,
            "split": data.split, "row_id": data.row_id, "idx": data.idx, "category": data.category,
            "distractor": data.distractor, "motion_in_prompt": data.motion_in_prompt,
            "episode_id": episode.id, "trace_id": trace.id, "judge": judge,
            "run_judge_reward": None if trace.has_error else trace.reward,
            "run_errors": [error.type for error in trace.errors],
            "status": None, "reward": None, "scoring": None, "judge_responses": None, "error": None,
        }
        failed = not episode.ok or trace.has_error
        errors = [(error.type, error.message) for error in trace.errors]
        if not trace.last_reply or (failed and not scoring_failure_only(errors, trace.stop_condition)):
            row["status"] = "no_completion"
            return row
        for attempt in range(RETRIES + 1):
            copy = trace.model_copy(deep=True)
            if failed:  # a scoring failure only: the completion is whole
                copy.ok, copy.errors, copy.is_timeout = True, [], False
            async with slots:
                try:
                    await SceneTask(data, config).rescore(copy)
                except Exception as error:  # sandbox or judge failure
                    row["error"] = f"{type(error).__name__}: {error}"
                else:
                    row.update(status="scored", reward=copy.reward, error=None,
                               scoring=copy.info.get("scoring"),
                               judge_responses=[call.get("response") for call in copy.info.get("judge_calls", [])])
                    return row
            if attempt < RETRIES:
                await asyncio.sleep(RETRY_DELAY * 2**attempt)
        row["status"] = "error"
        return row

    return list(await asyncio.gather(*(one(episode, trace) for episode, trace in work)))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--judge", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--model-tag", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--kind", choices=["train", "eval"], default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--max-error-rate", type=float, default=0.05)
    parser.add_argument("--thinking-on", action="store_true")
    args = parser.parse_args(argv)

    rows = asyncio.run(rescore_all(args.run_dir, args.judge, args.max_tokens, args.kind, args.limit,
                                   args.concurrency, args.model_tag, args.seed))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, default=str) + "\n")
    summary = {"judge": args.judge, "max_tokens": args.max_tokens, "model": args.model_tag, "seed": args.seed,
               "thinking_on": args.thinking_on, "rollouts": len(rows), "not_scored_rate": failure_rate(rows),
               "by_env": summarize(rows)}
    args.out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary))
    if not rows:
        print("rescore: no rollout found", file=sys.stderr)
        return 1
    return 2 if failure_rate(rows) > args.max_error_rate else 0


if __name__ == "__main__":
    sys.exit(main())
