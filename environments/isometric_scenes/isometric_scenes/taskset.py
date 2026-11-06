"""Isometric scenes: animated isometric SVG scenes of places and objects.

Each task is one prompt from the released dataset. The policy answers with
one SVG document. The reward (README, "Scoring"):

1. code gates on the answer (svg_scoring.gates);
2. rendering in a fresh sandbox with no network (isometric_scenes.render);
3. the not-blank and motion checks on the frames (svg_scoring.motion);
4. a vision judge on the frames (isometric_scenes.judge).

A failed gate, a failed render or a blank frame scores 0. If the judge sees
a person, animal, face or figure, the reward is 0. Otherwise it is the mean
of the judge's scene, objects, isometric and composition items, the code
motion check, and, for prompts that name a moving element, the judge's
motion item. Failures of the sandbox or the judge raise: verifiers records
the rollout with an error, and training and evaluation leave such rollouts
out. Every stage is recorded in `trace.info["scoring"]`.
"""

import json
import os
from pathlib import Path
from typing import Literal

import verifiers.v1 as vf

from isometric_scenes.judge import SceneJudge
from isometric_scenes.prompts import SYSTEM_PROMPT
from isometric_scenes.render import render
from svg_scoring.gates import check
from svg_scoring.motion import CHANGE_TOLERANCE, frame_times, judge_frames

DATA_DIR = Path(__file__).resolve().parent / "data"
SCORING_TIMEOUT = 300
"""Seconds for the whole scoring stage of one rollout: sandbox start, render and judge."""
Split = Literal["train", "val", "test", "test_unseen"]


class SceneData(vf.TaskData):
    row_id: str
    split: str
    category: str
    base_prompt: str
    distractor: bool
    excluded_element: str | None = None
    motion_in_prompt: bool


class SceneTaskConfig(vf.TaskConfig):
    judge: vf.JudgeConfig = vf.JudgeConfig(
        model="google/gemma-3-27b-it",
        sampling={"temperature": 0.0, "max_tokens": 256},
    )
    """The training judge. Evaluation runs override it with `--taskset.task.judge.model`."""
    render_image: str = ""
    """Sandbox image for rendering. Empty means the RENDER_IMAGE environment variable."""
    max_sandboxes: int = 64
    """Render sandboxes in flight at once, per process."""
    render_time_limit: int = 60
    """Seconds the browser may take for one SVG; longer counts as a failed render."""


class SceneTask(vf.Task[SceneData, vf.State, SceneTaskConfig]):
    @vf.reward(weight=1.0)
    async def scene(self, trace: vf.Trace) -> float:
        info = trace.info.setdefault("scoring", {})
        gate = check(trace.last_reply, self.data.prompt_text)
        info["gate_failures"] = gate.failures
        if not gate.passed:
            return 0.0

        image = self.config.render_image or os.environ.get("RENDER_IMAGE", "")
        if not image:
            raise RuntimeError("set taskset.task.render_image or RENDER_IMAGE to the render image")
        times = frame_times(gate.svg)
        measures = await render(gate.svg, times, CHANGE_TOLERANCE, image,
                                max_sandboxes=self.config.max_sandboxes, time_limit=self.config.render_time_limit)
        if "error" in measures:
            info["render_error"] = measures["error"]
            return 0.0
        frames = judge_frames(measures)
        info.update(times=times, browser=measures.get("browser"), not_blank=frames.not_blank,
                    moves=frames.moves, largest_change=round(frames.largest_change, 4),
                    largest_brightness_change=round(frames.largest_brightness_change, 2))
        if not frames.not_blank:
            return 0.0

        verdict = await SceneJudge(self.config.judge).verdict(
            trace, description=self.data.base_prompt, frames=measures["frames"], times=times)
        info["judge"] = verdict.model_dump()
        if verdict.figure_present:
            return 0.0
        items = [verdict.scene, verdict.objects, verdict.isometric, verdict.composition, frames.moves]
        if self.data.motion_in_prompt:
            items.append(verdict.motion)
        return sum(items) / len(items)


class SceneTasksetConfig(vf.TasksetConfig):
    split: Split = "train"
    """Which released split to load."""
    num_tasks: int | None = None
    """Load only the first N rows of the split, for quick checks."""
    task: SceneTaskConfig = SceneTaskConfig()


class SceneTaskset(vf.Taskset[SceneTask, SceneTasksetConfig]):
    def load(self) -> list[SceneTask]:
        lines = (DATA_DIR / f"{self.config.split}.jsonl").read_text(encoding="utf-8").splitlines()
        rows = [json.loads(line) for line in lines if line.strip()]
        if self.config.num_tasks is not None:
            rows = rows[: self.config.num_tasks]
        return [
            SceneTask(
                SceneData(
                    idx=row["idx"], row_id=row["id"], split=row["split"], category=row["category"],
                    prompt=row["prompt"], system_prompt=SYSTEM_PROMPT, base_prompt=row["base_prompt"],
                    distractor=row["distractor"], excluded_element=row["excluded_element"],
                    motion_in_prompt=row["motion_in_prompt"], timeout=vf.TaskTimeout(scoring=SCORING_TIMEOUT),
                ),
                self.config.task,
            )
            for row in rows
        ]
