"""The vision judge (README, "Scoring").

The judge sees the scene description and the rendered frames, never the SVG
source, and answers six yes/no items in a fixed JSON schema.
"""

from pydantic import BaseModel

import verifiers.v1 as vf

from isometric_scenes.prompts import JUDGE_PROMPT


JUDGE_TIMEOUT = 30
"""Seconds per judge call; the OpenAI client retries a failed call twice."""


class SceneVerdict(BaseModel):
    """Answers only: with a free-text field, constrained decoding made the judge emit blank tokens until its limit."""

    scene: bool
    objects: bool
    isometric: bool
    composition: bool
    figure_present: bool
    motion: bool


class SceneJudge(vf.Judge[SceneVerdict]):
    prompt = JUDGE_PROMPT
    schema = SceneVerdict

    def build_messages(self, *, description: str, frames: list[str], times: list[float]) -> vf.Messages:
        text = self.prompt.format(frame_count=len(frames), description=description)
        content: list = [vf.TextContentPart(text=text)]
        for number, (frame, t) in enumerate(zip(frames, times), start=1):
            content.append(vf.TextContentPart(text=f"Frame {number}, at {t:.3f} s:"))
            content.append(vf.ImageUrlContentPart(image_url=vf.ImageUrlSource(url=f"data:image/png;base64,{frame}")))
        return [vf.UserMessage(content=content)]

    async def verdict(self, trace: vf.Trace, *, description: str, frames: list[str],
                      times: list[float]) -> SceneVerdict:
        messages = self.build_messages(description=description, frames=frames, times=times)
        response = await self.complete(messages, trace=trace, schema=SceneVerdict, timeout=JUDGE_TIMEOUT)
        return response.parsed
