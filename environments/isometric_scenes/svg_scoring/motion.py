"""Frame times, the not-blank check and the motion check (README, "Scoring").

The renderer measures the frames; this module turns its measures into
decisions. All values are fixed before any policy output is rendered.
"""

import hashlib
import random
from dataclasses import dataclass

FRAME_COUNT = 4
LAST_TIME = 3.0
CHANGE_TOLERANCE = 8
MIN_CHANGED_SHARE = 0.002
MAX_CHANGED_SHARE = 0.5
MAX_BRIGHTNESS_CHANGE = 10.0
MAX_DOMINANT_SHARE = 0.98


def frame_times(answer: str) -> list[float]:
    """t = 0 and three times in (0, LAST_TIME] seconds, seeded from the answer's text."""
    seed = int(hashlib.sha256(answer.encode("utf-8")).hexdigest()[:16], 16)
    rng = random.Random(seed)
    later = sorted(round(LAST_TIME * (1.0 - rng.random()), 3) for _ in range(FRAME_COUNT - 1))
    return [0.0, *later]


@dataclass(frozen=True)
class FrameChecks:
    not_blank: bool
    moves: bool
    largest_change: float
    largest_brightness_change: float


def judge_frames(measures: dict) -> FrameChecks:
    """Decide the not-blank and motion checks from the renderer's measures."""
    largest_change = max((share for _, _, share in measures["changed"]), default=0.0)
    brightness = measures["brightness"]
    largest_brightness = max(brightness) - min(brightness) if brightness else 0.0
    later_change = max((share for a, _, share in measures["changed"] if a > 0), default=0.0)
    moves = (MIN_CHANGED_SHARE <= largest_change <= MAX_CHANGED_SHARE
             and later_change >= MIN_CHANGED_SHARE
             and largest_brightness < MAX_BRIGHTNESS_CHANGE)
    dominant = max(measures["colour_shares"], default=1.0)
    return FrameChecks(dominant <= MAX_DOMINANT_SHARE, moves, largest_change, largest_brightness)
