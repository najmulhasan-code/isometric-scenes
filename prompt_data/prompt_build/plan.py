"""Request text, seeds and sizes.

Each request fills one template. The focus item rotates with the request
index. The examples in the motion and element rules come from a balanced
schedule over the released lists, and the insertion phrase is drawn with the
request seed, so every request text can be rebuilt.
"""

import hashlib
import math
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from string import Template

from .lexicon import read_word_list
from .strata import Stratum

SEED_RANGE = 1_000_000
MAX_INDEX = 1_000
MAIN_BLOCK = 1
DIAGNOSTIC_BLOCK = 10
EXAMPLES_PER_RULE = 3

MOTION_RULES = {
    (False, True): "Each description must name at least one thing that is moving, "
    "such as {examples}. Vary what moves.",
    (True, True): "Each base description must name at least one thing that is moving, "
    "apart from the added phrase, such as {examples}. Vary what moves.",
    (False, False): "No description may mention anything moving or changing, such as falling "
    "leaves, drifting clouds, floating objects, running water or blinking lights. "
    "Describe still scenes only.",
    (True, False): "Neither the base description nor the added phrase may mention anything "
    "moving or changing, such as falling leaves, drifting clouds, floating objects, running "
    "water or blinking lights. Describe still scenes only.",
}

ELEMENT_RULES = {
    "human": "one person, named by a single noun such as {examples}",
    "animal": "one animal, named by a single noun such as {examples}",
    "face_figure": "one figure with a face or a character form, such as {examples}",
}


def seed_block(pilot_round: int | None, diagnostic: bool = False) -> int:
    """Seed block of a run: 1 for the main run, r + 1 for pilot round r (r from 1
    to 8), and 10 for diagnostic runs."""
    if diagnostic:
        return DIAGNOSTIC_BLOCK
    if pilot_round is None:
        return MAIN_BLOCK
    if not 1 <= pilot_round < DIAGNOSTIC_BLOCK - 1:
        raise ValueError("pilot rounds are numbered from 1 to 8")
    return pilot_round + 1


def request_seed(base_seed: int, stratum_index: int, request_index: int, block: int) -> int:
    """Seed for request j of stratum i in a seed block: b + 1000000 * block + 1000 i + j."""
    if not (0 <= stratum_index < MAX_INDEX and 0 <= request_index < MAX_INDEX):
        raise ValueError("stratum and request indices must lie in [0, 1000)")
    return base_seed + SEED_RANGE * block + MAX_INDEX * stratum_index + request_index


def join_examples(examples: list[str]) -> str:
    return ", ".join(examples[:-1]) + " or " + examples[-1]


@dataclass(frozen=True)
class Templates:
    in_scope: Template
    distractor: Template
    in_scope_sha256: str
    distractor_sha256: str

    @classmethod
    def load(cls, templates_dir: Path) -> "Templates":
        texts = {
            name: (templates_dir / f"{name}.txt").read_text(encoding="utf-8")
            for name in ("in_scope", "distractor")
        }
        return cls(
            in_scope=Template(texts["in_scope"]),
            distractor=Template(texts["distractor"]),
            in_scope_sha256=hashlib.sha256(texts["in_scope"].encode("utf-8")).hexdigest(),
            distractor_sha256=hashlib.sha256(texts["distractor"].encode("utf-8")).hexdigest(),
        )


@dataclass(frozen=True)
class ExampleLists:
    motion: tuple[str, ...]
    elements: dict[str, tuple[str, ...]]
    insertion_frames: tuple[str, ...]

    @classmethod
    def load(cls, lists_dir: Path) -> "ExampleLists":
        return cls(
            motion=tuple(read_word_list(lists_dir / "motion_examples.txt")),
            elements={
                kind: tuple(read_word_list(lists_dir / f"{kind}_examples.txt"))
                for kind in ELEMENT_RULES
            },
            insertion_frames=tuple(read_word_list(lists_dir / "insertion_frames.txt")),
        )


def lists_used(stratum: Stratum) -> list[str]:
    """Names of the example lists a stratum's requests draw from."""
    names = ["motion"] if stratum.motion else []
    if stratum.distractor:
        names.append(stratum.kind)
    return names


@dataclass(frozen=True)
class ExampleSchedule:
    """The examples each request shows.

    Each example list is put in one seeded order and read three entries at a
    time, so that over a run every entry is shown about equally often. The
    generator tends to copy the examples it is shown, so this keeps the
    elements and motion cues in the pool close to uniform over each list.

    A stratum's requests read consecutive positions, starting where the
    previous stratum using the same list stopped. Positions are reserved for
    the number of requests each stratum is planned to send. Any further
    request of a stratum (after a failed or short response, or a refill)
    reads on into the next stratum's block and repeats its first examples;
    only the last stratum on a list wraps around to the start of the order.
    """

    orders: dict[str, tuple[str, ...]]
    starts: dict[tuple[str, int], int]

    @classmethod
    def build(
        cls, strata: list[Stratum], examples: "ExampleLists", seed: int, items_per_request: int
    ) -> "ExampleSchedule":
        lists = {"motion": examples.motion, **examples.elements}
        orders = {}
        for offset, (name, entries) in enumerate(lists.items()):
            order = list(entries)
            random.Random(seed + offset).shuffle(order)
            orders[name] = tuple(order)
        starts: dict[tuple[str, int], int] = {}
        next_position: Counter[str] = Counter()
        for stratum in strata:
            planned = math.ceil(stratum.target / items_per_request)
            for name in lists_used(stratum):
                starts[name, stratum.index] = next_position[name]
                next_position[name] += EXAMPLES_PER_RULE * planned
        return cls(orders, starts)

    def draw(self, name: str, stratum_index: int, request_index: int) -> list[str]:
        order = self.orders[name]
        start = self.starts[name, stratum_index] + EXAMPLES_PER_RULE * request_index
        return [order[(start + k) % len(order)] for k in range(EXAMPLES_PER_RULE)]


def exposure(
    schedule: ExampleSchedule, strata: list[Stratum], items_per_request: int
) -> dict[str, Counter]:
    """How often each example is shown over the planned requests of a run."""
    shown: dict[str, Counter] = {name: Counter() for name in schedule.orders}
    for stratum in strata:
        for request_index in range(math.ceil(stratum.target / items_per_request)):
            for name in lists_used(stratum):
                shown[name].update(schedule.draw(name, stratum.index, request_index))
    return shown


@dataclass(frozen=True)
class RequestSpec:
    stratum_index: int
    request_index: int
    seed: int
    size: int
    template_name: str
    template_sha256: str
    text: str


def build_request(
    stratum: Stratum,
    request_index: int,
    size: int,
    seed: int,
    templates: Templates,
    examples: ExampleLists,
    schedule: ExampleSchedule,
) -> RequestSpec:
    """Fill the template for one request."""
    rng = random.Random(seed)
    focus = stratum.category.focus
    values = {
        "n": str(size),
        "category_name": stratum.category.name,
        "category_definition": stratum.category.definition,
        "focus": focus[(stratum.index + request_index) % len(focus)],
    }

    motion_rule = MOTION_RULES[stratum.distractor, stratum.motion]
    if stratum.motion:
        drawn = schedule.draw("motion", stratum.index, request_index)
        motion_rule = motion_rule.format(examples=join_examples(drawn))
    values["motion_rule"] = motion_rule

    if stratum.distractor:
        drawn = schedule.draw(stratum.kind, stratum.index, request_index)
        values["element_rule"] = ELEMENT_RULES[stratum.kind].format(examples=join_examples(drawn))
        frame = rng.choice(examples.insertion_frames)
        values["insertion_example"] = frame.format(element=drawn[-1])
        template, name, digest = templates.distractor, "distractor", templates.distractor_sha256
    else:
        template, name, digest = templates.in_scope, "in_scope", templates.in_scope_sha256

    return RequestSpec(
        stratum_index=stratum.index,
        request_index=request_index,
        seed=seed,
        size=size,
        template_name=name,
        template_sha256=digest,
        text=template.substitute(values),
    )
