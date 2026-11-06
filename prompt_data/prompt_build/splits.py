"""Split assignment.

Every stratum is shuffled with its own seed. In-scope strata fill their
per-category quotas, and distractor strata fill the slots of the slot table,
in split order: test_unseen and test, then val, then train. What is left
over stays in each stratum as reserves, in shuffled order. Candidates with
a motion flag have already been removed from the pool, so every split is
drawn the same way.
"""

import random
from collections import defaultdict
from dataclasses import dataclass, field

from .config import UNSEEN_SPLIT, BuildConfig
from .pool import Candidate
from .strata import Slot, Stratum

FILL_ORDER = (UNSEEN_SPLIT, "test", "val", "train")


@dataclass
class Assignment:
    splits: dict[str, list[Candidate]] = field(default_factory=lambda: defaultdict(list))
    reserves: dict[int, list[Candidate]] = field(default_factory=dict)
    shortfalls: dict[str, int] = field(default_factory=dict)

    @property
    def complete(self) -> bool:
        return not self.shortfalls


def assign_splits(
    candidates: list[Candidate], strata: list[Stratum], slots: list[Slot], config: BuildConfig
) -> Assignment:
    by_stratum: dict[int, list[Candidate]] = defaultdict(list)
    for candidate in sorted(candidates, key=lambda c: c.candidate_id):
        if candidate.kept:
            by_stratum[candidate.stratum.index].append(candidate)
    queues = {}
    for stratum in strata:
        queue = list(by_stratum[stratum.index])
        random.Random(config.base_seed + stratum.index).shuffle(queue)
        queues[stratum.index] = queue
    in_scope = {(s.category.slug, s.motion): s for s in strata if not s.distractor}
    distractors = {(s.category.slug, s.kind, s.motion): s for s in strata if s.distractor}

    assignment = Assignment()
    short: dict[str, int] = defaultdict(int)

    def place(stratum: Stratum, split: str) -> None:
        queue = queues[stratum.index]
        if not queue:
            short[stratum.key] += 1
            return
        candidate = queue.pop(0)
        candidate.split = split
        assignment.splits[split].append(candidate)

    for split in FILL_ORDER:
        quota = config.splits[split]
        for category in config.split_categories(split):
            for motion in (True, False):
                for _ in range(quota.in_scope[motion]):
                    place(in_scope[category.slug, motion], split)
        for slot in slots:
            if slot.split == split:
                place(distractors[slot.category.slug, slot.kind, slot.motion], split)

    assignment.reserves = {index: list(queue) for index, queue in queues.items()}
    assignment.shortfalls = dict(short)
    return assignment
