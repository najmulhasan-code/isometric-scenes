"""Strata and the distractor slot table.

A stratum is one group of candidates generated with the same request
settings. In-scope strata are keyed by category and motion label, and
distractor strata by category, element kind and motion label. Both are
pooled across splits: splits are assigned only after filtering, so every
split draws from the same requests. A stratum's quota is the number of
prompts it gives to the final dataset, and its target is a multiple of that,
counted in parsed items.
"""

import random
from collections import Counter
from dataclasses import dataclass

from .config import KIND_ORDER, SPLITS, BuildConfig, Category

TARGET_FACTOR = 2


@dataclass(frozen=True)
class Stratum:
    index: int
    category: Category
    motion: bool
    distractor: bool
    kind: str | None
    quota: int

    @property
    def target(self) -> int:
        return TARGET_FACTOR * self.quota

    @property
    def key(self) -> str:
        """A readable identifier, also written to the run log."""
        motion = "motion" if self.motion else "still"
        if self.distractor:
            return f"distractor/{self.category.slug}/{self.kind}/{motion}"
        return f"in_scope/{self.category.slug}/{motion}"


@dataclass(frozen=True)
class Slot:
    split: str
    category: Category
    kind: str
    motion: bool


def spread_labels(
    labels: tuple, totals: dict, categories: tuple[Category, ...], rng: random.Random
) -> dict[str, list]:
    """Give each category an equal share of slots with labels as even as possible.

    Each category first gets the whole-number part of its share of every
    label. The remaining labels are shuffled and dealt so that every
    category ends with the same number of slots and the totals hold exactly.
    """
    total = sum(totals.values())
    if total % len(categories):
        raise ValueError(f"{total} slots cannot be spread evenly over {len(categories)} categories")
    per_category = total // len(categories)
    base = {label: per_category * totals[label] // total for label in labels}
    leftovers = [
        label for label in labels for _ in range(totals[label] - base[label] * len(categories))
    ]
    rng.shuffle(leftovers)
    spare = per_category - sum(base.values())
    spread = {}
    for position, category in enumerate(categories):
        own = [label for label in labels for _ in range(base[label])]
        spread[category.slug] = own + leftovers[position * spare : (position + 1) * spare]
    return spread


def distractor_slots(config: BuildConfig) -> list[Slot]:
    """Assign every distractor slot a split, category, kind and motion label.

    Within each split, kinds and motion labels are each spread evenly over
    the categories, and then paired at random within each category.
    """
    slots = []
    for split_index, split in enumerate(SPLITS):
        quota = config.splits[split]
        categories = config.split_categories(split)
        kind_rng = random.Random(
            config.base_seed + config.seed_offsets["element_type_assignment"] + split_index
        )
        motion_rng = random.Random(
            config.base_seed + config.seed_offsets["slot_table"] + split_index
        )
        kinds = spread_labels(KIND_ORDER, quota.distractor_kinds, categories, kind_rng)
        motions = spread_labels((True, False), quota.distractor_motion, categories, motion_rng)
        for category in categories:
            paired_motions = list(motions[category.slug])
            motion_rng.shuffle(paired_motions)
            slots.extend(
                Slot(split, category, kind, motion)
                for kind, motion in zip(kinds[category.slug], paired_motions, strict=True)
            )
    return slots


def build_strata(config: BuildConfig) -> list[Stratum]:
    """All strata in their fixed order: in-scope first, then distractors.

    Both follow category order, then kind order for distractors, with the
    motion label before the still one.
    """
    in_scope_quota: Counter[tuple[str, bool]] = Counter()
    for split in SPLITS:
        for category in config.split_categories(split):
            for motion, count in config.splits[split].in_scope.items():
                in_scope_quota[category.slug, motion] += count
    distractor_quota = Counter(
        (slot.category.slug, slot.kind, slot.motion) for slot in distractor_slots(config)
    )

    specs = []
    for category in config.categories:
        for motion in (True, False):
            if in_scope_quota[category.slug, motion]:
                specs.append((category, motion, False, None, in_scope_quota[category.slug, motion]))
    for category in config.categories:
        for kind in KIND_ORDER:
            for motion in (True, False):
                quota = distractor_quota[category.slug, kind, motion]
                if quota:
                    specs.append((category, motion, True, kind, quota))

    return [
        Stratum(index, category, motion, distractor, kind, quota)
        for index, (category, motion, distractor, kind, quota) in enumerate(specs)
    ]
