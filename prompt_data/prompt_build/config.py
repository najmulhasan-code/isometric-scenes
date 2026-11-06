"""Build configuration, read from prompt_data/config.toml."""

import tomllib
from dataclasses import dataclass
from pathlib import Path

OFFSET_MIN = 10_000
OFFSET_MAX = 19_999

SPLITS = ("train", "val", "test", "test_unseen")
UNSEEN_SPLIT = "test_unseen"
KIND_ORDER = ("human", "animal", "face_figure")


@dataclass(frozen=True)
class Category:
    slug: str
    name: str
    definition: str
    focus: tuple[str, ...]


@dataclass(frozen=True)
class SplitQuota:
    """Quotas for one split.

    `in_scope` is per category, keyed by motion label (True means with a cue).
    `distractor_kinds` and `distractor_motion` are totals for the split.
    """

    in_scope: dict[bool, int]
    distractor_kinds: dict[str, int]
    distractor_motion: dict[bool, int]

    @property
    def distractor_total(self) -> int:
        return sum(self.distractor_kinds.values())


@dataclass(frozen=True)
class GeneratorConfig:
    model: str
    base_url: str
    temperature: float
    top_p: float
    reasoning_effort: str
    max_tokens: int
    items_per_request: int
    use_json_schema: bool
    timeout_seconds: float
    max_workers: int


@dataclass(frozen=True)
class PrescreenConfig:
    """The model pre-screen of each review round."""

    model: str
    base_url: str
    temperature: float
    max_tokens: int
    timeout_seconds: float
    max_workers: int


@dataclass(frozen=True)
class BuildSettings:
    """Cleaning and splitting settings."""

    jaccard_threshold: float
    cosine_threshold: float
    word_cap_share: float
    element_cap_share: float
    encoder_model: str
    encoder_revision: str
    audit_size: int


@dataclass(frozen=True)
class BuildConfig:
    spec_version: str
    base_seed: int
    held_out_category: str
    seed_offsets: dict[str, int]
    categories: tuple[Category, ...]
    splits: dict[str, SplitQuota]
    generator: GeneratorConfig | None
    build: BuildSettings | None
    prescreen: PrescreenConfig | None = None

    def split_categories(self, split: str) -> tuple[Category, ...]:
        """Categories used by a split: the held-out one for test_unseen, the rest otherwise."""
        if split == UNSEEN_SPLIT:
            return tuple(c for c in self.categories if c.slug == self.held_out_category)
        return tuple(c for c in self.categories if c.slug != self.held_out_category)


def _motion_counts(raw: dict) -> dict[bool, int]:
    return {True: raw["with_motion"], False: raw["without_motion"]}


def _load_split(name: str, raw: dict, category_count: int) -> SplitQuota:
    kinds = {kind: raw["distractor_kinds"][kind] for kind in KIND_ORDER}
    quota = SplitQuota(
        in_scope=_motion_counts(raw["in_scope"]),
        distractor_kinds=kinds,
        distractor_motion=_motion_counts(raw["distractor_motion"]),
    )
    if sum(quota.distractor_motion.values()) != quota.distractor_total:
        raise ValueError(f"split {name!r}: distractor motion counts do not sum to the kind counts")
    if quota.distractor_total % category_count:
        raise ValueError(f"split {name!r}: distractors cannot be spread evenly over categories")
    return quota


def load_config(path: Path) -> BuildConfig:
    with path.open("rb") as handle:
        raw = tomllib.load(handle)

    categories = tuple(
        Category(
            slug=entry["slug"],
            name=entry["name"],
            definition=entry["definition"],
            focus=tuple(entry["focus"]),
        )
        for entry in raw["categories"]
    )
    slugs = [category.slug for category in categories]
    if len(set(slugs)) != len(slugs):
        raise ValueError("category slugs must be unique")
    held_out = raw["held_out_category"]
    if held_out not in slugs:
        raise ValueError(f"unknown held-out category {held_out!r}")
    for category in categories:
        if not category.focus:
            raise ValueError(f"category {category.slug!r} has an empty focus list")

    offsets = dict(raw["seed_offsets"])
    for name, value in offsets.items():
        if not OFFSET_MIN <= value <= OFFSET_MAX:
            raise ValueError(f"seed offset {name!r} must lie in [{OFFSET_MIN}, {OFFSET_MAX}]")
    if len(set(offsets.values())) != len(offsets):
        raise ValueError("seed offsets must be distinct")

    splits = {}
    if "splits" in raw:
        if set(raw["splits"]) != set(SPLITS):
            raise ValueError(f"splits must be exactly {SPLITS}")
        for name in SPLITS:
            count = 1 if name == UNSEEN_SPLIT else len(categories) - 1
            splits[name] = _load_split(name, raw["splits"][name], count)

    generator = GeneratorConfig(**raw["generator"]) if "generator" in raw else None
    build = BuildSettings(**raw["build"]) if "build" in raw else None
    prescreen = PrescreenConfig(**raw["prescreen"]) if "prescreen" in raw else None
    if build is not None:
        for name in ("jaccard_threshold", "cosine_threshold", "word_cap_share", "element_cap_share"):
            if not 0 < getattr(build, name) <= 1:
                raise ValueError(f"build setting {name!r} must lie in (0, 1]")

    return BuildConfig(
        spec_version=raw["spec_version"],
        base_seed=raw["base_seed"],
        held_out_category=held_out,
        seed_offsets=offsets,
        categories=categories,
        splits=splits,
        generator=generator,
        build=build,
        prescreen=prescreen,
    )
