"""Final checks on a split assignment.

Each check returns a list of problems; an empty list means it passed. The
build fails when any check reports a problem.
"""

from collections import Counter

import numpy as np

from .config import KIND_ORDER, SPLITS, UNSEEN_SPLIT, BuildConfig
from .dedup import SCORE_DECIMALS, content_tokens, jaccard
from .splits import Assignment
from .strata import Slot

MIN_DISTINCT_ELEMENTS = 10


def check_counts(assignment: Assignment, slots: list[Slot], config: BuildConfig) -> list[str]:
    """Every split matches its quotas by category and motion label, and every slot is filled."""
    problems = []
    for split in SPLITS:
        quota = config.splits[split]
        placed = assignment.splits.get(split, [])
        in_scope = Counter(
            (c.stratum.category.slug, c.stratum.motion) for c in placed if not c.distractor
        )
        for category in config.split_categories(split):
            for motion in (True, False):
                found, wanted = in_scope[category.slug, motion], quota.in_scope[motion]
                if found != wanted:
                    problems.append(
                        f"{split}/{category.slug}/{'motion' if motion else 'still'}: {found} of {wanted}"
                    )
        found = Counter(
            (c.stratum.category.slug, c.stratum.kind, c.stratum.motion) for c in placed if c.distractor
        )
        wanted = Counter((s.category.slug, s.kind, s.motion) for s in slots if s.split == split)
        if found != wanted:
            missing = sum((wanted - found).values())
            problems.append(f"{split}: distractor slots differ from the slot table ({missing} unfilled)")
    return problems


def check_elements(assignment: Assignment) -> list[str]:
    """The dataset holds at least ten distinct excluded elements of every kind."""
    distinct = {kind: set() for kind in KIND_ORDER}
    for placed in assignment.splits.values():
        for candidate in placed:
            if candidate.distractor:
                distinct[candidate.stratum.kind].add(candidate.excluded_element)
    return [
        f"{kind}: only {len(found)} distinct excluded elements"
        for kind, found in distinct.items()
        if len(found) < MIN_DISTINCT_ELEMENTS
    ]


def check_cross_split(
    assignment: Assignment,
    stopwords: frozenset[str],
    embeddings: dict[str, np.ndarray],
    jaccard_threshold: float,
    cosine_threshold: float,
) -> list[str]:
    """No evaluation prompt is a near-duplicate of a train prompt by either measure."""
    train = assignment.splits.get("train", [])
    if not train:
        return []
    train_tokens = [content_tokens(c.base_prompt, stopwords) for c in train]
    train_matrix = np.array([embeddings[c.base_prompt] for c in train], dtype=np.float64)
    problems = []
    for split in ("val", "test", UNSEEN_SPLIT):
        for candidate in assignment.splits.get(split, []):
            tokens = content_tokens(candidate.base_prompt, stopwords)
            if any(jaccard(tokens, other) >= jaccard_threshold for other in train_tokens):
                problems.append(f"{split} candidate {candidate.candidate_id}: Jaccard match in train")
            vector = np.asarray(embeddings[candidate.base_prompt], dtype=np.float64)
            if np.round(train_matrix @ vector, SCORE_DECIMALS).max() >= cosine_threshold:
                problems.append(f"{split} candidate {candidate.candidate_id}: embedding match in train")
    return problems
