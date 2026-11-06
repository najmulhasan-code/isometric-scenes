"""The threshold check.

A person reads pairs close to each deduplication threshold and says whether
each pair describes the same scene. The pairs are shuffled together and
shown without scores or outcomes, so the reader cannot tell which pairs
were removed. The key is written to a separate file.

Sample: every Jaccard pair within 0.1 of its threshold, and a seeded sample
of 30 embedding pairs just above the cosine threshold (removed) and 30 just
below it (kept).
"""

import json
import random
from dataclasses import dataclass
from pathlib import Path

from .dedup import ScoredPair

EMBEDDING_PER_SIDE = 30
DECISION_SHARE = 0.20
LOWER_THRESHOLD = 0.80
HIGHER_THRESHOLD = 0.90


@dataclass(frozen=True)
class CheckItem:
    """One pair to read. A pair sampled for both methods is shown once."""

    number: int
    first: str
    second: str
    entries: tuple[tuple[str, str, float], ...]


def select_pairs(
    near: dict[str, list[ScoredPair]],
    texts: dict[int, str],
    thresholds: dict[str, float],
    seed: int,
) -> list[CheckItem]:
    rng = random.Random(seed)
    chosen = [("jaccard", pair) for pair in near["jaccard"]]
    cosine = thresholds["embedding"]
    removed = sorted((p for p in near["embedding"] if p.score >= cosine), key=lambda p: p.candidate_id)
    kept = sorted((p for p in near["embedding"] if p.score < cosine), key=lambda p: p.candidate_id)
    for side in (removed, kept):
        chosen += [("embedding", pair) for pair in rng.sample(side, min(EMBEDDING_PER_SIDE, len(side)))]

    grouped: dict[tuple[int, int], list[tuple[str, str, float]]] = {}
    for method, pair in chosen:
        side = "removed" if pair.score >= thresholds[method] else "kept"
        grouped.setdefault((pair.match_id, pair.candidate_id), []).append((method, side, pair.score))
    pairs = sorted(grouped)
    rng.shuffle(pairs)
    return [
        CheckItem(number, texts[first], texts[second], tuple(grouped[first, second]))
        for number, (first, second) in enumerate(pairs, start=1)
    ]


def write_check(items: list[CheckItem], out_dir: Path) -> None:
    """Write the blind list for the reader and the key, in separate files."""
    lines = [f"{item.number}. A: {item.first}\n   B: {item.second}" for item in items]
    (out_dir / "threshold_check.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8", newline="\n"
    )
    key = [
        {"number": item.number, "method": method, "side": side, "score": round(score, 6)}
        for item in items
        for method, side, score in item.entries
    ]
    (out_dir / "threshold_check_key.json").write_text(
        json.dumps(key, indent=2) + "\n", encoding="utf-8", newline="\n"
    )


def decide(key: list[dict], judged_same: set[int], threshold: float) -> tuple[float, dict]:
    """Apply the decision rule to the embedding pairs, given the numbers judged the same scene.

    If more than 20% of kept pairs are judged the same scene, the threshold is
    lowered to 0.80. If more than 20% of removed pairs are judged different,
    it is raised to 0.90. Otherwise it stays.
    """
    counts = {}
    for method in ("jaccard", "embedding"):
        for side in ("removed", "kept"):
            items = [k for k in key if k["method"] == method and k["side"] == side]
            same = sum(k["number"] in judged_same for k in items)
            counts[f"{method}_{side}"] = {"pairs": len(items), "judged_same": same}
    kept, removed = counts["embedding_kept"], counts["embedding_removed"]
    if kept["pairs"] and kept["judged_same"] > DECISION_SHARE * kept["pairs"]:
        return LOWER_THRESHOLD, counts
    different = removed["pairs"] - removed["judged_same"]
    if removed["pairs"] and different > DECISION_SHARE * removed["pairs"]:
        return HIGHER_THRESHOLD, counts
    return threshold, counts
