"""Deduplication and the word cap.

Both deduplication passes are greedy in `candidate_id` order: a candidate is
removed when it is too similar to an earlier candidate that was kept. When
several earlier candidates score the same, the lowest id is the match.
Distractors are compared by their base prompt.
"""

import random
from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np

from .pool import Candidate
from .text import matching_tokens

NEAR_THRESHOLD_BAND = 0.10
SCORE_DECIMALS = 6


def content_tokens(text: str, stopwords: frozenset[str]) -> frozenset[str]:
    """The normalised tokens of a text, without stopwords."""
    return frozenset(token for token in matching_tokens(text) if token not in stopwords)


def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    union = a | b
    return len(a & b) / len(union) if union else 1.0


@dataclass(frozen=True)
class ScoredPair:
    """A candidate and its most similar earlier kept candidate."""

    candidate_id: int
    match_id: int
    score: float


def jaccard_dedup(
    candidates: list[Candidate], stopwords: frozenset[str], threshold: float
) -> list[ScoredPair]:
    """Remove near-duplicates by token-set Jaccard similarity.

    Returns, for every kept candidate checked, its best match when that
    score lies within 0.1 of the threshold, for the threshold check.
    """
    kept_tokens: dict[int, frozenset[str]] = {}
    postings: dict[str, list[int]] = defaultdict(list)
    near = []
    for candidate in sorted(candidates, key=lambda c: c.candidate_id):
        if not candidate.kept:
            continue
        tokens = content_tokens(candidate.base_prompt, stopwords)
        best_id, best = -1, 0.0
        for other in sorted({other for token in tokens for other in postings[token]}):
            score = jaccard(tokens, kept_tokens[other])
            if score > best:
                best_id, best = other, score
        if abs(best - threshold) <= NEAR_THRESHOLD_BAND:
            near.append(ScoredPair(candidate.candidate_id, best_id, best))
        if best >= threshold:
            candidate.removed = f"duplicate_jaccard:{best_id}"
            continue
        kept_tokens[candidate.candidate_id] = tokens
        for token in tokens:
            postings[token].append(candidate.candidate_id)
    return near


def embedding_dedup(
    candidates: list[Candidate], embeddings: dict[str, np.ndarray], threshold: float
) -> list[ScoredPair]:
    """Remove near-duplicates by cosine similarity of normalised sentence embeddings.

    Scores are computed in float64 and rounded to six decimals before the
    comparison, so a decision near the threshold does not depend on the
    machine's floating-point details.
    """
    ordered = [c for c in sorted(candidates, key=lambda c: c.candidate_id) if c.kept]
    if not ordered:
        return []
    width = len(next(iter(embeddings.values())))
    kept_matrix = np.zeros((len(ordered), width), dtype=np.float64)
    kept_ids: list[int] = []
    near = []
    for candidate in ordered:
        vector = np.asarray(embeddings[candidate.base_prompt], dtype=np.float64)
        best_id, best = -1, 0.0
        if kept_ids:
            scores = np.round(kept_matrix[: len(kept_ids)] @ vector, SCORE_DECIMALS)
            position = int(np.argmax(scores))
            best_id, best = kept_ids[position], float(scores[position])
        if abs(best - threshold) <= NEAR_THRESHOLD_BAND:
            near.append(ScoredPair(candidate.candidate_id, best_id, best))
        if best >= threshold:
            candidate.removed = f"duplicate_embedding:{best_id}"
            continue
        kept_matrix[len(kept_ids)] = vector
        kept_ids.append(candidate.candidate_id)
    return near


def word_frequencies(
    candidates: list[Candidate], stopwords: frozenset[str], exempt: frozenset[str]
) -> Counter:
    """How many kept prompts contain each content word, counted once per prompt."""
    return Counter(
        word
        for c in candidates
        if c.kept
        for word in content_tokens(c.base_prompt, stopwords) - exempt
    )


def apply_word_cap(
    candidates: list[Candidate],
    stopwords: frozenset[str],
    exempt: frozenset[str],
    share: float,
    rng: random.Random,
) -> tuple[int, list[str]]:
    """Drop candidates until no content word appears in more than `share` of the pool.

    A content word is a token that is neither a stopword nor an exempt
    category noun, counted once per prompt. Only candidates holding an
    over-cap word are dropped, one at a time in shuffled order, with the
    counts recomputed after each drop. Returns the number dropped and the
    words that were over the cap before any drop.
    """
    alive = [c for c in candidates if c.kept]
    words = {c.candidate_id: content_tokens(c.base_prompt, stopwords) - exempt for c in alive}
    counts = Counter(word for group in words.values() for word in group)
    total = len(alive)
    binding = sorted(word for word, count in counts.items() if count > share * total)
    order = list(alive)
    rng.shuffle(order)
    dropped = 0
    changed = True
    while changed:
        changed = False
        for candidate in order:
            if not candidate.kept:
                continue
            if any(counts[word] > share * total for word in words[candidate.candidate_id]):
                candidate.removed = "word_cap"
                counts.subtract(words[candidate.candidate_id])
                total -= 1
                dropped += 1
                changed = True
    return dropped, binding
