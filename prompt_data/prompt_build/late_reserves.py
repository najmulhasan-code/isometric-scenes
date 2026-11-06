"""Reserves for a stratum that runs out during review.

New candidates come from a refill run for the stratum. They are numbered
after every existing candidate and pass the same steps as the pool:
code rules, motion flags, deduplication against every candidate still in
the pool, the word cap and the element cap. Earlier candidates and
decisions are never changed. The survivors are shuffled with the stratum's
seed and appended to its reserves.
"""

import random
from collections import Counter
from pathlib import Path

from .build import build_settings, read_runs, word_set
from .config import BuildConfig
from .dedup import content_tokens, jaccard, word_frequencies
from .embed import encode
from .lexicon import Lexicon, load_lexicon
from .pool import Candidate, load_candidates, remove_motion_flagged
from .review import Review
from .strata import Stratum, build_strata


def screen_new(
    records: list[dict],
    stratum: Stratum,
    pool: list[Candidate],
    config: BuildConfig,
    lexicon: Lexicon,
    stopwords: frozenset[str],
    exempt: frozenset[str],
    cache_dir: Path,
) -> list[Candidate]:
    """New candidates for `stratum` that pass every pool step, in seeded order."""
    settings = config.build
    new = load_candidates(records, {stratum.key: stratum}, lexicon)
    first_id = max(c.candidate_id for c in pool) + 1
    for offset, candidate in enumerate(new):
        candidate.candidate_id = first_id + offset
    remove_motion_flagged(new)

    kept = [c for c in pool if c.kept]
    texts = [c.base_prompt for c in kept] + [c.base_prompt for c in new if c.kept]
    embeddings = encode(texts, settings.encoder_model, settings.encoder_revision, cache_dir)
    frequencies = word_frequencies(pool, stopwords, exempt)
    total = len(kept)
    elements = Counter(
        c.excluded_element for c in kept if c.distractor and c.stratum.kind == stratum.kind
    )
    kind_total = sum(elements.values())

    accepted: list[Candidate] = []
    for candidate in new:
        if not candidate.kept:
            continue
        tokens = content_tokens(candidate.base_prompt, stopwords)
        vector = embeddings[candidate.base_prompt]
        earlier = kept + accepted
        if any(jaccard(tokens, content_tokens(o.base_prompt, stopwords)) >= settings.jaccard_threshold
               for o in earlier):
            candidate.removed = "duplicate_jaccard"
        elif any(round(float(vector @ embeddings[o.base_prompt]), 6) >= settings.cosine_threshold
                 for o in earlier):
            candidate.removed = "duplicate_embedding"
        elif any(frequencies[word] + 1 > settings.word_cap_share * (total + 1)
                 for word in tokens - exempt):
            candidate.removed = "word_cap"
        elif candidate.distractor and (
            elements[candidate.excluded_element] + 1 > settings.element_cap_share * (kind_total + 1)
        ):
            candidate.removed = "element_cap"
        else:
            accepted.append(candidate)
            frequencies.update(tokens - exempt)
            elements[candidate.excluded_element] += 1
            total += 1
            kind_total += 1

    random.Random(config.base_seed + stratum.index).shuffle(accepted)
    return accepted


def add_late_reserves(
    config: BuildConfig, root: Path, runs_root: Path, build_dir: Path, stratum_key: str, run: str
) -> int:
    """Screen refill run `run` for one stratum and append the survivors to the review's reserves.

    Word lists come from `root` and run logs from `runs_root`/runs. Returns
    the number of new reserves.
    """
    config = build_settings(config, build_dir)
    strata = {s.key: s for s in build_strata(config)}
    review = Review(build_dir, config)
    pool = [
        Candidate(row.candidate_id, strata[row.stratum], 0, row.prompt, row.base_prompt,
                  row.excluded_element, False, row.removed)
        for row in review.rows.values()
    ]
    accepted = screen_new(
        read_runs(runs_root, [run], config.generator.model),
        strata[stratum_key],
        pool,
        config,
        load_lexicon(root / "terms"),
        word_set(root / "terms" / "stopwords.txt"),
        word_set(root / "terms" / "category_nouns.txt"),
        build_dir / "review",
    )
    review.add_reserves([
        {"candidate_id": c.candidate_id, "stratum": c.stratum.key, "prompt": c.prompt,
         "base_prompt": c.base_prompt, "excluded_element": c.excluded_element,
         "split": None, "removed": None, "run": run}
        for c in accepted
    ])
    return len(accepted)
