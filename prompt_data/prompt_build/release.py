"""The released dataset files.

The final splits come from the build and the review state. Before anything
is written, the release checks that review covered every evaluation prompt
and the full audit, re-runs the code rules on every final row, and re-runs
the final checks; it stops if any fails. Each split is then
written as one JSONL file: UTF-8 without a byte order mark, one object per
line with keys in a fixed field order, and `\\n` line endings. Rows are
sorted by candidate id, so the files do not depend on review order.
"""

import hashlib
import json
from pathlib import Path

from .build import build_settings, word_set
from .checks import check_counts, check_cross_split, check_elements
from .config import SPLITS, BuildConfig
from .embed import encode
from .lexicon import Lexicon, load_lexicon
from .pool import Candidate
from .review import Review
from .rules import check_distractor, in_scope_failures
from .splits import Assignment
from .strata import build_strata, distractor_slots

KEY_ORDER = (
    "idx",
    "id",
    "split",
    "category",
    "prompt",
    "base_prompt",
    "distractor",
    "excluded_element",
    "excluded_element_type",
    "motion_in_prompt",
    "candidate_id",
    "spec_version",
)


def final_assignment(review: Review, config: BuildConfig) -> Assignment:
    strata = {s.key: s for s in build_strata(config)}
    assignment = Assignment()
    for cid, split in sorted(review.state["splits"].items(), key=lambda item: int(item[0])):
        row = review.rows[int(cid)]
        candidate = Candidate(
            row.candidate_id, strata[row.stratum], 0, row.prompt, row.base_prompt,
            row.excluded_element, False, None, split,
        )
        assignment.splits[split].append(candidate)
    return assignment


def rule_problems(assignment: Assignment, lexicon: Lexicon) -> list[str]:
    """Final rows that fail a code rule."""
    problems = []
    for split, items in assignment.splits.items():
        for c in items:
            if c.distractor:
                failures = check_distractor(c.prompt, c.base_prompt, c.stratum.kind, lexicon).failures
            else:
                failures = in_scope_failures(c.prompt, lexicon)
            if failures:
                problems.append(f"{split} candidate {c.candidate_id} fails {','.join(failures)}")
    return problems


def final_problems(
    assignment: Assignment,
    config: BuildConfig,
    lexicon: Lexicon,
    stopwords: frozenset[str],
    cache_dir: Path,
) -> list[str]:
    problems = (
        rule_problems(assignment, lexicon)
        + check_counts(assignment, distractor_slots(config), config)
        + check_elements(assignment)
    )
    if problems:
        return problems
    texts = [c.base_prompt for items in assignment.splits.values() for c in items]
    settings = config.build
    embeddings = encode(texts, settings.encoder_model, settings.encoder_revision, cache_dir)
    return check_cross_split(
        assignment, stopwords, embeddings, settings.jaccard_threshold, settings.cosine_threshold
    )


def record(candidate: Candidate, split: str, idx: int, spec_version: str) -> dict:
    stratum = candidate.stratum
    row = {
        "idx": idx,
        "id": f"{split}-{idx:04d}",
        "split": split,
        "category": stratum.category.slug,
        "prompt": candidate.prompt,
        "base_prompt": candidate.base_prompt,
        "distractor": stratum.distractor,
        "excluded_element": candidate.excluded_element,
        "excluded_element_type": stratum.kind,
        "motion_in_prompt": stratum.motion,
        "candidate_id": candidate.candidate_id,
        "spec_version": spec_version,
    }
    return {key: row[key] for key in KEY_ORDER}


def write_release(assignment: Assignment, config: BuildConfig, out_dir: Path) -> dict[str, str]:
    """Write one file per split and return each file's SHA-256."""
    out_dir.mkdir(parents=True, exist_ok=True)
    digests = {}
    for split in SPLITS:
        rows = sorted(assignment.splits[split], key=lambda c: c.candidate_id)
        lines = [
            json.dumps(record(c, split, idx, config.spec_version), ensure_ascii=False)
            for idx, c in enumerate(rows)
        ]
        data = ("\n".join(lines) + "\n").encode("utf-8")
        path = out_dir / f"{split}.jsonl"
        path.write_bytes(data)
        digests[path.name] = hashlib.sha256(data).hexdigest()
    return digests


def release_build(config: BuildConfig, root: Path, build_dir: Path) -> tuple[list[str], dict[str, str]]:
    """Check a reviewed build and, if every check passes, write the split files and SHA256SUMS.

    Returns the problems found, and the SHA-256 of each file written.
    """
    config = build_settings(config, build_dir)
    review = Review(build_dir, config)
    assignment = final_assignment(review, config)
    stopwords = word_set(root / "terms" / "stopwords.txt")
    problems = review.coverage_problems() or final_problems(
        assignment, config, load_lexicon(root / "terms"), stopwords, build_dir / "release_check"
    )
    if problems:
        return problems, {}
    out_dir = build_dir / "release"
    digests = write_release(assignment, config, out_dir)
    (out_dir / "SHA256SUMS").write_text(
        "".join(f"{digest}  {name}\n" for name, digest in sorted(digests.items())),
        encoding="utf-8", newline="\n",
    )
    return [], digests
