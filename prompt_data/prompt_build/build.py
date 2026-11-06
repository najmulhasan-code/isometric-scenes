"""The build: code rules, motion flags, element
cap, deduplication, word cap, split assignment and the final checks.

The build reads the main run and any refill runs, and writes its outputs to
builds/<name>/: every candidate with its outcome, a stage-by-stage summary
with word frequencies, the embeddings used for deduplication, and pairs
near each similarity threshold for the threshold check. Removals are
recorded, never deleted. Pilot and diagnostic runs are refused.
"""

import dataclasses
import json
import random
from dataclasses import dataclass
from pathlib import Path

from .checks import check_counts, check_cross_split, check_elements
from .config import BuildConfig
from .dedup import ScoredPair, apply_word_cap, embedding_dedup, jaccard_dedup, word_frequencies
from .embed import encode
from .generate import RunLog
from .lexicon import load_lexicon, read_word_list
from .plan import MAIN_BLOCK
from .pool import Candidate, apply_element_cap, load_candidates, remove_motion_flagged
from .splits import Assignment, assign_splits
from .strata import build_strata, distractor_slots
from .text import matching_tokens
from .threshold_check import select_pairs, write_check

THRESHOLD_SAMPLE = 200
TOP_WORDS = 20


@dataclass
class BuildResult:
    candidates: list[Candidate]
    assignment: Assignment
    stages: list[tuple[str, int]]
    near_pairs: dict[str, list[ScoredPair]]
    words: dict[str, object]
    problems: list[str]


def word_set(path: Path) -> frozenset[str]:
    return frozenset(token for entry in read_word_list(path) for token in matching_tokens(entry))


def read_runs(root: Path, run_names: list[str], model: str) -> list[dict]:
    """Records of the named runs, refusing any run that is not a main-run or refill run."""
    records = []
    for name in run_names:
        manifest = json.loads((root / "runs" / name / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("pilot_round") is not None or manifest.get("overrides"):
            raise ValueError(f"run {name!r} is a pilot or diagnostic run and cannot enter the pool")
        if manifest.get("model") != model:
            raise ValueError(f"run {name!r} used model {manifest.get('model')!r}, not {model!r}")
        for record in RunLog(root / "runs" / name / "requests.jsonl").read():
            if record.get("seed_block") != MAIN_BLOCK:
                raise ValueError(f"run {name!r} holds a request outside the main seed block")
            records.append(record)
    return records


def build_settings(config: BuildConfig, build_dir: Path) -> BuildConfig:
    """The config with the cosine threshold the build was made with."""
    summary = json.loads((build_dir / "summary.json").read_text(encoding="utf-8"))
    if "cosine_threshold" not in summary:
        raise ValueError(f"{build_dir.name}: summary.json does not record the cosine threshold; rebuild it")
    settings = dataclasses.replace(config.build, cosine_threshold=summary["cosine_threshold"])
    return dataclasses.replace(config, build=settings)


def run_build(
    config: BuildConfig, root: Path, run_names: list[str], out_dir: Path, runs_root: Path | None = None
) -> BuildResult:
    """Build from the named runs. Run logs are read from `runs_root`/runs, or `root`/runs."""
    if config.build is None or config.generator is None:
        raise ValueError("config.toml needs [generator] and [build] sections")
    settings, offsets, seed = config.build, config.seed_offsets, config.base_seed
    lexicon = load_lexicon(root / "terms")
    stopwords = word_set(root / "terms" / "stopwords.txt")
    exempt = word_set(root / "terms" / "category_nouns.txt")
    strata = build_strata(config)
    slots = distractor_slots(config)

    records = read_runs(runs_root or root, run_names, config.generator.model)
    candidates = load_candidates(records, {s.key: s for s in strata}, lexicon, run_names)

    def kept() -> int:
        return sum(c.kept for c in candidates)

    stages = [("parsed items", len(candidates)), ("code rules", kept())]
    remove_motion_flagged(candidates)
    stages.append(("motion flags", kept()))
    apply_element_cap(candidates, settings.element_cap_share, random.Random(seed + offsets["element_cap"]))
    stages.append(("element cap", kept()))

    near = {"jaccard": jaccard_dedup(candidates, stopwords, settings.jaccard_threshold)}
    stages.append(("Jaccard deduplication", kept()))
    embeddings = encode(
        [c.base_prompt for c in candidates if c.kept],
        settings.encoder_model,
        settings.encoder_revision,
        out_dir,
    )
    near["embedding"] = embedding_dedup(candidates, embeddings, settings.cosine_threshold)
    stages.append(("embedding deduplication", kept()))

    before_cap = word_frequencies(candidates, stopwords, exempt)
    _, binding = apply_word_cap(
        candidates, stopwords, exempt, settings.word_cap_share, random.Random(seed + offsets["word_cap"])
    )
    stages.append(("word cap", kept()))
    words = {
        "before_cap": before_cap.most_common(TOP_WORDS),
        "after_cap": word_frequencies(candidates, stopwords, exempt).most_common(TOP_WORDS),
        "capped": binding,
    }

    assignment = assign_splits(candidates, strata, slots, config)
    problems = check_counts(assignment, slots, config) + check_elements(assignment)
    if not problems:
        problems = check_cross_split(
            assignment, stopwords, embeddings, settings.jaccard_threshold, settings.cosine_threshold
        )
    return BuildResult(candidates, assignment, stages, near, words, problems)


def _cell(text: str) -> str:
    return text.replace("\t", " ").replace("\n", " ")


def write_outputs(result: BuildResult, config: BuildConfig, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "candidates.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for c in result.candidates:
            row = {
                "candidate_id": c.candidate_id,
                "stratum": c.stratum.key,
                "request_index": c.request_index,
                "prompt": c.prompt,
                "base_prompt": c.base_prompt,
                "excluded_element": c.excluded_element,
                "motion_flag": c.motion_flag,
                "removed": c.removed,
                "split": c.split,
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    texts = {c.candidate_id: c.base_prompt for c in result.candidates}
    rng = random.Random(config.base_seed + config.seed_offsets["threshold_sample"])
    with (out_dir / "threshold_pairs.tsv").open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("method\tscore\tcandidate\tearlier_kept_candidate\n")
        for method, pairs in result.near_pairs.items():
            for pair in rng.sample(pairs, min(THRESHOLD_SAMPLE, len(pairs))):
                handle.write(
                    f"{method}\t{pair.score:.3f}\t{_cell(texts[pair.candidate_id])}"
                    f"\t{_cell(texts[pair.match_id])}\n"
                )

    if config.build is not None:
        thresholds = {"jaccard": config.build.jaccard_threshold, "embedding": config.build.cosine_threshold}
        seed = config.base_seed + config.seed_offsets["threshold_sample"]
        write_check(select_pairs(result.near_pairs, texts, thresholds, seed), out_dir)

    strata = {c.stratum.index: c.stratum.key for c in result.candidates}
    reserves = {
        strata[index]: [c.candidate_id for c in items]
        for index, items in sorted(result.assignment.reserves.items())
        if index in strata
    }
    (out_dir / "reserves.json").write_text(
        json.dumps(reserves, indent=1) + "\n", encoding="utf-8", newline="\n"
    )

    summary = {
        "stages": result.stages,
        "splits": {split: len(items) for split, items in sorted(result.assignment.splits.items())},
        "reserves": sum(len(items) for items in result.assignment.reserves.values()),
        "cosine_threshold": config.build.cosine_threshold,
        "shortfalls": result.assignment.shortfalls,
        "near_threshold_pairs": {method: len(pairs) for method, pairs in result.near_pairs.items()},
        "words": result.words,
        "problems": result.problems,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
