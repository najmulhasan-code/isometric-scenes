import json
import random
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import numpy as np

from prompt_build.build import read_runs
from prompt_build.checks import check_counts, check_cross_split, check_elements
from prompt_build.config import SPLITS, load_config
from prompt_build.dedup import apply_word_cap, embedding_dedup, jaccard, jaccard_dedup
from prompt_build.lexicon import load_lexicon
from prompt_build.pool import Candidate, apply_element_cap, load_candidates, remove_motion_flagged
from prompt_build.splits import assign_splits
from prompt_build.strata import build_strata, distractor_slots

ROOT = Path(__file__).resolve().parent.parent
CONFIG = load_config(ROOT / "config.toml")
STRATA = build_strata(CONFIG)
LEXICON = load_lexicon(ROOT / "terms")
STOP = frozenset({"a", "an", "the", "with", "on", "and"})


def candidate(cid: int, text: str, stratum=None, element=None, flag=False) -> Candidate:
    return Candidate(cid, stratum or STRATA[0], 0, text, text, element, flag)


class LoadTest(unittest.TestCase):
    def test_ids_follow_stratum_and_request_order(self):
        in_scope = STRATA[1]
        records = [
            {"stratum": in_scope.key, "request_index": 1, "status": "ok", "items": ["a red kettle on a stove top"]},
            {"stratum": STRATA[0].key, "request_index": 0, "status": "ok", "items": ["a moon base with a dome"]},
            {"stratum": in_scope.key, "request_index": 0, "status": "malformed", "items": []},
        ]
        loaded = load_candidates(records, {s.key: s for s in STRATA}, LEXICON)
        self.assertEqual([c.stratum.index for c in loaded], [0, 1])
        self.assertEqual([c.candidate_id for c in loaded], [0, 1])

    def test_refill_runs_are_numbered_after_the_main_run(self):
        records = [
            {"run": "refill-1", "stratum": STRATA[0].key, "request_index": 5, "status": "ok",
             "items": ["a moon base with a dome and a red lamp"]},
            {"run": "main", "stratum": STRATA[1].key, "request_index": 0, "status": "ok",
             "items": ["a moon base with a dome and a blue lamp"]},
        ]
        loaded = load_candidates(records, {s.key: s for s in STRATA}, LEXICON, ["main", "refill-1"])
        self.assertEqual([c.request_index for c in loaded], [0, 5])

    def test_distractor_base_that_is_a_spec_example_is_removed(self):
        stratum = next(s for s in STRATA if s.distractor and s.kind == "animal")
        base = "a bakery counter with bread on wooden shelves"
        item = {"base": base, "with_element": base + " and a cat nearby"}
        records = [{"stratum": stratum.key, "request_index": 0, "status": "ok", "items": [item]}]
        loaded = load_candidates(records, {stratum.key: stratum}, LEXICON)
        self.assertIn("spec_example", loaded[0].removed)

    def test_motion_flagged_candidates_are_removed(self):
        pool = [candidate(0, "x", flag=True), candidate(1, "y")]
        self.assertEqual(remove_motion_flagged(pool), 1)
        self.assertEqual([c.removed for c in pool], ["motion_flag", None])

    def test_rule_failures_are_recorded(self):
        records = [{"stratum": STRATA[0].key, "request_index": 0, "status": "ok", "items": ["A Moon Base"]}]
        loaded = load_candidates(records, {s.key: s for s in STRATA}, LEXICON)
        self.assertTrue(loaded[0].removed.startswith("rules:"))


class ElementCapTest(unittest.TestCase):
    def test_no_element_above_the_cap(self):
        stratum = next(s for s in STRATA if s.distractor and s.kind == "animal")
        pool = [candidate(i, f"x{i}", stratum, "cat" if i < 5 else f"animal{i}") for i in range(20)]
        apply_element_cap(pool, 0.10, random.Random(1))
        kept = Counter(c.excluded_element for c in pool if c.kept)
        self.assertLessEqual(max(kept.values()) / sum(kept.values()), 0.10)
        self.assertTrue(all(c.removed in (None, "element_cap") for c in pool))

    def test_too_few_elements_is_an_error(self):
        stratum = next(s for s in STRATA if s.distractor and s.kind == "animal")
        pool = [candidate(i, f"x{i}", stratum, f"animal{i % 4}") for i in range(20)]
        with self.assertRaises(ValueError):
            apply_element_cap(pool, 0.10, random.Random(1))


class DedupTest(unittest.TestCase):
    def test_jaccard(self):
        self.assertEqual(jaccard(frozenset("ab"), frozenset("ab")), 1.0)
        self.assertEqual(jaccard(frozenset("ab"), frozenset("cd")), 0.0)

    def test_greedy_keeps_the_earlier_candidate(self):
        pool = [
            candidate(0, "a red kettle on a stove"),
            candidate(1, "a red kettle on the stove"),
            candidate(2, "a blue lamp on a desk"),
        ]
        jaccard_dedup(pool, STOP, 0.6)
        self.assertEqual([c.kept for c in pool], [True, False, True])
        self.assertEqual(pool[1].removed, "duplicate_jaccard:0")

    def test_ties_match_the_lowest_id(self):
        pool = [
            candidate(0, "a red kettle"),
            candidate(1, "a red stove"),
            candidate(2, "a red kettle stove"),
        ]
        near = jaccard_dedup(pool, STOP, 0.6)
        tie = next(pair for pair in near if pair.candidate_id == 2)
        self.assertAlmostEqual(tie.score, 2 / 3)
        self.assertEqual(pool[2].removed, "duplicate_jaccard:0")

    def test_removed_candidates_are_not_matched(self):
        pool = [candidate(0, "a red kettle"), candidate(1, "a red kettle on a stove")]
        pool[0].removed = "rules:R2"
        jaccard_dedup(pool, STOP, 0.6)
        self.assertTrue(pool[1].kept)

    def test_embedding_dedup(self):
        pool = [candidate(0, "one"), candidate(1, "two"), candidate(2, "three")]
        vectors = {
            "one": np.array([1.0, 0.0], dtype=np.float32),
            "two": np.array([0.99, 0.141], dtype=np.float32),
            "three": np.array([0.0, 1.0], dtype=np.float32),
        }
        embedding_dedup(pool, vectors, 0.85)
        self.assertEqual([c.kept for c in pool], [True, False, True])

    def test_word_cap(self):
        pool = [candidate(i, f"a small item{i}") for i in range(10)]
        _, binding = apply_word_cap(pool, STOP, frozenset(), 0.5, random.Random(0))
        kept = [c for c in pool if c.kept]
        self.assertEqual(binding, ["small"])
        self.assertLessEqual(sum("small" in c.prompt for c in kept), 0.5 * len(kept))

    def test_word_cap_exempts_category_nouns(self):
        pool = [candidate(i, f"a desk item{i}") for i in range(10)]
        apply_word_cap(pool, STOP, frozenset({"desk"}), 0.5, random.Random(0))
        self.assertTrue(all(c.kept for c in pool))


class SplitTest(unittest.TestCase):
    @staticmethod
    def full_pool(flag_every: int = 0) -> list[Candidate]:
        pool = []
        for stratum in STRATA:
            for _ in range(stratum.target):
                cid = len(pool)
                flagged = bool(flag_every) and cid % flag_every == 0
                pool.append(candidate(cid, f"text {cid}", stratum, "e", flagged))
        return pool

    def test_counts_match_the_spec(self):
        assignment = assign_splits(self.full_pool(), STRATA, distractor_slots(CONFIG), CONFIG)
        self.assertTrue(assignment.complete)
        expected = {"train": 1600, "val": 88, "test": 240, "test_unseen": 120}
        self.assertEqual({s: len(assignment.splits[s]) for s in SPLITS}, expected)

    def test_full_pool_passes_the_count_and_element_checks(self):
        pool = self.full_pool()
        for c in pool:
            c.excluded_element = f"e{c.candidate_id % 12}"
        assignment = assign_splits(pool, STRATA, distractor_slots(CONFIG), CONFIG)
        self.assertEqual(check_counts(assignment, distractor_slots(CONFIG), CONFIG), [])
        self.assertEqual(check_elements(assignment), [])

    def test_count_check_reports_a_short_split(self):
        pool = [c for c in self.full_pool() if c.stratum.index != STRATA[-1].index]
        assignment = assign_splits(pool, STRATA, distractor_slots(CONFIG), CONFIG)
        self.assertTrue(check_counts(assignment, distractor_slots(CONFIG), CONFIG))

    def test_cross_split_check_finds_a_train_duplicate(self):
        assignment = assign_splits(self.full_pool(), STRATA, distractor_slots(CONFIG), CONFIG)
        shared = assignment.splits["train"][0].base_prompt
        assignment.splits["test"][0].base_prompt = shared
        texts = {c.base_prompt for split in assignment.splits.values() for c in split}
        vectors = {t: np.eye(1, 4096, i, dtype=np.float32)[0] for i, t in enumerate(sorted(texts))}
        problems = check_cross_split(assignment, STOP, vectors, 0.6, 0.85)
        self.assertEqual(len(problems), 2)

    def test_assignment_is_deterministic(self):
        first = assign_splits(self.full_pool(), STRATA, distractor_slots(CONFIG), CONFIG)
        second = assign_splits(self.full_pool(), STRATA, distractor_slots(CONFIG), CONFIG)
        for split in SPLITS:
            self.assertEqual(
                [c.candidate_id for c in first.splits[split]],
                [c.candidate_id for c in second.splits[split]],
            )

    def test_shortfall_is_reported(self):
        pool = [c for c in self.full_pool() if c.stratum.index != STRATA[-1].index]
        assignment = assign_splits(pool, STRATA, distractor_slots(CONFIG), CONFIG)
        self.assertIn(STRATA[-1].key, assignment.shortfalls)


class ReadRunsTest(unittest.TestCase):
    def write_run(self, root: Path, name: str, manifest: dict, block: int = 1) -> None:
        folder = root / "runs" / name
        folder.mkdir(parents=True)
        (folder / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        record = {"stratum": STRATA[0].key, "request_index": 0, "status": "ok", "items": [],
                  "seed_block": block}
        (folder / "requests.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")

    def test_pilot_and_diagnostic_runs_are_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_run(root, "main", {"model": "m", "pilot_round": None, "overrides": {}})
            self.write_run(root, "pilot-2", {"model": "m", "pilot_round": 2, "overrides": {}}, 3)
            self.write_run(root, "diagnostic-x", {"model": "m", "pilot_round": None,
                                                   "overrides": {"use_json_schema": True}}, 10)
            self.assertEqual(len(read_runs(root, ["main"], "m")), 1)
            for name in ("pilot-2", "diagnostic-x"):
                with self.subTest(name=name), self.assertRaises(ValueError):
                    read_runs(root, ["main", name], "m")
            with self.assertRaises(ValueError):
                read_runs(root, ["main"], "another-model")


if __name__ == "__main__":
    unittest.main()
