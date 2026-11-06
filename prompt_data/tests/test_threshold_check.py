import unittest

from prompt_build.dedup import ScoredPair
from prompt_build.threshold_check import (
    EMBEDDING_PER_SIDE,
    HIGHER_THRESHOLD,
    LOWER_THRESHOLD,
    decide,
    select_pairs,
)

THRESHOLDS = {"jaccard": 0.6, "embedding": 0.85}


def key(method: str, side: str, numbers: range) -> list[dict]:
    return [{"number": n, "method": method, "side": side, "score": 0.0} for n in numbers]


class SelectTest(unittest.TestCase):
    def test_sample_is_blind_shuffled_and_sized(self):
        texts = {i: f"text {i}" for i in range(400)}
        near = {
            "jaccard": [ScoredPair(i, i + 200, 0.55) for i in range(5)],
            "embedding": [ScoredPair(i, i + 200, 0.80 + 0.001 * i) for i in range(10, 110)],
        }
        items = select_pairs(near, texts, THRESHOLDS, seed=3)
        entries = [entry for item in items for entry in item.entries]
        self.assertEqual(sum(e[0] == "jaccard" for e in entries), 5)
        self.assertEqual(sum(e[:2] == ("embedding", "removed") for e in entries), EMBEDDING_PER_SIDE)
        self.assertEqual(sum(e[:2] == ("embedding", "kept") for e in entries), EMBEDDING_PER_SIDE)
        self.assertEqual([i.number for i in items], list(range(1, len(items) + 1)))
        self.assertEqual(items, select_pairs(near, texts, THRESHOLDS, seed=3))

    def test_a_pair_sampled_by_both_methods_is_shown_once(self):
        texts = {0: "first", 1: "second"}
        near = {"jaccard": [ScoredPair(1, 0, 0.62)], "embedding": [ScoredPair(1, 0, 0.88)]}
        items = select_pairs(near, texts, THRESHOLDS, seed=1)
        self.assertEqual(len(items), 1)
        self.assertEqual({e[0] for e in items[0].entries}, {"jaccard", "embedding"})


class DecideTest(unittest.TestCase):
    KEY = key("embedding", "removed", range(1, 31)) + key("embedding", "kept", range(31, 61))

    def test_keeps_the_threshold_when_judgements_agree(self):
        same = set(range(1, 31)) | {31, 32}
        self.assertEqual(decide(self.KEY, same, 0.85)[0], 0.85)

    def test_lowers_when_many_kept_pairs_are_the_same_scene(self):
        same = set(range(1, 31)) | set(range(31, 38))
        self.assertEqual(decide(self.KEY, same, 0.85)[0], LOWER_THRESHOLD)

    def test_raises_when_many_removed_pairs_differ(self):
        same = set(range(1, 23))
        self.assertEqual(decide(self.KEY, same, 0.85)[0], HIGHER_THRESHOLD)

    def test_counts_are_reported(self):
        _, counts = decide(self.KEY, {1, 31}, 0.85)
        self.assertEqual(counts["embedding_removed"], {"pairs": 30, "judged_same": 1})
        self.assertEqual(counts["embedding_kept"], {"pairs": 30, "judged_same": 1})


if __name__ == "__main__":
    unittest.main()
