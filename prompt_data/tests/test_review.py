import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from prompt_build.config import load_config
from prompt_build.review import BATCH_SIZE, SPOT_CHECK_SIZE, Review, wilson_interval

ROOT = Path(__file__).resolve().parent.parent
CONFIG = load_config(ROOT / "config.toml")
SMALL = replace(CONFIG, build=replace(CONFIG.build, audit_size=3))
CATEGORIES = {c.slug: c.name for c in CONFIG.categories}


def write_build(directory: Path) -> None:
    rows = []
    splits = ["val", "test", "test_unseen"] * 2 + ["train"] * 6
    for cid, split in enumerate(splits):
        rows.append({"candidate_id": cid, "stratum": "in_scope/space/motion", "prompt": f"p{cid}",
                     "base_prompt": f"p{cid}", "excluded_element": None, "split": split, "removed": None})
    for cid in range(12, 16):
        rows.append({"candidate_id": cid, "stratum": "in_scope/space/motion", "prompt": f"p{cid}",
                     "base_prompt": f"p{cid}", "excluded_element": None, "split": None, "removed": None})
    (directory / "candidates.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    (directory / "reserves.json").write_text(json.dumps({"in_scope/space/motion": [12, 13, 14, 15]}), encoding="utf-8")


class StatisticsTest(unittest.TestCase):
    def test_wilson_interval_matches_known_values(self):
        low, high = wilson_interval(0, 50)
        self.assertEqual(low, 0.0)
        self.assertAlmostEqual(high, 0.0713, places=4)
        low, high = wilson_interval(6, 212)
        self.assertAlmostEqual(low, 0.0130, places=4)
        self.assertAlmostEqual(high, 0.0604, places=4)


class ReviewTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name)
        write_build(self.path)
        self.review = Review(self.path, SMALL)
        self.review.start()

    def tearDown(self):
        self.directory.cleanup()

    def item(self, role: str) -> dict:
        return next(i for i in self.review.current["items"] if i["role"] == role)

    def flag(self, flags: dict[int, str], sample: int = SPOT_CHECK_SIZE) -> None:
        self.review.flag(flags)
        self.review.spot_check(sample)

    def test_first_round_holds_every_eval_prompt_and_the_audit(self):
        roles = [i["role"] for i in self.review.current["items"]]
        self.assertEqual((roles.count("eval"), roles.count("audit")), (6, 3))
        self.assertLessEqual(self.review.batches(), 1 + 9 // BATCH_SIZE)

    def test_flagged_eval_item_is_removed_replaced_and_screened_next_round(self):
        item = self.item("eval")
        self.flag({item["number"]: "R4"})
        self.review.decide({})
        self.assertEqual(self.review.close(), 2)
        state = self.review.state
        self.assertEqual(state["removed"][str(item["candidate_id"])], "R4")
        self.assertEqual(state["replaced"][str(item["candidate_id"])], 12)
        roles = sorted(i["role"] for i in self.review.current["items"])
        self.assertEqual(roles, ["audit", "eval"])

    def test_flagged_audit_item_is_replaced_without_review(self):
        item = self.item("audit")
        self.flag({item["number"]: "R8"})
        self.review.decide({})
        self.assertEqual(self.review.close(), 0)
        self.assertEqual(self.review.audit_result(), (1, 3))

    def test_author_removes_a_sampled_item(self):
        self.flag({})
        sampled = self.review.current["spot_check"][0]
        self.review.decide({sampled: "R3"})
        self.assertEqual(self.review.current["marks"], {str(sampled): "R3"})
        summary = self.review.summary()
        self.assertEqual((summary["rounds"][0]["spot_check_removed_by_author"], summary["rounds"][0]["removed"]), (1, 1))
        role = next(i["role"] for i in self.review.current["items"] if i["number"] == sampled)
        self.assertEqual(summary["removed_by_rule"][role], {"R3": 1})

    def test_author_may_only_remove_sampled_items(self):
        self.flag({}, sample=1)
        outside = next(i["number"] for i in self.review.current["items"]
                       if i["number"] not in self.review.current["spot_check"])
        with self.assertRaises(ValueError):
            self.review.decide({outside: "R3"})

    def test_later_rounds_have_no_sample(self):
        self.flag({self.item("eval")["number"]: "R4"})
        self.review.decide({})
        self.review.close()
        self.review.flag({})
        self.assertEqual(self.review.spot_check(0), [])
        self.review.decide({})
        self.assertEqual(self.review.close(), 0)
        rounds = self.review.summary()["rounds"]
        self.assertIn("miss_rate_wilson_95", rounds[0])
        self.assertNotIn("miss_rate_wilson_95", rounds[1])

    def test_state_survives_reload(self):
        self.review.flag({self.item("eval")["number"]: "R3"})
        again = Review(self.path, SMALL)
        self.assertEqual(again.current["model_flags"], self.review.current["model_flags"])

    def test_spot_check_excludes_flagged_items(self):
        flagged = self.item("eval")["number"]
        self.review.flag({flagged: "R3"})
        sample = self.review.spot_check(4)
        self.assertNotIn(flagged, sample)
        self.assertEqual(len(sample), 4)

    def test_flags_are_fixed_once_the_sample_is_drawn(self):
        self.flag({})
        with self.assertRaises(ValueError):
            self.review.flag({1: "R4"})

    def test_sheet_lists_the_sample_only(self):
        self.flag({self.item("eval")["number"]: "R3"}, sample=4)
        self.assertEqual(len(self.review.sheet(CATEGORIES)), 4)

    def test_close_needs_the_spot_check_result(self):
        self.flag({})
        with self.assertRaises(ValueError):
            self.review.close()

    def test_coverage_is_complete_after_a_clean_round(self):
        self.assertTrue(self.review.coverage_problems())
        self.flag({})
        self.review.decide({})
        self.review.close()
        self.assertEqual(self.review.coverage_problems(), [])

    def test_running_out_of_reserves_raises(self):
        eval_items = [i for i in self.review.current["items"] if i["role"] == "eval"]
        self.flag({i["number"]: "R4" for i in eval_items[:5]})
        self.review.decide({})
        with self.assertRaises(ValueError):
            self.review.close()

    def test_late_reserves_are_used_for_replacement(self):
        self.review.state["reserves"]["in_scope/space/motion"] = []
        self.review.add_reserves([{"candidate_id": 99, "stratum": "in_scope/space/motion", "prompt": "p99",
                                  "base_prompt": "p99", "excluded_element": None, "split": None,
                                  "removed": None}])
        item = self.item("eval")
        self.flag({item["number"]: "R3"})
        self.review.decide({})
        self.review.close()
        self.assertEqual(self.review.state["replaced"][str(item["candidate_id"])], 99)

    def test_bad_flags_are_refused(self):
        with self.assertRaises(ValueError):
            self.review.flag({999: "R4"})
        with self.assertRaises(ValueError):
            self.review.flag({1: "looks odd"})


if __name__ == "__main__":
    unittest.main()
