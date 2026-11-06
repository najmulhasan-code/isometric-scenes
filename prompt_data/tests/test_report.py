import unittest
from pathlib import Path

from prompt_build.lexicon import load_lexicon
from prompt_build.report import StratumInfo, build_report, format_report, hand_check_sample

LEXICON = load_lexicon(Path(__file__).resolve().parent.parent / "terms")


def record(stratum: str, items: list, status: str = "ok") -> dict:
    return {"stratum": stratum, "status": status, "items": items, "size": len(items)}


RECORDS = [
    record(
        "in_scope/home_interiors/motion",
        ["a small kitchen with a kettle steaming on the stove", "A Kitchen In Paris"],
    ),
    record(
        "distractor/workspaces/animal/still",
        [
            {"base": "a desk with a brass lamp", "with_element": "a desk with a cat and a brass lamp"},
            {"base": "a desk with a brass lamp", "with_element": "a desk a cat with a brass lamp"},
        ],
    ),
    record("in_scope/space/still", [], status="length"),
]


class ReportTest(unittest.TestCase):
    def test_stratum_keys(self):
        self.assertEqual(
            StratumInfo.from_key("distractor/workspaces/animal/still"),
            StratumInfo(True, "workspaces", "animal", False),
        )
        self.assertEqual(
            StratumInfo.from_key("in_scope/space/motion"), StratumInfo(False, "space", None, True)
        )

    def test_counts(self):
        report = build_report(RECORDS, LEXICON)
        self.assertEqual((report.requests, report.parsed, report.length), (3, 2, 1))
        self.assertEqual((report.in_scope_items, report.in_scope_passed), (2, 1))
        self.assertEqual((report.distractor_items, report.distractor_passed), (2, 2))
        self.assertEqual(report.elements["animal"]["cat"], 2)
        self.assertEqual(report.failures["R6"], 1)
        self.assertEqual((report.items_total, report.well_formed), (4, 3))

    def test_degenerate_logged_as_ok_is_recounted(self):
        broken = record("in_scope/space/still", ["a\u00a0\u2026"])
        report = build_report([broken], LEXICON)
        self.assertEqual((report.degenerate, report.parsed), (1, 0))

    def test_format_marks_failed_criteria(self):
        lines = format_report(build_report(RECORDS, LEXICON))
        self.assertTrue(any(line.startswith("FAIL  responses that parse") for line in lines))

    def test_hand_check_sample_is_seeded(self):
        first = hand_check_sample(RECORDS, seed=5, size=3)
        self.assertEqual(first, hand_check_sample(RECORDS, seed=5, size=3))
        self.assertEqual(len(first), 3)


if __name__ == "__main__":
    unittest.main()
