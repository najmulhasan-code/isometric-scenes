import contextlib
import io
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from prompt_build.__main__ import parse_marks, run_review_command, run_threshold_check
from prompt_build.review import Review
from tests.test_review import SMALL, write_build


def run(command, root: Path, **fields) -> tuple[int, str]:
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        code = command(SMALL, root, Namespace(**fields))
    return code, out.getvalue()


class ReviewCommandTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.build = self.root / "builds" / "b"
        self.build.mkdir(parents=True)
        write_build(self.build)

    def tearDown(self):
        self.directory.cleanup()

    def step(self, step: str, **fields) -> tuple[int, str]:
        return run(run_review_command, self.root, build="b", step=step, **fields)

    def test_a_round_runs_end_to_end(self):
        self.assertEqual(self.step("start")[0], 0)
        self.assertEqual(self.step("status")[0], 0)
        self.assertEqual(self.step("show")[0], 0)
        self.assertEqual(len((self.build / "review" / "round1.txt").read_text(encoding="utf-8").splitlines()), 9)
        Review(self.build, SMALL).flag({1: "R4"})
        self.assertEqual(self.step("sample")[0], 0)
        with self.assertRaises(ValueError):
            Review(self.build, SMALL).flag({2: "R4"})
        self.assertEqual(self.step("sheet")[0], 0)
        sheet = (self.build / "review" / "author_sheet.txt").read_text(encoding="utf-8")
        self.assertTrue(sheet.endswith("]\n"))
        self.assertEqual(self.step("decide", remove=[])[0], 0)
        code, out = self.step("close")
        self.assertEqual(code, 0)
        code, out = self.step("status")
        self.assertEqual(json.loads(out[out.index("{"):])["rounds"][0]["removed"], 1)

    def test_errors_are_reported_not_raised(self):
        self.step("start")
        code, out = self.step("close")
        self.assertEqual(code, 1)
        self.assertIn("author", out)

    def test_marks_must_be_well_formed(self):
        self.assertEqual(parse_marks(["3=R8"]), {3: "R8"})
        for bad in (["3"], ["x=R8"], ["3=R8", "3=R4"]):
            with self.assertRaises(ValueError):
                parse_marks(bad)


class ThresholdCheckCommandTest(unittest.TestCase):
    def test_rule_uses_the_build_threshold(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            build = root / "builds" / "t"
            build.mkdir(parents=True)
            key = [{"number": n, "method": "embedding", "side": "kept", "score": 0.7} for n in range(1, 6)]
            (build / "threshold_check_key.json").write_text(json.dumps(key), encoding="utf-8")
            (build / "summary.json").write_text(json.dumps({"cosine_threshold": 0.85}), encoding="utf-8")
            code, _ = run(run_threshold_check, root, build="t", same=[1, 2])
            result = json.loads((build / "threshold_check_result.json").read_text(encoding="utf-8"))
            self.assertEqual((code, result["threshold"]), (0, 0.80))
            self.assertEqual(run(run_threshold_check, root, build="t", same=[9])[0], 1)


if __name__ == "__main__":
    unittest.main()
