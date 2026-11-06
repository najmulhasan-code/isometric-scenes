import json
import tempfile
import unittest
from pathlib import Path

from prompt_build.config import SPLITS, load_config
from prompt_build.pool import Candidate
from prompt_build.release import KEY_ORDER, record, write_release
from prompt_build.splits import Assignment
from prompt_build.strata import build_strata

ROOT = Path(__file__).resolve().parent.parent
CONFIG = load_config(ROOT / "config.toml")
STRATA = build_strata(CONFIG)
DISTRACTOR = next(s for s in STRATA if s.distractor)


class ReleaseTest(unittest.TestCase):
    def test_record_has_the_spec_keys_in_order(self):
        candidate = Candidate(7, DISTRACTOR, 0, "a desk and a cat nearby", "a desk", "cat", False)
        row = record(candidate, "train", 3, "9.9")
        self.assertEqual(tuple(row), KEY_ORDER)
        self.assertEqual((row["id"], row["excluded_element_type"], row["distractor"]), ("train-0003", DISTRACTOR.kind, True))

    def test_files_are_sorted_lf_and_without_bom(self):
        assignment = Assignment()
        for split in SPLITS:
            for cid in (5, 2):
                assignment.splits[split].append(Candidate(cid, STRATA[0], 0, f"p{cid}", f"p{cid}", None, False))
        with tempfile.TemporaryDirectory() as directory:
            digests = write_release(assignment, CONFIG, Path(directory))
            data = (Path(directory) / "train.jsonl").read_bytes()
        self.assertEqual(set(digests), {f"{s}.jsonl" for s in SPLITS})
        self.assertFalse(data.startswith(b"\xef\xbb\xbf"))
        self.assertNotIn(b"\r", data)
        rows = [json.loads(line) for line in data.decode("utf-8").splitlines()]
        self.assertEqual([r["candidate_id"] for r in rows], [2, 5])
        self.assertEqual([r["idx"] for r in rows], [0, 1])


if __name__ == "__main__":
    unittest.main()
