import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from prompt_build.config import load_config
from prompt_build.late_reserves import screen_new
from prompt_build.lexicon import load_lexicon
from prompt_build.pool import Candidate
from prompt_build.strata import build_strata

ROOT = Path(__file__).resolve().parent.parent
CONFIG = load_config(ROOT / "config.toml")
STRATUM = next(s for s in build_strata(CONFIG) if s.distractor and s.kind == "animal" and not s.motion)
LEXICON = load_lexicon(ROOT / "terms")
STOP = frozenset({"a", "with", "and"})


def pair(base: str, element: str) -> dict:
    return {"base": base, "with_element": f"{base} and a {element} nearby"}


def fake_encode(texts, *args):
    vectors = {}
    for i, text in enumerate(sorted(set(texts))):
        v = np.zeros(64)
        v[i % 64] = 1.0
        vectors[text] = v
    return vectors


class ScreenTest(unittest.TestCase):
    def test_new_candidates_are_numbered_after_the_pool_and_screened(self):
        pool = [Candidate(i, STRATUM, 0, f"p{i}", f"a quiet desk number {i}", f"e{i}", False) for i in range(40)]
        pool.append(Candidate(40, STRATUM, 0, "x", "a brass lamp on a quiet desk", "cat", False))
        records = [{"stratum": STRATUM.key, "request_index": 9, "status": "ok", "items": [
            pair("a brass lamp on a quiet desk", "fox"),
            pair("a tin of paint beside a folded tarp", "owl"),
        ]}]
        with tempfile.TemporaryDirectory() as directory, mock.patch(
            "prompt_build.late_reserves.encode", side_effect=fake_encode
        ):
            accepted = screen_new(records, STRATUM, pool, CONFIG, LEXICON, STOP, frozenset(), Path(directory))
        self.assertEqual([c.candidate_id for c in accepted], [42])
        self.assertEqual(accepted[0].excluded_element, "owl")


if __name__ == "__main__":
    unittest.main()
