import json
import tempfile
import unittest
from pathlib import Path

from prompt_build.client import HttpResult
from prompt_build.config import load_config
from prompt_build.generate import Generation, RunLog
from prompt_build.lexicon import load_lexicon
from prompt_build.plan import ExampleLists, ExampleSchedule, Templates
from prompt_build.pool import load_candidates
from prompt_build.refill import MAX_BATCHES, refill_stratum
from prompt_build.strata import build_strata

ROOT = Path(__file__).resolve().parent.parent
CONFIG = load_config(ROOT / "config.toml")
STRATA = build_strata(CONFIG)
LEXICON = load_lexicon(ROOT / "terms")
EXAMPLES = ExampleLists.load(ROOT / "lists")
STRATUM = next(s for s in STRATA if s.distractor and not s.motion and s.kind == "animal")
GOOD = {"base": "a quiet desk with a brass lamp", "with_element": "a quiet desk with a cat and a brass lamp"}
BAD = {"base": "a quiet desk with a brass lamp", "with_element": "a quiet desk with a cat"}


def response(pair: dict) -> HttpResult:
    content = json.dumps({"pairs": [pair] * 20})
    return HttpResult(200, {"choices": [{"message": {"content": content}, "finish_reason": "stop"}]}, None, 1)


class RefillTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.log = RunLog(Path(self.directory.name) / "requests.jsonl")

    def tearDown(self):
        self.directory.cleanup()

    def generation(self, outcomes: list) -> Generation:
        return Generation(
            run_name="refill-test",
            pilot_round=None,
            seed_block=1,
            base_seed=CONFIG.base_seed,
            settings=CONFIG.generator,
            templates=Templates.load(ROOT / "templates"),
            examples=EXAMPLES,
            schedule=ExampleSchedule.build(STRATA, EXAMPLES, 1, 20),
            log=self.log,
            complete=lambda payload: outcomes.pop(0),
        )

    def test_continues_request_indices_and_stops_when_met(self):
        earlier = [
            {
                "stratum": STRATUM.key,
                "stratum_index": STRATUM.index,
                "request_index": 0,
                "status": "ok",
                "items": [],
            }
        ]
        outcome = refill_stratum(self.generation([response(GOOD)]), STRATUM, 3, earlier, [], LEXICON)
        self.assertTrue(outcome.met)
        self.assertEqual(outcome.batches, 1)
        self.assertEqual([r["request_index"] for r in self.log.read()], [1])

    def test_gives_up_after_five_batches(self):
        outcomes = [response(BAD) for _ in range(MAX_BATCHES)]
        outcome = refill_stratum(self.generation(outcomes), STRATUM, 3, [], [], LEXICON)
        self.assertFalse(outcome.met)
        self.assertEqual(outcome.batches, MAX_BATCHES)

    def test_earlier_refill_batches_count_towards_the_limit(self):
        earlier = [
            {"stratum": STRATUM.key, "stratum_index": STRATUM.index, "request_index": i,
             "status": "malformed", "items": []}
            for i in range(MAX_BATCHES - 1)
        ]
        outcome = refill_stratum(self.generation([response(BAD)]), STRATUM, 3, earlier, earlier, LEXICON)
        self.assertEqual(outcome.batches, MAX_BATCHES)
        self.assertEqual([r["request_index"] for r in self.log.read()], [MAX_BATCHES - 1])


class DuplicateRequestTest(unittest.TestCase):
    def test_same_request_in_two_logs_raises(self):
        record = {"stratum": STRATUM.key, "request_index": 0, "status": "ok", "items": []}
        with self.assertRaises(ValueError):
            load_candidates([record, dict(record)], {STRATUM.key: STRATUM}, LEXICON)


if __name__ == "__main__":
    unittest.main()
