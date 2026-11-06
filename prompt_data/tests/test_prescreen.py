import json
import tempfile
import threading
import unittest
from dataclasses import replace
from pathlib import Path
from string import Template

from prompt_build.client import HttpResult
from prompt_build.prescreen import log_path, parse_verdict, prescreen_round, verdicts
from prompt_build.review import Review
from tests.test_review import CONFIG, SMALL, write_build

ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = Template((ROOT / "templates" / "prescreen.txt").read_text(encoding="utf-8"))
CATEGORIES = CONFIG.categories


def reply(content: str) -> dict:
    return {"choices": [{"message": {"content": content}}]}


class ParseTest(unittest.TestCase):
    def test_keep_and_flag(self):
        self.assertIsNone(parse_verdict(reply('{"verdict": "keep"}')).flag)
        verdict = parse_verdict(reply('Answer: {"verdict": "flag", "rule": "R4", "reason": "five objects"}'))
        self.assertEqual((verdict.flag, verdict.reason), ("R4", "five objects"))

    def test_bad_replies_give_no_verdict(self):
        for body in (None, {}, reply("keep"), reply('{"verdict": "flag", "rule": "R99"}'),
                     reply('{"verdict": "maybe"}'), reply("[1, 2]")):
            self.assertIsNone(parse_verdict(body))


class PrescreenTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name)
        write_build(self.path)
        self.review = Review(self.path, SMALL)
        self.review.start()
        self.flagged_prompt = self.review.rows[self.review.current["items"][0]["candidate_id"]].base_prompt
        self.lock = threading.Lock()
        self.calls = 0

    def tearDown(self):
        self.directory.cleanup()

    def complete(self, payload: dict, broken: set[str] = frozenset()) -> HttpResult:
        with self.lock:
            self.calls += 1
        text = payload["messages"][0]["content"]
        prompt = text.split("The prompt:\n", 1)[1].split("\n", 1)[0]
        if prompt in broken:
            return HttpResult(200, reply("not json"), None, 1)
        if prompt == self.flagged_prompt:
            return HttpResult(200, reply('{"verdict": "flag", "rule": "R8", "reason": "a person"}'), None, 1)
        return HttpResult(200, reply('{"verdict": "keep"}'), None, 1)

    def test_flags_are_returned_when_every_item_has_a_verdict(self):
        outcome = prescreen_round(self.review, CONFIG.prescreen, TEMPLATE, CATEGORIES, self.complete)
        self.assertEqual(outcome.sent, 9)
        flags = {n: v.flag for n, v in outcome.flags.items() if v.flag}
        self.assertEqual(flags, {self.review.current["items"][0]["number"]: "R8"})
        records = [json.loads(line) for line in log_path(self.review).read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(records), 9)
        self.assertEqual(records[0]["payload"]["temperature"], CONFIG.prescreen.temperature)

    def test_a_rerun_retries_only_missing_verdicts(self):
        broken = {self.flagged_prompt}
        outcome = prescreen_round(self.review, CONFIG.prescreen, TEMPLATE, CATEGORIES,
                                  lambda p: self.complete(p, broken))
        self.assertIsNone(outcome.flags)
        self.calls = 0
        outcome = prescreen_round(self.review, CONFIG.prescreen, TEMPLATE, CATEGORIES, self.complete)
        self.assertEqual((outcome.sent, self.calls), (1, 1))
        self.assertEqual(len(outcome.flags), 9)

    def test_a_client_error_stops_the_run(self):
        settings = replace(CONFIG.prescreen, max_workers=1)
        outcome = prescreen_round(self.review, settings, TEMPLATE, CATEGORIES,
                                  lambda p: HttpResult(400, None, "HTTP 400: bad request", 1))
        self.assertEqual(outcome.sent, 1)
        self.assertIsNone(outcome.flags)
        self.assertTrue(outcome.stopped)

    def test_a_cut_off_reply_stops_the_run(self):
        settings = replace(CONFIG.prescreen, max_workers=1)
        cut = {"choices": [{"message": {"content": None}, "finish_reason": "length"}]}
        outcome = prescreen_round(self.review, settings, TEMPLATE, CATEGORIES, lambda p: HttpResult(200, cut, None, 1))
        self.assertEqual(outcome.sent, 1)
        self.assertIn("max_tokens", outcome.stopped[0])

    def test_records_for_another_candidate_or_request_are_ignored(self):
        prescreen_round(self.review, CONFIG.prescreen, TEMPLATE, CATEGORIES, self.complete)
        records = [json.loads(line) for line in log_path(self.review).read_text(encoding="utf-8").splitlines()]
        expected = {r["number"]: (r["candidate_id"], r["payload_sha256"]) for r in records}
        number = records[0]["number"]
        self.assertIn(number, verdicts(records, expected))
        stale = dict(expected)
        stale[number] = (records[0]["candidate_id"], "other")
        self.assertNotIn(number, verdicts(records, stale))

    def test_the_first_valid_verdict_is_kept(self):
        record = {"number": 1, "candidate_id": 5, "payload_sha256": "h"}
        records = [dict(record, response=reply('{"verdict": "keep"}')),
                   dict(record, response=reply('{"verdict": "flag", "rule": "R3"}'))]
        self.assertIsNone(verdicts(records, {1: (5, "h")})[1].flag)

    def test_a_second_run_is_refused_while_one_holds_the_lock(self):
        lock = log_path(self.review).with_suffix(".lock")
        lock.parent.mkdir(parents=True, exist_ok=True)
        lock.write_text("")
        with self.assertRaises(ValueError):
            prescreen_round(self.review, CONFIG.prescreen, TEMPLATE, CATEGORIES, self.complete)

    def test_the_model_sees_no_split(self):
        seen = []

        def record(payload):
            seen.append(payload["messages"][0]["content"])
            return self.complete(payload)

        prescreen_round(self.review, CONFIG.prescreen, TEMPLATE, CATEGORIES, record)
        splits = {self.review.rows[i["candidate_id"]].split for i in self.review.current["items"]}
        for text in seen:
            self.assertNotIn("split", text.lower())
            for split in splits - {"train"}:
                self.assertNotIn(split, text)


if __name__ == "__main__":
    unittest.main()
