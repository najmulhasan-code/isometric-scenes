import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from prompt_build.client import HttpResult
from prompt_build.generate import RunLog
from prompt_build.prescreen import log_path, prescreen_round, recorded_round
from prompt_build.public import export_public, late_reserve_events, scan, scrub, scrub_error, scrub_record
from prompt_build.review import Review
from tests.test_prescreen import CATEGORIES, TEMPLATE, reply
from tests.test_review import CONFIG, SMALL, write_build

RESPONSE = {
    "id": "abc123-IAD",
    "object": "chat.completion",
    "model": "m",
    "system_fingerprint": "fp",
    "choices": [{"message": {"content": "{}", "reasoning_details": [{"text": "t", "signature": "s"}]}}],
}
ERROR = ('HTTP 402: {"error":{"message":"no funds","code":"x"},'
         '"request_id":"r-1","inference_id":"req_1"}')


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data) + "\n", encoding="utf-8")


def write_records(path: Path, records: list[dict]) -> None:
    log = RunLog(path)
    for record in records:
        log.append(record)


class ScrubTest(unittest.TestCase):
    def test_private_fields_are_removed_at_any_depth(self):
        removed = Counter()
        clean = scrub(RESPONSE, removed)
        self.assertEqual(removed, Counter({"id": 1, "system_fingerprint": 1, "signature": 1}))
        self.assertEqual(list(clean), ["object", "model", "choices"])
        self.assertEqual(clean["choices"][0]["message"]["reasoning_details"], [{"text": "t"}])
        self.assertIn("id", RESPONSE)

    def test_error_bodies_lose_their_request_ids(self):
        removed = Counter()
        text = scrub_error(ERROR, removed)
        self.assertEqual(text, 'HTTP 402: {"error":{"message":"no funds","code":"x"}}')
        self.assertEqual(removed, Counter({"request_id": 1, "inference_id": 1}))

    def test_a_cut_off_error_body_is_cleaned_too(self):
        removed = Counter()
        text = scrub_error(ERROR[:-10], removed)
        self.assertNotIn("request_id", text)
        self.assertEqual(removed["request_id"], 1)
        self.assertEqual(scrub_error("network error: timeout", removed), "network error: timeout")

    def test_record_fields_other_than_response_and_error_are_kept(self):
        record = {"run": "main", "items": ["a"], "response": RESPONSE, "error": None, "id": "kept"}
        clean = scrub_record(record, Counter())
        self.assertEqual(clean["id"], "kept")
        self.assertEqual(clean["items"], ["a"])
        self.assertNotIn("id", clean["response"])


class ExportTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name) / "data"
        self.out = Path(self.directory.name) / "public"
        for run in ("main", "refill-1", "refill-2"):
            write_json(self.root / "runs" / run / "manifest.json", {"model": "m", "pilot_round": None})
            write_records(self.root / "runs" / run / "requests.jsonl",
                          [{"run": run, "items": ["a"], "response": RESPONSE, "error": None}])
        write_records(self.root / "runs" / "pilot-1" / "requests.jsonl", [{"run": "pilot-1"}])
        build = self.root / "builds" / "main"
        write_json(build / "review" / "state.json",
                   {"rounds": [{"author": {"removed": {"3": "R8"}}}, {"author": {"removed": {}}}]})
        for number in (1, 2):
            write_records(build / "review" / f"prescreen_round{number}.jsonl",
                          [{"number": 1, "response": None, "error": ERROR}])
        late = [{"run": "refill-2", "stratum": "s", "candidate_id": n} for n in (10, 11)]
        (build / "review" / "late_candidates.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in late), encoding="utf-8")
        (build / "review" / "round1.txt").write_text("reading list\n", encoding="utf-8")
        write_json(build / "release" / "release.json", {"inputs": {}})
        (build / "release" / "SHA256SUMS").write_text("00  train.jsonl\n", encoding="utf-8")
        check = self.root / "builds" / "threshold-check"
        write_json(check / "summary.json", {"cosine_threshold": 0.85})
        write_json(check / "threshold_check_result.json", {"judged_same": [2]})
        write_json(check / "threshold_check_key.json", [])

    def tearDown(self):
        self.directory.cleanup()

    def export(self):
        return export_public(self.root, self.out, pool_runs=["main", "refill-1"], threshold_runs=["main"])

    def test_only_the_rebuild_inputs_are_copied(self):
        manifest = self.export()
        self.assertEqual(sorted(manifest["files"]), [
            "expected/SHA256SUMS", "expected/release.json", "review/decisions.json",
            "review/prescreen_round1.jsonl", "review/prescreen_round2.jsonl",
            *(f"runs/{run}/{name}" for run in ("main", "refill-1", "refill-2")
              for name in ("manifest.json", "requests.jsonl")),
        ])
        self.assertEqual(json.loads((self.out / "MANIFEST.json").read_text(encoding="utf-8")), manifest)

    def test_logs_lose_private_fields_and_keep_the_rest(self):
        manifest = self.export()
        record = RunLog(self.out / "runs" / "main" / "requests.jsonl").read()[0]
        self.assertEqual(record["items"], ["a"])
        self.assertEqual(record["response"]["choices"][0]["message"]["content"], "{}")
        for path in self.out.rglob("*.jsonl"):
            text = path.read_text(encoding="utf-8")
            for value in ("abc123-IAD", '"fp"', "req_1", "r-1", '"signature"'):
                self.assertNotIn(value, text)
        removed = {f["field"]: f["count"] for f in manifest["removed_fields"]}
        self.assertEqual(removed, {"id": 3, "system_fingerprint": 3, "signature": 3,
                                   "request_id": 2, "inference_id": 2})
        self.assertEqual(scan(self.out), [])

    def test_decisions_hold_the_persons_choices(self):
        self.export()
        decisions = json.loads((self.out / "review" / "decisions.json").read_text(encoding="utf-8"))
        self.assertEqual(decisions["spot_check_removals"], [{"3": "R8"}, {}])
        self.assertEqual(decisions["late_reserves"], [{"run": "refill-2", "stratum": "s"}])
        self.assertEqual(decisions["threshold_check"]["judged_same"], [2])
        self.assertEqual(decisions["threshold_check"]["cosine_threshold"], 0.85)

    def test_a_folder_in_use_is_refused(self):
        self.out.mkdir()
        (self.out / "x").write_text("x", encoding="utf-8")
        with self.assertRaises(ValueError):
            self.export()

    def test_scan_reports_private_text(self):
        self.out.mkdir()
        (self.out / "a.txt").write_text(
            "ok\nwrite to someone@example.com\nC:\\Users\\someone\\file\n"
            '{"team_id": "t-123456"}\n' + str(Path.home()) + "\n", encoding="utf-8")
        found = {hit.split(": ")[1] for hit in scan(self.out)}
        self.assertTrue({"email", "windows path", "private field", "home folder"} <= found)

    def test_late_reserve_events_keep_their_order(self):
        build = Path(self.directory.name) / "b"
        rows = [("r4", "x"), ("r4", "x"), ("r5", "y"), ("r6", "x")]
        (build / "review").mkdir(parents=True)
        (build / "review" / "late_candidates.jsonl").write_text(
            "".join(json.dumps({"run": r, "stratum": s}) + "\n" for r, s in rows), encoding="utf-8")
        self.assertEqual(late_reserve_events(build), [
            {"run": "r4", "stratum": "x"}, {"run": "r5", "stratum": "y"}, {"run": "r6", "stratum": "x"}])


class RecordedRoundTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name)
        write_build(self.path)
        self.review = Review(self.path, SMALL)
        self.review.start()
        self.flagged = self.review.current["items"][0]["number"]
        first = self.review.rows[self.review.current["items"][0]["candidate_id"]].base_prompt

        def complete(payload):
            flag = first in payload["messages"][0]["content"]
            content = '{"verdict": "flag", "rule": "R8", "reason": "r"}' if flag else '{"verdict": "keep"}'
            return HttpResult(200, {"id": "x", **reply(content)}, None, 1)

        self.outcome = prescreen_round(self.review, CONFIG.prescreen, TEMPLATE, CATEGORIES, complete)
        self.records = RunLog(log_path(self.review)).read()

    def tearDown(self):
        self.directory.cleanup()

    def replay(self, records):
        return recorded_round(self.review, CONFIG.prescreen, TEMPLATE, CATEGORIES, records)

    def test_a_scrubbed_log_gives_the_same_verdicts(self):
        clean = [scrub_record(r, Counter()) for r in self.records]
        self.assertEqual(self.replay(clean), self.outcome.flags)
        self.assertEqual({n for n, v in self.replay(clean).items() if v.flag}, {self.flagged})

    def test_a_log_missing_an_item_gives_nothing(self):
        self.assertIsNone(self.replay(self.records[1:]))


if __name__ == "__main__":
    unittest.main()
