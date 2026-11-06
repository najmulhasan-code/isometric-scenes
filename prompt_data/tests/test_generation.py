import http.client
import io
import json
import os
import random
import tempfile
import unittest
import urllib.error
from collections import Counter
from pathlib import Path

from prompt_build.__main__ import load_env_file, parse_env_value
from prompt_build.client import ChatClient, HttpResult
from prompt_build.config import SPLITS, load_config
from prompt_build.generate import (
    Generation,
    RunLog,
    fill_stratum,
    parse_response,
    pilot_strata,
    run_main,
    run_pilot,
)
from prompt_build.manifest import build_manifest, check_or_write
from prompt_build.plan import (
    ExampleLists,
    ExampleSchedule,
    Templates,
    build_request,
    exposure,
    request_seed,
    seed_block,
)
from prompt_build.strata import build_strata, distractor_slots, spread_labels

ROOT = Path(__file__).resolve().parent.parent
CONFIG = load_config(ROOT / "config.toml")
STRATA = build_strata(CONFIG)
TEMPLATES = Templates.load(ROOT / "templates")
EXAMPLES = ExampleLists.load(ROOT / "lists")
PER_REQUEST = CONFIG.generator.items_per_request
SCHEDULE = ExampleSchedule.build(STRATA, EXAMPLES, 99, PER_REQUEST)
PILOT = pilot_strata(STRATA, CONFIG.split_categories("train"))
PAIR = {"base": "a desk", "with_element": "a desk with a cat"}


def completion(content: str, finish_reason: str = "stop") -> dict:
    return {"choices": [{"message": {"content": content}, "finish_reason": finish_reason}]}


def pairs(count: int) -> dict:
    return completion(json.dumps({"pairs": [PAIR] * count}))


class StrataTest(unittest.TestCase):
    def test_quotas_match_the_spec(self):
        in_scope = sum(s.quota for s in STRATA if not s.distractor)
        distractors = sum(s.quota for s in STRATA if s.distractor)
        self.assertEqual((in_scope, distractors), (1820, 228))

    def test_slot_table_meets_every_split_total(self):
        slots = distractor_slots(CONFIG)
        for split in SPLITS:
            quota = CONFIG.splits[split]
            mine = [slot for slot in slots if slot.split == split]
            with self.subTest(split=split):
                self.assertEqual(Counter(s.kind for s in mine), Counter(quota.distractor_kinds))
                self.assertEqual(Counter(s.motion for s in mine), Counter(quota.distractor_motion))
                per_category = Counter(s.category.slug for s in mine)
                self.assertEqual(len(set(per_category.values())), 1)

    def test_kinds_are_spread_within_categories(self):
        slots = [s for s in distractor_slots(CONFIG) if s.split == "test"]
        for category in CONFIG.split_categories("test"):
            kinds = Counter(s.kind for s in slots if s.category == category)
            with self.subTest(category=category.slug):
                self.assertEqual(kinds, Counter({"human": 2, "animal": 2, "face_figure": 1}))

    def test_distractor_strata_are_pooled_across_splits(self):
        slots = distractor_slots(CONFIG)
        for stratum in (s for s in STRATA if s.distractor):
            matching = [
                slot
                for slot in slots
                if (slot.category, slot.kind, slot.motion)
                == (stratum.category, stratum.kind, stratum.motion)
            ]
            self.assertEqual(stratum.quota, len(matching))

    def test_uneven_spread_is_rejected(self):
        categories = CONFIG.split_categories("train")
        with self.assertRaises(ValueError):
            spread_labels(("human",), {"human": len(categories) + 1}, categories, random.Random(0))

    def test_strata_are_deterministic(self):
        self.assertEqual(build_strata(CONFIG), STRATA)
        self.assertEqual([s.index for s in STRATA], list(range(len(STRATA))))


class RequestTest(unittest.TestCase):
    def test_seed_blocks(self):
        self.assertEqual(seed_block(None), 1)
        self.assertEqual(seed_block(1), 2)
        self.assertEqual(seed_block(8), 9)
        self.assertEqual(seed_block(None, diagnostic=True), 10)
        for bad in (0, 9):
            with self.assertRaises(ValueError):
                seed_block(bad)

    def test_seed_ranges(self):
        self.assertEqual(request_seed(7, 3, 4, 1), 7 + 1_000_000 + 3_004)
        self.assertEqual(request_seed(7, 3, 4, 2), 7 + 2_000_000 + 3_004)
        with self.assertRaises(ValueError):
            request_seed(7, 1000, 0, 1)

    def test_every_stratum_fills_its_template(self):
        for stratum in STRATA:
            spec = build_request(stratum, 0, 20, 123, TEMPLATES, EXAMPLES, SCHEDULE)
            with self.subTest(stratum=stratum.key):
                self.assertNotIn("$", spec.text)
                self.assertIn(stratum.category.definition, spec.text)

    def test_focus_rotates_with_request_index(self):
        stratum = STRATA[0]
        focus = stratum.category.focus
        for j in range(3):
            text = build_request(stratum, j, 20, 1, TEMPLATES, EXAMPLES, SCHEDULE).text
            self.assertIn(f"Focus for this batch: {focus[(stratum.index + j) % len(focus)]}.", text)

    def test_insertion_example_uses_a_frame_and_the_stratum_kind(self):
        for stratum in (s for s in STRATA if s.distractor):
            text = build_request(stratum, 0, 20, stratum.index, TEMPLATES, EXAMPLES, SCHEDULE).text
            example = text.split('as in "', 1)[1].split('"', 1)[0]
            with self.subTest(stratum=stratum.key):
                self.assertTrue(
                    any(example.startswith(f.split("{element}")[0]) for f in EXAMPLES.insertion_frames)
                )
                self.assertTrue(any(e in example for e in EXAMPLES.elements[stratum.kind]))

    def test_schedule_balances_examples(self):
        shown = exposure(SCHEDULE, STRATA, PER_REQUEST)
        for name, counts in shown.items():
            with self.subTest(name=name):
                self.assertEqual(len(counts), len(SCHEDULE.orders[name]))
                self.assertLessEqual(max(counts.values()) / sum(counts.values()), 0.10)

    def test_schedule_draws_three_distinct_examples(self):
        for stratum in (s for s in STRATA if s.distractor):
            for request_index in range(3):
                drawn = SCHEDULE.draw(stratum.kind, stratum.index, request_index)
                self.assertEqual(len(set(drawn)), 3)

    def test_examples_follow_the_seed(self):
        stratum = next(s for s in STRATA if s.distractor and s.motion)
        first = build_request(stratum, 0, 20, 99, TEMPLATES, EXAMPLES, SCHEDULE).text
        self.assertEqual(first, build_request(stratum, 0, 20, 99, TEMPLATES, EXAMPLES, SCHEDULE).text)
        self.assertIn("apart from the added phrase", first)


class ParseTest(unittest.TestCase):
    def test_prompts(self):
        parsed = parse_response(completion('{"prompts": ["a lamp", "a desk"]}'), False, 2)
        self.assertEqual((parsed.status, parsed.items), ("ok", ["a lamp", "a desk"]))

    def test_degenerate_responses(self):
        few = parse_response(completion('{"prompts": ["a lamp", "a desk"]}'), False, 5)
        self.assertEqual((few.status, few.items), ("degenerate", ["a lamp", "a desk"]))
        broken = parse_response(completion('{"prompts": ["a\u00a0\u2026"]}'), False, 1)
        self.assertEqual(broken.status, "degenerate")
        accented = parse_response(completion('{"prompts": ["a small caf\u00e9 counter"]}'), False, 1)
        self.assertEqual(accented.status, "ok")

    def test_one_code_fence_is_removed(self):
        parsed = parse_response(completion('```json\n{"prompts": ["a lamp"]}\n```'), False, 5)
        self.assertEqual(parsed.items, ["a lamp"])

    def test_extra_items_are_dropped(self):
        parsed = parse_response(completion('{"prompts": ["a", "b", "c"]}'), False, 2)
        self.assertEqual(parsed.items, ["a", "b"])

    def test_curly_quotes_are_straightened(self):
        body = completion('{"prompts": ["a door with an \u201cOPEN\u201d sign"]}')
        self.assertEqual(parse_response(body, False, 1).items, ['a door with an "OPEN" sign'])

    def test_pairs(self):
        self.assertEqual(parse_response(pairs(1), True, 1).items, [PAIR])

    def test_length_is_reported_even_without_content(self):
        body = {"choices": [{"message": {}, "finish_reason": "length"}]}
        self.assertEqual(parse_response(body, False, 5).status, "length")

    def test_failures(self):
        cases = {
            "length": (completion('{"prompts": ["a lamp"', "length"), False),
            "not json": (completion("Here are some prompts"), False),
            "wrong key": (completion('{"items": ["a lamp"]}'), False),
            "bad pair": (completion('{"pairs": [{"base": "a desk"}]}'), True),
            "no choices": ({}, False),
            "no body": (None, False),
        }
        for name, (body, distractor) in cases.items():
            with self.subTest(name):
                parsed = parse_response(body, distractor, 5)
                self.assertIsNone(parsed.items)
                self.assertNotEqual(parsed.status, "ok")


class FakeResponse:
    def __init__(self, body: dict, status: int = 200) -> None:
        self._raw = json.dumps(body).encode("utf-8")
        self.status = status

    def read(self) -> bytes:
        return self._raw

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def http_error(code: int, headers: dict | None = None) -> urllib.error.HTTPError:
    return urllib.error.HTTPError("url", code, "error", headers or {}, io.BytesIO(b"server said no"))


class ClientTest(unittest.TestCase):
    def make_client(self, outcomes: list):
        waits = []

        def opener(request, timeout):
            self.assertEqual(request.get_header("Authorization"), "Bearer secret")
            self.assertEqual(request.get_header("User-agent"), "prompt-build/0.1.0")
            outcome = outcomes.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return FakeResponse(outcome)

        client = ChatClient(
            "https://example.test/v1/", "secret", 5, sleep=waits.append, opener=opener
        )
        return client, waits

    def test_retries_server_errors(self):
        client, waits = self.make_client([http_error(503), {"ok": True}])
        result = client.complete({})
        self.assertEqual((result.status, result.body, result.attempts), (200, {"ok": True}, 2))
        self.assertEqual(waits, [2.0])

    def test_honours_retry_after(self):
        client, waits = self.make_client([http_error(429, {"Retry-After": "30"}), {"ok": True}])
        client.complete({})
        self.assertEqual(waits, [30.0])

    def test_does_not_retry_client_errors(self):
        client, waits = self.make_client([http_error(400)])
        result = client.complete({})
        self.assertEqual((result.status, result.attempts, waits), (400, 1, []))
        self.assertIn("server said no", result.error)

    def test_incomplete_read_is_a_network_error(self):
        client, _ = self.make_client([http.client.IncompleteRead(b"par"), {"ok": True}])
        self.assertEqual(client.complete({}).attempts, 2)

    def test_gives_up_after_three_retries(self):
        client, waits = self.make_client([urllib.error.URLError("down")] * 4)
        result = client.complete({})
        self.assertEqual((result.status, result.attempts), (None, 4))
        self.assertEqual(waits, [2.0, 4.0, 8.0])

    def test_requires_a_key(self):
        with self.assertRaises(ValueError):
            ChatClient("https://example.test/v1", "", 5)


class FillStratumTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.log = RunLog(Path(self.directory.name) / "requests.jsonl")
        self.stratum = next(s for s in STRATA if s.distractor)
        self.schedule = SCHEDULE

    def tearDown(self):
        self.directory.cleanup()

    def generation(self, outcomes: list) -> Generation:
        def complete(payload):
            outcome = outcomes.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            if isinstance(outcome, HttpResult):
                return outcome
            return HttpResult(200, outcome, None, 1)

        return Generation(
            run_name="test",
            pilot_round=None,
            seed_block=1,
            base_seed=CONFIG.base_seed,
            settings=CONFIG.generator,
            templates=TEMPLATES,
            examples=EXAMPLES,
            schedule=self.schedule,
            log=self.log,
            complete=complete,
        )

    def test_fills_target_and_resumes_without_resending(self):
        half = self.stratum.target // 2 + 1
        fill_stratum(self.generation([pairs(half), pairs(half)]), self.stratum, self.log.read())
        records = self.log.read()
        self.assertEqual([r["request_index"] for r in records], [0, 1])
        self.assertEqual({r["size"] for r in records}, {CONFIG.generator.items_per_request})

        fill_stratum(self.generation([]), self.stratum, self.log.read())
        self.assertEqual(len(self.log.read()), 2)

    def test_stops_after_repeated_failures(self):
        outcome = fill_stratum(self.generation([completion("not json")] * 5), self.stratum, [])
        self.assertIn("stopped", outcome)
        self.assertEqual(len(self.log.read()), 5)

    def test_rejected_request_aborts_every_stratum(self):
        generation = self.generation([HttpResult(400, None, "HTTP 400: bad parameter", 1)])
        lines = run_main(generation, [self.stratum, STRATA[0]])
        self.assertTrue(generation.abort.is_set())
        self.assertEqual(len(self.log.read()), 1)
        self.assertTrue(any("aborted" in line for line in lines))

    def test_pilot_retries_failed_strata_with_the_next_index(self):
        stratum = PILOT[0]
        self.schedule = ExampleSchedule.build(PILOT, EXAMPLES, 99, PER_REQUEST)
        failed = HttpResult(503, None, "HTTP 503: unavailable", 4)
        run_pilot(self.generation([failed]), [stratum])
        good = completion(json.dumps({"prompts": ["a lamp on a desk"] * 20}))
        run_pilot(self.generation([good]), [stratum])
        run_pilot(self.generation([]), [stratum])
        records = self.log.read()
        self.assertEqual([r["request_index"] for r in records], [0, 1])
        self.assertEqual([r["status"] for r in records], ["http_error", "ok"])

    def test_exception_is_reported_per_stratum(self):
        outcome = fill_stratum(self.generation([RuntimeError("boom")]), self.stratum, [])
        self.assertIn("failed with RuntimeError: boom", outcome)

    def test_log_holds_request_text_but_no_key(self):
        fill_stratum(self.generation([completion("not json")] * 5), self.stratum, [])
        record = self.log.read()[0]
        self.assertIn(f"Write {CONFIG.generator.items_per_request} pairs", record["request_text"])
        self.assertNotIn("Authorization", json.dumps(record))

    def test_cut_off_last_line_is_skipped(self):
        self.log.append({"stratum_index": 0, "request_index": 0, "status": "ok", "items": []})
        with self.log.path.open("a", encoding="utf-8") as handle:
            handle.write('{"stratum_index": 0, "requ')
        self.assertEqual(len(self.log.read()), 1)

    def test_append_after_cut_off_line_keeps_the_new_record(self):
        self.log.append({"a": 1})
        with self.log.path.open("a", encoding="utf-8") as handle:
            handle.write('{"b": ')
        self.log.append({"c": 3})
        self.assertEqual(self.log.read(), [{"a": 1}, {"c": 3}])

    def test_broken_line_in_the_middle_raises(self):
        self.log.path.write_text('{"a": 1\n{"b": 2}\n', encoding="utf-8")
        with self.assertRaises(ValueError):
            self.log.read()


class ManifestTest(unittest.TestCase):
    def test_resume_is_refused_when_inputs_change(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            manifest = build_manifest(ROOT, "model-a", 1)
            self.assertEqual(check_or_write(path, manifest), [])
            self.assertEqual(check_or_write(path, manifest), [])

            changed = dict(manifest, files=dict(manifest["files"], **{"config.toml": "x"}))
            self.assertIn("config.toml", check_or_write(path, changed)[0])
            self.assertTrue(check_or_write(path, dict(manifest, pilot_round=None)))

    def test_manifest_hashes_every_input_file(self):
        files = build_manifest(ROOT, "model-a", None)["files"]
        self.assertIn("config.toml", files)
        self.assertIn("templates/in_scope.txt", files)
        self.assertIn("terms/out_of_scope.tsv", files)


class PilotAndEnvTest(unittest.TestCase):
    def test_pilot_strata(self):
        pilot = pilot_strata(STRATA, CONFIG.split_categories("train"))
        self.assertEqual(len(pilot), 24)
        self.assertEqual([s.index for s in pilot], list(range(24)))
        self.assertEqual(sum(s.distractor for s in pilot), 6)

    def test_env_values(self):
        self.assertEqual(parse_env_value('"quoted # value"'), "quoted # value")
        self.assertEqual(parse_env_value("plain # comment"), "plain")
        self.assertEqual(parse_env_value("\"mismatched'"), "\"mismatched'")
        self.assertEqual(parse_env_value('"abc" # note'), "abc")

    def test_env_file_does_not_override(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            path.write_text(
                'export PROMPT_TEST_A="from file"\nPROMPT_TEST_B=2\n# comment\n', encoding="utf-8"
            )
            os.environ["PROMPT_TEST_B"] = "already set"
            try:
                load_env_file(path)
                self.assertEqual(os.environ["PROMPT_TEST_A"], "from file")
                self.assertEqual(os.environ["PROMPT_TEST_B"], "already set")
            finally:
                os.environ.pop("PROMPT_TEST_A", None)
                os.environ.pop("PROMPT_TEST_B", None)


if __name__ == "__main__":
    unittest.main()
