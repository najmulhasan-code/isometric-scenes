"""Candidate generation and the run log.

Every request is written to runs/<run>/requests.jsonl as one JSON object,
with the filled template, parameters, raw response and parsed items. The log
is the source of the candidate pool: candidates are read back from it and
numbered by stratum, request and item position, never by arrival order.

Every request asks for the same number of items. A stratum keeps asking
until it holds its target; items beyond the target are kept as reserves.
A stopped run resumes from its log and never sends a logged request again.
"""

import json
import os
import re
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .client import HttpResult
from .config import KIND_ORDER, Category, GeneratorConfig
from .plan import ExampleLists, ExampleSchedule, RequestSpec, Templates, build_request, request_seed
from .strata import Stratum
from .text import normalise_punctuation

MAX_CONSECUTIVE_FAILURES = 5
FATAL_STATUSES = frozenset({400, 401, 403, 404})
_FENCE = re.compile(r"\A```(?:json)?\s*\n(.*)\n```\s*\Z", re.DOTALL)
# Characters that do not occur in a well-formed prompt but fill broken responses:
# non-breaking and other unusual spaces, zero-width characters, the replacement
# character and the ellipsis.
_BROKEN_TEXT = re.compile("[\u00a0\u1680\u2000-\u200d\u2026\u202f\u205f\u2060\u3000\ufeff\ufffd]")


def is_degenerate(items: list, size: int) -> bool:
    """A response is degenerate when it holds fewer than half the requested items,
    or when any item contains characters typical of broken output."""
    if len(items) < size / 2:
        return True
    texts = (
        text
        for item in items
        for text in (item.values() if isinstance(item, dict) else (item,))
    )
    return any(_BROKEN_TEXT.search(text) for text in texts)


def _schema(distractor: bool) -> dict:
    if distractor:
        item = {
            "type": "object",
            "properties": {"base": {"type": "string"}, "with_element": {"type": "string"}},
            "required": ["base", "with_element"],
            "additionalProperties": False,
        }
        key = "pairs"
    else:
        item = {"type": "string"}
        key = "prompts"
    return {
        "type": "object",
        "properties": {key: {"type": "array", "items": item}},
        "required": [key],
        "additionalProperties": False,
    }


def build_payload(spec: RequestSpec, distractor: bool, settings: GeneratorConfig) -> dict:
    payload = {
        "model": settings.model,
        "messages": [{"role": "user", "content": spec.text}],
        "temperature": settings.temperature,
        "top_p": settings.top_p,
        "max_tokens": settings.max_tokens,
        "seed": spec.seed,
    }
    if settings.reasoning_effort:
        payload["reasoning_effort"] = settings.reasoning_effort
    if settings.use_json_schema:
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "scene_pairs" if distractor else "scene_prompts",
                "strict": True,
                "schema": _schema(distractor),
            },
        }
    return payload


@dataclass(frozen=True)
class Parsed:
    items: list | None
    status: str
    detail: str | None = None


def parse_response(body: dict | None, distractor: bool, size: int) -> Parsed:
    """Extract the items from a chat completion, or say why there are none."""
    try:
        choice = body["choices"][0]
    except (KeyError, IndexError, TypeError):
        return Parsed(None, "malformed", "no choices in the response")
    if not isinstance(choice, dict):
        return Parsed(None, "malformed", "the first choice is not an object")
    if choice.get("finish_reason") == "length":
        return Parsed(None, "length", "the response hit the token limit")
    content = (choice.get("message") or {}).get("content")
    if not isinstance(content, str):
        return Parsed(None, "malformed", "message content is missing or not text")

    text = content.strip()
    fenced = _FENCE.match(text)
    if fenced:
        text = fenced.group(1)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return Parsed(None, "malformed", "message content is not valid JSON")

    key = "pairs" if distractor else "prompts"
    if not isinstance(data, dict) or not isinstance(data.get(key), list):
        return Parsed(None, "malformed", f"expected an object with a {key!r} list")

    items = []
    for entry in data[key]:
        if distractor:
            if not (
                isinstance(entry, dict)
                and isinstance(entry.get("base"), str)
                and isinstance(entry.get("with_element"), str)
            ):
                return Parsed(None, "malformed", "a pair is not two strings")
            items.append(
                {
                    "base": normalise_punctuation(entry["base"]),
                    "with_element": normalise_punctuation(entry["with_element"]),
                }
            )
        else:
            if not isinstance(entry, str):
                return Parsed(None, "malformed", "a prompt is not a string")
            items.append(normalise_punctuation(entry))
    items = items[:size]
    if is_degenerate(items, size):
        return Parsed(items, "degenerate", "too few items, or broken characters in the items")
    return Parsed(items, "ok")


class RunLog:
    """Append-only JSONL log, safe to write from several threads.

    Each record is flushed to disk before the next request is sent. A last
    line cut off by a crash is skipped when the log is read back, and
    removed before the next record is written, so it cannot swallow it.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        path.parent.mkdir(parents=True, exist_ok=True)

    def read(self) -> list[dict]:
        if not self.path.exists():
            return []
        lines = [line for line in self.path.read_text(encoding="utf-8").split("\n") if line.strip()]
        records = []
        for number, line in enumerate(lines, start=1):
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                if number < len(lines):
                    raise ValueError(f"{self.path}: line {number} is not valid JSON") from None
        return records

    def append(self, record: dict) -> None:
        line = json.dumps(record, ensure_ascii=False)
        with self._lock:
            self._drop_cut_off_tail()
            with self.path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(line + "\n")
                handle.flush()
                os.fsync(handle.fileno())

    def _drop_cut_off_tail(self) -> None:
        if not self.path.exists():
            return
        with self.path.open("rb+") as handle:
            size = handle.seek(0, os.SEEK_END)
            if size == 0:
                return
            handle.seek(size - 1)
            if handle.read(1) == b"\n":
                return
            handle.seek(0)
            handle.truncate(handle.read().rfind(b"\n") + 1)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass(frozen=True)
class StratumProgress:
    held: int
    next_request: int


def progress_from_log(records: list[dict], stratum_index: int) -> StratumProgress:
    mine = [r for r in records if r["stratum_index"] == stratum_index]
    held = sum(len(r["items"]) for r in mine if r["status"] == "ok")
    next_request = max((r["request_index"] for r in mine), default=-1) + 1
    return StratumProgress(held, next_request)


@dataclass
class Generation:
    """Everything needed to send requests for one run.

    `pilot_round` is None for the main run and for diagnostic runs, whose
    seeds come from `seed_block`. `abort` is set when a request fails in a
    way that further requests cannot fix, such as a rejected parameter or a
    bad key; every stratum then stops.
    """

    run_name: str
    pilot_round: int | None
    seed_block: int
    base_seed: int
    settings: GeneratorConfig
    templates: Templates
    examples: ExampleLists
    schedule: ExampleSchedule
    log: RunLog
    complete: Callable[[dict], HttpResult]
    abort: threading.Event = field(default_factory=threading.Event)
    abort_reason: list[str] = field(default_factory=list)

    def send(self, stratum: Stratum, request_index: int) -> dict:
        size = self.settings.items_per_request
        seed = request_seed(self.base_seed, stratum.index, request_index, self.seed_block)
        spec = build_request(
            stratum, request_index, size, seed, self.templates, self.examples, self.schedule
        )
        payload = build_payload(spec, stratum.distractor, self.settings)
        started = _now()
        result = self.complete(payload)
        if result.error is None:
            parsed = parse_response(result.body, stratum.distractor, size)
        else:
            parsed = Parsed(None, "http_error", result.error)
            if result.status in FATAL_STATUSES:
                self.abort_reason.append(f"{stratum.key}: {result.error}")
                self.abort.set()
        record = {
            "run": self.run_name,
            "pilot_round": self.pilot_round,
            "seed_block": self.seed_block,
            "stratum_index": stratum.index,
            "stratum": stratum.key,
            "request_index": request_index,
            "seed": seed,
            "size": size,
            "template": spec.template_name,
            "template_sha256": spec.template_sha256,
            "request_text": spec.text,
            "params": {key: value for key, value in payload.items() if key != "messages"},
            "started_at": started,
            "finished_at": _now(),
            "attempts": result.attempts,
            "http_status": result.status,
            "status": parsed.status,
            "detail": parsed.detail,
            "response": result.body,
            "items": parsed.items or [],
        }
        self.log.append(record)
        return record


def fill_stratum(generation: Generation, stratum: Stratum, done: list[dict]) -> str:
    """Send requests for one stratum until it holds its target, and report the outcome."""
    try:
        progress = progress_from_log(done, stratum.index)
        held, request_index = progress.held, progress.next_request
        failures = 0
        while held < stratum.target:
            if generation.abort.is_set():
                return f"{stratum.key}: aborted ({held}/{stratum.target})"
            if failures >= MAX_CONSECUTIVE_FAILURES:
                return f"{stratum.key}: stopped after {failures} failed requests ({held}/{stratum.target})"
            record = generation.send(stratum, request_index)
            request_index += 1
            if record["status"] == "ok" and record["items"]:
                held += len(record["items"])
                failures = 0
            else:
                failures += 1
        return f"{stratum.key}: {held}/{stratum.target}"
    except Exception as exc:  # report and let the other strata finish
        return f"{stratum.key}: failed with {type(exc).__name__}: {exc}"


def run_main(generation: Generation, strata: list[Stratum]) -> list[str]:
    done = generation.log.read()
    with ThreadPoolExecutor(max_workers=generation.settings.max_workers) as pool:
        futures = [pool.submit(fill_stratum, generation, s, done) for s in strata]
        return [future.result() for future in futures]


def run_pilot(generation: Generation, strata: list[Stratum]) -> list[str]:
    """Send one request for each pilot stratum without a successful one logged.

    A stratum whose earlier pilot request failed is sent again with the next
    request index, so no logged request index is reused.
    """
    records = generation.log.read()
    succeeded = {r["stratum_index"] for r in records if r["status"] == "ok"}

    def send_once(stratum: Stratum) -> str:
        if generation.abort.is_set():
            return f"{stratum.key}: aborted"
        request_index = progress_from_log(records, stratum.index).next_request
        try:
            record = generation.send(stratum, request_index)
        except Exception as exc:  # report and let the other strata finish
            return f"{stratum.key}: failed with {type(exc).__name__}: {exc}"
        return f"{stratum.key}: {record['status']} ({len(record['items'])} items)"

    todo = [s for s in strata if s.index not in succeeded]
    with ThreadPoolExecutor(max_workers=generation.settings.max_workers) as pool:
        return list(pool.map(send_once, todo))


def pilot_strata(strata: list[Stratum], seen: tuple[Category, ...]) -> list[Stratum]:
    """One stratum per in-scope category and motion label, and one per kind and motion label.

    Pilot distractor strata rotate through the seen categories. Pilot strata
    are renumbered from 0 and use the pilot seed range.
    """
    chosen = [s for s in strata if not s.distractor]
    pairs = [(kind, motion) for kind in KIND_ORDER for motion in (True, False)]
    for number, (kind, motion) in enumerate(pairs):
        chosen.append(Stratum(0, seen[number % len(seen)], motion, True, kind, 1))
    return [
        Stratum(i, s.category, s.motion, s.distractor, s.kind, s.quota)
        for i, s in enumerate(chosen)
    ]
