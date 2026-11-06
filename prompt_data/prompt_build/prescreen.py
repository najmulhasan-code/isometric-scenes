"""Model pre-screen of a review round.

Each item of the round is sent on its own to one pinned model, with the
fixed prompt in templates/prescreen.txt, at temperature 0. The model sees
the base prompt, its category and its motion label, never its split. Its
answer is one JSON object: keep, or flag with a rule and a reason.

Every request and reply is appended to a log in the review folder. A logged
verdict counts only for the same candidate and the same request, so a log
left by another round or another prompt is never reused. The first valid
verdict for an item is kept. A re-run sends only the items that have no
valid verdict yet, so an interrupted pre-screen resumes where it stopped.

The run stops sending at the first reply that a retry cannot fix: an error
status other than 429 or 5xx, or a reply cut off at the token limit. A lock
file stops two runs from writing the same log at once.
"""

import hashlib
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from string import Template
from threading import Event

from .client import HttpResult
from .config import Category, PrescreenConfig
from .generate import RunLog
from .review import REASONS, Review

TEMPLATE_FIELDS = {"prompt", "category_table", "category_name", "motion_label"}
JSON_OBJECT = re.compile(r"\{.*\}", re.S)


@dataclass(frozen=True)
class Verdict:
    flag: str | None
    reason: str


@dataclass
class Outcome:
    """The result of one pre-screen run."""

    flags: dict[int, Verdict] | None
    sent: int
    stopped: list[str] = field(default_factory=list)
    totals: dict = field(default_factory=dict)


def build_payload(template: Template, settings: PrescreenConfig, fields: dict[str, str]) -> dict:
    return {
        "model": settings.model,
        "messages": [{"role": "user", "content": template.substitute(fields)}],
        "temperature": settings.temperature,
        "max_tokens": settings.max_tokens,
    }


def payload_hash(payload: dict) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def parse_verdict(body: dict | None) -> Verdict | None:
    """The model's verdict, or None if the reply does not follow the format."""
    try:
        content = body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return None
    match = JSON_OBJECT.search(content or "")
    if not match:
        return None
    try:
        answer = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(answer, dict):
        return None
    if answer.get("verdict") == "keep":
        return Verdict(None, "")
    if answer.get("verdict") == "flag" and answer.get("rule") in REASONS:
        return Verdict(answer["rule"], str(answer.get("reason", "")))
    return None


def stop_reason(result: HttpResult) -> str | None:
    """Why a reply means the run must stop, or None if it may go on."""
    if result.error and result.status is not None and result.status != 429 and result.status < 500:
        return result.error[:200]
    try:
        choice = result.body["choices"][0]
    except (KeyError, IndexError, TypeError):
        return None if result.error else "the reply has no choices"
    if choice.get("finish_reason") == "length":
        return "a reply was cut off at max_tokens"
    return None


def category_table(categories: tuple[Category, ...]) -> str:
    """The categories in their order of precedence."""
    return "\n".join(f"{n}. {c.name}: {c.definition}" for n, c in enumerate(categories, start=1))


def item_fields(review: Review, candidate_id: int, categories: tuple[Category, ...]) -> dict[str, str]:
    row = review.rows[candidate_id]
    parts = row.stratum.split("/")
    names = {c.slug: c.name for c in categories}
    motion = "should show motion" if parts[-1] == "motion" else "should be still"
    return {"prompt": row.base_prompt, "category_table": category_table(categories),
            "category_name": names[parts[1]], "motion_label": motion}


def log_path(review: Review) -> Path:
    return review.path.parent / f"prescreen_round{len(review.state['rounds'])}.jsonl"


def verdicts(records: list[dict], expected: dict[int, tuple[int, str]]) -> dict[int, Verdict]:
    """The first valid verdict for each item, from records made for that item's current request.

    `expected` maps an item number to its candidate id and request hash.
    """
    found = {}
    for record in records:
        number = record["number"]
        if number in found or expected.get(number) != (record["candidate_id"], record["payload_sha256"]):
            continue
        verdict = parse_verdict(record.get("response"))
        if verdict is not None:
            found[number] = verdict
    return found


def run_totals(records: list[dict]) -> dict:
    """Token use over every logged request, the served model names, and requests that needed a retry."""
    totals = {"requests": len(records), "prompt_tokens": 0, "completion_tokens": 0, "retried": 0}
    served = set()
    for record in records:
        response = record.get("response") or {}
        usage = response.get("usage") or {}
        totals["prompt_tokens"] += int(usage.get("prompt_tokens") or 0)
        totals["completion_tokens"] += int(usage.get("completion_tokens") or 0)
        totals["retried"] += record.get("attempts", 1) > 1
        if response.get("model"):
            served.add(response["model"])
    totals["served_models"] = sorted(served)
    return totals


def round_requests(
    review: Review, settings: PrescreenConfig, template: Template, categories: tuple[Category, ...]
) -> tuple[dict[int, dict], dict[int, tuple[int, str]]]:
    """The request for each item of the current round, and each item's candidate id and request hash."""
    items = review.current["items"]
    payloads = {
        item["number"]: build_payload(template, settings, item_fields(review, item["candidate_id"], categories))
        for item in items
    }
    expected = {item["number"]: (item["candidate_id"], payload_hash(payloads[item["number"]])) for item in items}
    return payloads, expected


def recorded_round(
    review: Review,
    settings: PrescreenConfig,
    template: Template,
    categories: tuple[Category, ...],
    records: list[dict],
) -> dict[int, Verdict] | None:
    """The verdicts of the current round read from a log, without sending anything.

    None when some item has no valid verdict in the log for its current request.
    """
    _, expected = round_requests(review, settings, template, categories)
    found = verdicts(records, expected)
    return dict(sorted(found.items())) if len(found) == len(expected) else None


class Lock:
    """An exclusive lock file; a second run fails at once instead of waiting."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def __enter__(self) -> "Lock":
        try:
            os.close(os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
        except FileExistsError:
            raise ValueError(f"{self.path} exists: another pre-screen is running, or one crashed; "
                             "delete the file if none is running") from None
        return self

    def __exit__(self, *exc) -> None:
        self.path.unlink(missing_ok=True)


def prescreen_round(
    review: Review,
    settings: PrescreenConfig,
    template: Template,
    categories: tuple[Category, ...],
    complete,
) -> Outcome:
    """Screen every item of the current round without a valid verdict.

    The outcome holds the verdicts of all items when every item has one,
    else None, with the number of requests sent and any reason for stopping.
    """
    items = review.current["items"]
    payloads, expected = round_requests(review, settings, template, categories)
    log = RunLog(log_path(review))
    with Lock(log.path.with_suffix(".lock")):
        done = verdicts(log.read(), expected)
        todo = [item for item in items if item["number"] not in done]
        stop = Event()
        stopped: list[str] = []

        def screen(item: dict) -> bool:
            if stop.is_set():
                return False
            payload = payloads[item["number"]]
            result: HttpResult = complete(payload)
            log.append({
                "sent_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "number": item["number"],
                "candidate_id": item["candidate_id"],
                "payload_sha256": expected[item["number"]][1],
                "payload": payload,
                "status": result.status,
                "attempts": result.attempts,
                "error": result.error,
                "response": result.body,
            })
            reason = stop_reason(result)
            if reason:
                stopped.append(f"item {item['number']}: {reason}")
                stop.set()
            return True

        pool = ThreadPoolExecutor(max_workers=settings.max_workers)
        try:
            sent = sum(future.result() for future in [pool.submit(screen, item) for item in todo])
        finally:
            stop.set()
            pool.shutdown(wait=True, cancel_futures=True)

        records = log.read()
        done = verdicts(records, expected)
    complete_round = all(item["number"] in done for item in items)
    return Outcome(dict(sorted(done.items())) if complete_round else None, sent, stopped, run_totals(records))
