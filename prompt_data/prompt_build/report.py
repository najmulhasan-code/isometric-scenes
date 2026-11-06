"""Pilot and diagnostic reports, scored against the pilot pass criteria.

The report reads a run log, applies the code-checked rules to every parsed
item, and measures each pass criterion that code can measure. It also draws
a seeded sample of items for the hand check, which a person then reviews.
Responses are checked for degenerate output again here, so logs written
before that check existed are scored the same way.
"""

import random
from collections import Counter
from dataclasses import dataclass, field

from .generate import is_degenerate
from .lexicon import Lexicon
from .rules import check_distractor, in_scope_failures, opens_correctly, has_allowed_length

HAND_CHECK_SIZE = 50

CRITERIA = (
    ("responses that parse", "parse_rate", ">=", 0.95),
    ("responses that stop at the token limit", "length_rate", "<=", 0.02),
    ("degenerate responses", "degenerate_rate", "<=", 0.05),
    ("mean items per response", "mean_items", ">=", 18.0),
    ("items that are well-formed prompts (R1 and R2)", "well_formed_rate", ">=", 0.80),
    ("in-scope items passing the code filters", "in_scope_pass_rate", ">=", 0.60),
    ("distractor items passing the code filters", "distractor_pass_rate", ">=", 0.60),
    ("distractor pairs passing D1 and D2", "span_pass_rate", ">=", 0.70),
    ("fewest distinct excluded elements in one kind", "min_distinct_elements", ">=", 10),
    ("motion labels agreeing with the motion-verb check", "motion_agreement", ">=", 0.90),
)


@dataclass(frozen=True)
class StratumInfo:
    distractor: bool
    category: str
    kind: str | None
    motion: bool

    @classmethod
    def from_key(cls, key: str) -> "StratumInfo":
        parts = key.split("/")
        motion = parts[-1] == "motion"
        if parts[0] == "distractor":
            return cls(True, parts[1], parts[2], motion)
        return cls(False, parts[1], None, motion)


@dataclass
class Report:
    requests: int = 0
    parsed: int = 0
    length: int = 0
    degenerate: int = 0
    with_reasoning: int = 0
    items_total: int = 0
    well_formed: int = 0
    items_per_request: list[int] = field(default_factory=list)
    in_scope_items: int = 0
    in_scope_passed: int = 0
    distractor_items: int = 0
    distractor_passed: int = 0
    span_passed: int = 0
    motion_checked: int = 0
    motion_agreed: int = 0
    failures: Counter = field(default_factory=Counter)
    elements: dict[str, Counter] = field(default_factory=dict)
    statuses: Counter = field(default_factory=Counter)

    @staticmethod
    def _rate(part: int, whole: int) -> float | None:
        return part / whole if whole else None

    @property
    def parse_rate(self) -> float | None:
        return self._rate(self.parsed, self.requests)

    @property
    def length_rate(self) -> float | None:
        return self._rate(self.length, self.requests)

    @property
    def degenerate_rate(self) -> float | None:
        return self._rate(self.degenerate, self.requests)

    @property
    def mean_items(self) -> float | None:
        return self.items_total / self.requests if self.requests else None

    @property
    def well_formed_rate(self) -> float | None:
        return self._rate(self.well_formed, self.items_total)

    @property
    def in_scope_pass_rate(self) -> float | None:
        return self._rate(self.in_scope_passed, self.in_scope_items)

    @property
    def distractor_pass_rate(self) -> float | None:
        return self._rate(self.distractor_passed, self.distractor_items)

    @property
    def span_pass_rate(self) -> float | None:
        return self._rate(self.span_passed, self.distractor_items)

    @property
    def motion_agreement(self) -> float | None:
        return self._rate(self.motion_agreed, self.motion_checked)

    @property
    def min_distinct_elements(self) -> int | None:
        if not self.elements:
            return None
        return min(len(counter) for counter in self.elements.values())


def _motion_agrees(info: StratumInfo, item, lexicon: Lexicon) -> bool:
    if not info.distractor:
        return lexicon.mentions_motion(item) == info.motion
    if info.motion:
        return lexicon.mentions_motion(item["base"])
    return not lexicon.mentions_motion(item["with_element"])


def _reasoning_tokens(record: dict) -> int:
    usage = (record.get("response") or {}).get("usage") or {}
    return (usage.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0


def _is_well_formed(item) -> bool:
    text = item["with_element"] if isinstance(item, dict) else item
    return opens_correctly(text) and has_allowed_length(text)


def build_report(records: list[dict], lexicon: Lexicon) -> Report:
    report = Report()
    for record in records:
        report.requests += 1
        status = record["status"]
        if status == "ok" and is_degenerate(record["items"], record["size"]):
            status = "degenerate"
        report.statuses[status] += 1
        report.length += status == "length"
        report.degenerate += status == "degenerate"
        report.with_reasoning += _reasoning_tokens(record) > 0
        report.items_total += len(record["items"])
        report.well_formed += sum(_is_well_formed(item) for item in record["items"])
        if status != "ok":
            continue
        report.parsed += 1
        report.items_per_request.append(len(record["items"]))
        info = StratumInfo.from_key(record["stratum"])
        for item in record["items"]:
            report.motion_checked += 1
            report.motion_agreed += _motion_agrees(info, item, lexicon)
            if info.distractor:
                result = check_distractor(item["with_element"], item["base"], info.kind, lexicon)
                report.distractor_items += 1
                report.failures.update(result.failures)
                if not {"D1", "D2"} & set(result.failures):
                    report.span_passed += 1
                if not result.failures:
                    report.distractor_passed += 1
                    report.elements.setdefault(info.kind, Counter())[result.excluded_element] += 1
            else:
                failures = in_scope_failures(item, lexicon)
                if lexicon.is_spec_example(item):
                    failures.append("spec_example")
                report.in_scope_items += 1
                report.failures.update(failures)
                if not failures:
                    report.in_scope_passed += 1
    return report


def hand_check_sample(records: list[dict], seed: int, size: int = HAND_CHECK_SIZE) -> list[str]:
    """A seeded random sample of parsed items, written as one line each."""
    lines = []
    for record in records:
        if record["status"] != "ok":
            continue
        for item in record["items"]:
            if isinstance(item, dict):
                lines.append(f"{record['stratum']}\t{item['base']}\t{item['with_element']}")
            else:
                lines.append(f"{record['stratum']}\t{item}")
    rng = random.Random(seed)
    return rng.sample(lines, min(size, len(lines)))


def format_report(report: Report) -> list[str]:
    def show(value) -> str:
        if value is None:
            return "n/a"
        if isinstance(value, float) and value > 1:
            return f"{value:.1f}"
        return f"{value:.0%}" if isinstance(value, float) else str(value)

    items = report.items_per_request
    lines = [
        f"requests: {report.requests} ({dict(report.statuses)})",
        f"responses with reasoning tokens: {report.with_reasoning} of {report.requests}",
        f"items per parsed request: min {min(items, default=0)}, "
        f"mean {sum(items) / len(items) if items else 0:.1f}, max {max(items, default=0)}",
        "",
    ]
    passed_all = True
    for label, attribute, sign, limit in CRITERIA:
        value = getattr(report, attribute)
        ok = value is not None and (value >= limit if sign == ">=" else value <= limit)
        passed_all &= ok
        target = f"{limit:.0%}" if isinstance(limit, float) and limit <= 1 else str(limit)
        lines.append(f"{'PASS' if ok else 'FAIL'}  {label}: {show(value)} (needs {sign} {target})")
    lines.append("")
    lines.append("rule failures: " + ", ".join(f"{r} {n}" for r, n in report.failures.most_common()))
    for kind, counter in sorted(report.elements.items()):
        lines.append(f"{kind}: {len(counter)} distinct, top {counter.most_common(5)}")
    lines.append("")
    lines.append("code criteria: " + ("all pass" if passed_all else "not all pass"))
    lines.append("the hand check of the sample is still needed for the last criterion")
    return lines
