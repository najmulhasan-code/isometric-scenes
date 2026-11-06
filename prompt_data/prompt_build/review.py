"""Review of the evaluation splits and the train audit.

Review is a model filter with an author spot check. In each round a pinned
language model reads every item (prescreen.py), and every item it flags is
removed. In round 1 the author also reads a seeded sample of unflagged items
and removes any that break a rule; this measures the model's miss rate.
Model flags and author removals are stored separately.

Review works in rounds. The first round holds every prompt in `val`,
`test` and `test_unseen`, and a seeded audit sample of `train`, in one
shuffled list without split labels. Prompts are removed, each with a rule as
the reason, and never edited.

When a round closes, each removed prompt is replaced by the next reserve of
its stratum. A replacement that joins an evaluation split goes into the
next round, mixed with an equal number of not yet audited `train` prompts,
which join the audit. A replacement that joins `train` is not reviewed,
like the rest of `train`. Review is complete when a round removes nothing
from the evaluation splits.

The state is kept in builds/<name>/review/state.json, so every decision is
recorded and the final splits can be rebuilt from it.
"""

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

from .config import BuildConfig

BATCH_SIZE = 50
SPOT_CHECK_SIZE = 50
Z_95 = 1.959964
REASONS = ("R3", "R4", "R6", "R8", "R10", "motion")
EVAL_SPLITS = ("val", "test", "test_unseen")


def wilson_interval(successes: int, total: int, z: float = Z_95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if total == 0:
        return 0.0, 1.0
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return max(0.0, centre - half), min(1.0, centre + half)


class NoReserves(ValueError):
    """A stratum has no reserves left to replace a removed prompt."""

    def __init__(self, stratum: str) -> None:
        super().__init__(f"no reserves left for {stratum}")
        self.stratum = stratum


@dataclass(frozen=True)
class Row:
    candidate_id: int
    stratum: str
    prompt: str
    base_prompt: str
    excluded_element: str | None
    split: str | None
    removed: str | None


def load_rows(build_dir: Path) -> dict[int, Row]:
    """Every candidate of the build, and any added later as reserves during review."""
    rows = {}
    lines = (build_dir / "candidates.jsonl").read_text(encoding="utf-8").splitlines()
    late = build_dir / "review" / "late_candidates.jsonl"
    if late.exists():
        lines += late.read_text(encoding="utf-8").splitlines()
    for line in lines:
        data = json.loads(line)
        rows[data["candidate_id"]] = Row(
            data["candidate_id"],
            data["stratum"],
            data["prompt"],
            data["base_prompt"],
            data["excluded_element"],
            data["split"],
            data["removed"],
        )
    return rows


class Review:
    def __init__(self, build_dir: Path, config: BuildConfig) -> None:
        self.build_dir = build_dir
        self.path = build_dir / "review" / "state.json"
        self.config = config
        self.rows = load_rows(build_dir)
        if self.path.exists():
            self.state = json.loads(self.path.read_text(encoding="utf-8"))
        else:
            self.state = None

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.state, indent=1) + "\n", encoding="utf-8", newline="\n")

    def _seed(self, extra: int = 0) -> int:
        return self.config.base_seed + self.config.seed_offsets["train_audit"] + extra

    def start(self) -> None:
        if self.state is not None:
            raise ValueError("review has already started")
        splits = {cid: row.split for cid, row in self.rows.items() if row.split}
        evaluation = sorted(cid for cid, split in splits.items() if split in EVAL_SPLITS)
        train = sorted(cid for cid, split in splits.items() if split == "train")
        audit = random.Random(self._seed()).sample(train, self.config.build.audit_size)
        reserves = json.loads((self.build_dir / "reserves.json").read_text(encoding="utf-8"))
        self.state = {
            "splits": {str(cid): split for cid, split in splits.items()},
            "reserves": reserves,
            "audited": sorted(audit),
            "removed": {},
            "replaced": {},
            "rounds": [],
        }
        self._open_round([(cid, "eval") for cid in evaluation] + [(cid, "audit") for cid in audit])
        self.save()

    def _open_round(self, items: list[tuple[int, str]]) -> None:
        number = len(self.state["rounds"]) + 1
        ordered = sorted(items)
        random.Random(self._seed(number)).shuffle(ordered)
        self.state["rounds"].append(
            {
                "items": [
                    {"number": n, "candidate_id": cid, "role": role}
                    for n, (cid, role) in enumerate(ordered, start=1)
                ],
                "marks": {},
                "closed": False,
            }
        )

    @property
    def current(self) -> dict:
        if not self.state or not self.state["rounds"]:
            raise ValueError("review has not started")
        return self.state["rounds"][-1]

    def batches(self) -> int:
        return math.ceil(len(self.current["items"]) / BATCH_SIZE)

    def _line(self, number: int, candidate_id: int, categories: dict[str, str]) -> str:
        row = self.rows[candidate_id]
        parts = row.stratum.split("/")
        motion = "should show motion" if parts[-1] == "motion" else "should be still"
        note = f"[{categories[parts[1]]}; {motion}"
        if parts[0] == "distractor":
            note += f"; extra added separately: {row.excluded_element}"
        return f"{number}. {row.base_prompt}  {note}]"

    def show(self, batch: int, categories: dict[str, str]) -> list[str]:
        """One batch of the current round, as the first reader sees it."""
        items = self.current["items"][(batch - 1) * BATCH_SIZE : batch * BATCH_SIZE]
        return [self._line(item["number"], item["candidate_id"], categories) for item in items]

    def sheet(self, categories: dict[str, str]) -> list[str]:
        """The author's reading list: the spot-check sample of the current round."""
        round_ = self.current
        if "spot_check" not in round_:
            raise ValueError("draw the spot-check sample of this round first")
        by_number = {item["number"]: item["candidate_id"] for item in round_["items"]}
        return [self._line(n, by_number[n], categories) for n in round_["spot_check"]]

    def flag(self, flags: dict[int, str], reasons: dict[int, str] | None = None) -> None:
        """Record the language model's first-pass flags for the current round, with its reasons."""
        round_ = self.current
        if "spot_check" in round_:
            raise ValueError("the spot-check sample is drawn from these flags, so they can no longer change")
        self._validate(round_, flags)
        round_["model_flags"] = {str(n): r for n, r in flags.items()}
        round_["model_reasons"] = {str(n): (reasons or {}).get(n, "") for n in flags}
        self.save()

    def spot_check(self, size: int) -> list[int]:
        """A seeded sample of items the model did not flag, for the author to read.

        Round 1 uses SPOT_CHECK_SIZE and later rounds 0.
        """
        round_ = self.current
        if "model_flags" not in round_:
            raise ValueError("record the model flags for this round first")
        if "spot_check" in round_:
            raise ValueError("the spot-check sample of this round is already drawn")
        flagged = {int(n) for n in round_["model_flags"]}
        unflagged = sorted(item["number"] for item in round_["items"] if item["number"] not in flagged)
        number = len(self.state["rounds"])
        sample = random.Random(self._seed(200 + number)).sample(unflagged, min(size, len(unflagged)))
        round_["spot_check"] = sorted(sample)
        self.save()
        return round_["spot_check"]

    def _validate(self, round_: dict, marks: dict[int, str]) -> None:
        if round_["closed"]:
            raise ValueError("the current round is closed")
        numbers = {item["number"] for item in round_["items"]}
        for number, reason in marks.items():
            if number not in numbers:
                raise ValueError(f"item {number} is not in the current round")
            if reason not in REASONS:
                raise ValueError(f"reason {reason!r} is not one of {REASONS}")

    def decide(self, remove: dict[int, str]) -> None:
        """Record the author's spot-check result for the current round.

        Every model flag becomes a removal. `remove` lists items of the
        spot-check sample that the author removes as well.
        """
        round_ = self.current
        if round_["closed"]:
            raise ValueError("the current round is closed")
        if "spot_check" not in round_:
            raise ValueError("draw the spot-check sample of this round first")
        if not remove.keys() <= set(round_["spot_check"]):
            raise ValueError(f"items {sorted(remove.keys() - set(round_['spot_check']))} are not in the sample")
        if any(reason not in REASONS for reason in remove.values()):
            raise ValueError(f"every reason must be one of {REASONS}")
        removals = {int(n): r for n, r in round_["model_flags"].items()} | remove
        round_["marks"] = {str(n): r for n, r in sorted(removals.items())}
        round_["author"] = {"removed": {str(n): r for n, r in sorted(remove.items())}}
        self.save()

    def close(self) -> int:
        """Close the current round, apply replacements, and open the next round if needed.

        Returns the number of items in the new round, or 0 when review is complete.
        """
        round_ = self.current
        if round_["closed"]:
            raise ValueError("the current round is already closed")
        if "author" not in round_:
            raise ValueError("record the author's spot-check result for this round first")
        by_number = {item["number"]: item for item in round_["items"]}
        used = {int(cid) for cid in self.state["splits"]}
        next_items: list[tuple[int, str]] = []
        for number, reason in sorted((int(n), r) for n, r in round_["marks"].items()):
            item = by_number[number]
            cid = item["candidate_id"]
            split = self.state["splits"].pop(str(cid))
            self.state["removed"][str(cid)] = reason
            replacement = self._take_reserve(self.rows[cid], used)
            self.state["splits"][str(replacement)] = split
            self.state["replaced"][str(cid)] = replacement
            used.add(replacement)
            if split in EVAL_SPLITS:
                next_items.append((replacement, "eval"))
        round_["closed"] = True

        if next_items:
            audited = set(self.state["audited"])
            unaudited = sorted(
                int(cid) for cid, split in self.state["splits"].items()
                if split == "train" and int(cid) not in audited
            )
            extra = random.Random(self._seed(100 + len(self.state["rounds"]))).sample(
                unaudited, min(len(next_items), len(unaudited))
            )
            self.state["audited"] = sorted(audited | set(extra))
            self._open_round(next_items + [(cid, "audit") for cid in extra])
        self.save()
        return len(self.current["items"]) if next_items else 0

    def _take_reserve(self, row: Row, used: set[int]) -> int:
        """The next unused reserve of the row's stratum, in the build's shuffled order."""
        queue = self.state["reserves"].get(row.stratum, [])
        while queue:
            candidate = queue.pop(0)
            if candidate not in used:
                return candidate
        raise NoReserves(row.stratum)

    def add_reserves(self, rows: list[dict]) -> None:
        """Append late reserves, in the given order, and keep their rows with the review."""
        late = self.path.parent / "late_candidates.jsonl"
        with late.open("a", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                self.state["reserves"].setdefault(row["stratum"], []).append(row["candidate_id"])
        self.rows = load_rows(self.build_dir)
        self.save()

    def coverage_problems(self) -> list[str]:
        """Every evaluation prompt was reviewed and kept, and the audit is full size."""
        problems = []
        if any(not round_["closed"] for round_ in self.state["rounds"]):
            problems.append("a review round is still open")
        reviewed, marked = set(), set()
        for round_ in self.state["rounds"]:
            for item in round_["items"]:
                reviewed.add(item["candidate_id"])
                if str(item["number"]) in round_["marks"]:
                    marked.add(item["candidate_id"])
        for cid, split in self.state["splits"].items():
            if split in EVAL_SPLITS and (int(cid) not in reviewed or int(cid) in marked):
                problems.append(f"{split} candidate {cid} was not reviewed and kept")
        for index, round_ in enumerate(self.state["rounds"], start=1):
            if "author" not in round_:
                problems.append(f"round {index} has no spot-check result")
        if len(self.state["audited"]) < self.config.build.audit_size:
            problems.append(f"the audit holds {len(self.state['audited'])} prompts, fewer than required")
        return problems

    def audit_result(self) -> tuple[int, int]:
        """Audited train prompts removed, and audited train prompts reviewed."""
        audit_numbers = 0
        audit_removed = 0
        for round_ in self.state["rounds"]:
            for item in round_["items"]:
                if item["role"] == "audit":
                    audit_numbers += 1
                    audit_removed += str(item["number"]) in round_["marks"]
        return audit_removed, audit_numbers

    def summary(self) -> dict:
        """Removals, the model's miss rate in the spot check, and the audit result, for the release record.

        Only rounds with a recorded spot-check result are counted.
        """
        rounds = []
        for round_ in self.state["rounds"]:
            author = round_.get("author")
            if author is None:
                continue
            missed, sample = len(author["removed"]), len(round_["spot_check"])
            entry = {
                "items": len(round_["items"]),
                "model_flags": len(round_["model_flags"]),
                "spot_check": sample,
                "spot_check_removed_by_author": missed,
                "removed": len(round_["marks"]),
            }
            if sample:
                entry["miss_rate_wilson_95"] = rounded(wilson_interval(missed, sample))
            rounds.append(entry)
        audit_removed, audit_size = self.audit_result()
        by_rule: dict[str, dict[str, int]] = {"eval": {}, "audit": {}}
        for round_ in self.state["rounds"]:
            for item in round_["items"]:
                rule = round_["marks"].get(str(item["number"]))
                if rule:
                    counts = by_rule[item["role"]]
                    counts[rule] = counts.get(rule, 0) + 1
        return {
            "rounds": rounds,
            "removed_by_rule": {role: dict(sorted(counts.items())) for role, counts in by_rule.items()},
            "audit_removed": audit_removed,
            "audit_size": audit_size,
            "audit_error_wilson_95": rounded(wilson_interval(audit_removed, audit_size)),
        }


def rounded(interval: tuple[float, float]) -> list[float]:
    return [round(value, 4) for value in interval]
