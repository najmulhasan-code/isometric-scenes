"""Refill requests for strata that ran short.

Refill happens only before any review. A refill run sends further requests
for the short strata of a build, one request per batch. Request indices
continue after those already logged for the stratum, in the main seed
block, so no seed repeats. Examples are read on from the stratum's place in
the example schedule, which may repeat examples of the next stratum's block.

A stratum stops when it holds twice its missing count, counted as items that
pass the code rules and carry no motion flag, or when it has had five
refill batches in total, counting batches sent by earlier refill runs.
"""

from dataclasses import dataclass

from .generate import Generation, progress_from_log
from .lexicon import Lexicon
from .pool import load_candidates
from .strata import Stratum

MAX_BATCHES = 5
MARGIN = 2


@dataclass(frozen=True)
class RefillOutcome:
    stratum: str
    missing: int
    usable: int
    batches: int

    @property
    def met(self) -> bool:
        return self.usable >= MARGIN * self.missing


def usable_items(record: dict, stratum: Stratum, lexicon: Lexicon) -> int:
    """Items of one logged request that pass the code rules and carry no motion flag."""
    candidates = load_candidates([record], {stratum.key: stratum}, lexicon)
    return sum(c.kept and not c.motion_flag for c in candidates)


def refill_stratum(
    generation: Generation,
    stratum: Stratum,
    missing: int,
    earlier_records: list[dict],
    earlier_refill_records: list[dict],
    lexicon: Lexicon,
) -> RefillOutcome:
    """Send refill batches for one stratum.

    `earlier_records` holds every logged request of the main and refill runs,
    and sets the next request index. `earlier_refill_records` holds only the
    refill requests, which count towards the limit of five batches and
    towards the usable items already gained.
    """
    mine = [r for r in earlier_refill_records if r["stratum"] == stratum.key]
    batches = len(mine)
    usable = sum(usable_items(r, stratum, lexicon) for r in mine if r["status"] == "ok")
    next_request = progress_from_log(earlier_records, stratum.index).next_request
    while usable < MARGIN * missing and batches < MAX_BATCHES and not generation.abort.is_set():
        record = generation.send(stratum, next_request)
        next_request += 1
        batches += 1
        if record["status"] == "ok":
            usable += usable_items(record, stratum, lexicon)
    return RefillOutcome(stratum.key, missing, usable, batches)
