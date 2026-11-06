"""The candidate pool.

Candidates are read from one or more run logs and numbered by stratum
order, request index and item position, so `candidate_id` does not depend
on the order in which responses arrived. Each candidate is checked against
the code rules once, and removals are recorded with a reason rather than
deleted, so the build can report every step.
"""

import random
from collections import Counter
from dataclasses import dataclass

from .lexicon import ELEMENT_KINDS, Lexicon
from .rules import check_distractor, in_scope_failures
from .strata import Stratum


@dataclass
class Candidate:
    candidate_id: int
    stratum: Stratum
    request_index: int
    prompt: str
    base_prompt: str
    excluded_element: str | None
    motion_flag: bool
    removed: str | None = None
    split: str | None = None

    @property
    def kept(self) -> bool:
        return self.removed is None

    @property
    def distractor(self) -> bool:
        return self.stratum.distractor


def _motion_flag(stratum: Stratum, prompt: str, base_prompt: str, lexicon: Lexicon) -> bool:
    """True when the motion label and the motion-verb check disagree."""
    if stratum.distractor and stratum.motion:
        return not lexicon.mentions_motion(base_prompt)
    return lexicon.mentions_motion(prompt) != stratum.motion


def load_candidates(
    records: list[dict],
    strata: dict[str, Stratum],
    lexicon: Lexicon,
    run_order: list[str] | None = None,
) -> list[Candidate]:
    """Read every parsed item into a candidate and apply the code rules.

    Candidates are numbered by run (in `run_order`, so the main run comes
    before any refill run), then stratum order, request index and item
    position. Adding a refill run therefore never changes the numbers of
    earlier candidates.
    """
    seen = set()
    for record in records:
        key = (record["stratum"], record["request_index"])
        if key in seen:
            raise ValueError(f"request {key} appears in more than one log record")
        seen.add(key)
    rank = {name: position for position, name in enumerate(run_order or [])}
    usable = [r for r in records if r["status"] == "ok"]
    usable.sort(
        key=lambda r: (rank.get(r.get("run"), 0), strata[r["stratum"]].index, r["request_index"])
    )
    candidates = []
    for record in usable:
        stratum = strata[record["stratum"]]
        for item in record["items"]:
            if stratum.distractor:
                prompt, base = item["with_element"], item["base"]
                check = check_distractor(prompt, base, stratum.kind, lexicon)
                failures, element = list(check.failures), check.excluded_element
            else:
                prompt = base = item
                failures, element = in_scope_failures(prompt, lexicon), None
            if lexicon.is_spec_example(prompt) or lexicon.is_spec_example(base):
                failures.append("spec_example")
            candidates.append(
                Candidate(
                    candidate_id=len(candidates),
                    stratum=stratum,
                    request_index=record["request_index"],
                    prompt=prompt,
                    base_prompt=base,
                    excluded_element=element,
                    motion_flag=_motion_flag(stratum, prompt, base, lexicon),
                    removed=f"rules:{','.join(failures)}" if failures else None,
                )
            )
    return candidates


def remove_motion_flagged(candidates: list[Candidate]) -> int:
    """Remove every kept candidate whose motion label is flagged, in every split alike."""
    removed = 0
    for candidate in candidates:
        if candidate.kept and candidate.motion_flag:
            candidate.removed = "motion_flag"
            removed += 1
    return removed


def apply_element_cap(candidates: list[Candidate], share: float, rng: random.Random) -> int:
    """Drop distractors until no excluded element exceeds `share` of its kind.

    Only candidates holding an over-cap element are dropped, one at a time in
    shuffled order, with the shares recomputed after each drop. A kind with
    too few distinct elements could never meet the cap, so that is an error
    rather than a reason to drop the whole kind. Returns the number dropped.
    """
    dropped = 0
    for kind in ELEMENT_KINDS:
        group = [c for c in candidates if c.kept and c.distractor and c.stratum.kind == kind]
        counts = Counter(c.excluded_element for c in group)
        if group and len(counts) * share < 1:
            raise ValueError(
                f"{kind}: {len(counts)} distinct elements cannot meet a {share:.0%} cap"
            )
        total = len(group)
        order = list(group)
        rng.shuffle(order)
        changed = True
        while changed:
            changed = False
            for candidate in order:
                if candidate.kept and counts[candidate.excluded_element] > share * total:
                    candidate.removed = "element_cap"
                    counts[candidate.excluded_element] -= 1
                    total -= 1
                    dropped += 1
                    changed = True
    return dropped
