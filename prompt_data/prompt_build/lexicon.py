"""Word lists and term matching."""

from dataclasses import dataclass
from pathlib import Path

from .text import matching_tokens

TERM_KINDS = ("human", "animal", "face_figure", "implied_people", "animal_object")
ELEMENT_KINDS = ("human", "animal", "face_figure")


@dataclass(frozen=True)
class Term:
    tokens: tuple[str, ...]
    kind: str
    lemma: str


@dataclass(frozen=True)
class Match:
    """A term found in text; start and end are token positions."""

    term: Term
    start: int
    end: int


def read_word_list(path: Path) -> list[str]:
    """Return the entries of a list file: one per line, skipping blanks and # comments."""
    entries = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            entries.append(line)
    return entries


def read_terms(path: Path) -> list[Term]:
    """Read the typed term list: term, kind and lemma, separated by tabs."""
    terms = []
    for line in read_word_list(path):
        fields = line.split("\t")
        if len(fields) != 3:
            raise ValueError(f"{path.name}: expected three tab-separated fields in {line!r}")
        text, kind, lemma = fields
        if kind not in TERM_KINDS:
            raise ValueError(f"{path.name}: unknown kind {kind!r} in {line!r}")
        tokens = tuple(matching_tokens(text))
        if not tokens:
            raise ValueError(f"{path.name}: empty term in {line!r}")
        terms.append(Term(tokens, kind, lemma))
    return terms


class PhraseMask:
    """Marks the tokens covered by allowed phrases, longest phrase first."""

    def __init__(self, phrases: list[str]) -> None:
        self._phrases = {tuple(matching_tokens(phrase)) for phrase in phrases} - {()}
        self._longest = max((len(phrase) for phrase in self._phrases), default=0)

    def apply(self, tokens: list[str]) -> list[bool]:
        masked = [False] * len(tokens)
        position = 0
        while position < len(tokens):
            for size in range(min(self._longest, len(tokens) - position), 0, -1):
                if tuple(tokens[position : position + size]) in self._phrases:
                    masked[position : position + size] = [True] * size
                    position += size
                    break
            else:
                position += 1
        return masked


class TermMatcher:
    """Finds out-of-scope terms in text.

    Tokens covered by an allowed phrase are skipped. Terms are matched left to
    right as whole tokens, trying the longest term at each position first, so
    a multi-word term such as "teddy bear" is counted once.
    """

    def __init__(self, terms: list[Term], mask: PhraseMask) -> None:
        self._terms: dict[tuple[str, ...], Term] = {}
        for term in terms:
            known = self._terms.get(term.tokens)
            if known is not None and known != term:
                raise ValueError(f"conflicting entries for {' '.join(term.tokens)!r}")
            self._terms[term.tokens] = term
        self._longest = max((len(key) for key in self._terms), default=0)
        self._mask = mask

    def find(self, text: str) -> list[Match]:
        tokens = matching_tokens(text)
        masked = self._mask.apply(tokens)
        matches = []
        position = 0
        while position < len(tokens):
            match = self._match_at(tokens, masked, position)
            if match is None:
                position += 1
            else:
                matches.append(match)
                position = match.end
        return matches

    def _match_at(self, tokens: list[str], masked: list[bool], start: int) -> Match | None:
        for size in range(min(self._longest, len(tokens) - start), 0, -1):
            end = start + size
            if any(masked[start:end]):
                continue
            term = self._terms.get(tuple(tokens[start:end]))
            if term is not None:
                return Match(term, start, end)
        return None


@dataclass(frozen=True)
class Lexicon:
    """All word lists used by the code-checked rules.

    The allow-list applies to every list, so "3d printer" is not a style term
    and "rocking chair" is not a motion cue.
    """

    matcher: TermMatcher
    mask: PhraseMask
    style_terms: frozenset[str]
    overused_terms: frozenset[str]
    motion_terms: frozenset[str]
    spec_examples: frozenset[str]

    def free_tokens(self, text: str) -> set[str]:
        """Tokens of the text that are not part of an allowed phrase."""
        tokens = matching_tokens(text)
        masked = self.mask.apply(tokens)
        return {token for token, hidden in zip(tokens, masked) if not hidden}

    def mentions_motion(self, text: str) -> bool:
        return not self.motion_terms.isdisjoint(self.free_tokens(text))

    def is_spec_example(self, text: str) -> bool:
        return " ".join(matching_tokens(text)) in self.spec_examples


def load_lexicon(terms_dir: Path) -> Lexicon:
    def tokens_of(name: str) -> frozenset[str]:
        entries = read_word_list(terms_dir / name)
        return frozenset(token for entry in entries for token in matching_tokens(entry))

    mask = PhraseMask(read_word_list(terms_dir / "allow_list.txt"))
    examples = read_word_list(terms_dir / "spec_examples.txt")
    return Lexicon(
        matcher=TermMatcher(read_terms(terms_dir / "out_of_scope.tsv"), mask),
        mask=mask,
        style_terms=tokens_of("style_terms.txt"),
        overused_terms=tokens_of("overused_terms.txt"),
        motion_terms=tokens_of("motion_terms.txt"),
        spec_examples=frozenset(" ".join(matching_tokens(e)) for e in examples),
    )
