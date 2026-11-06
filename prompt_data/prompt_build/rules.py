"""Prompt rules checked by code.

Each check returns the ids of the rules a prompt breaks. Rule ids R1 to R9
follow the dataset rules. D1 and D2 are the two distractor checks:
D1, no single inserted span turns the prompt back into its base prompt;
D2, the prompt does not hold exactly one out-of-scope term, of the expected
kind, inside such a span.

Curly quotes are straightened before any check. The intake
code stores straightened text, so the text that is checked is the text that
is stored; the checks also straighten their input, in case a caller does not.
"""

import re
from dataclasses import dataclass

from .lexicon import ELEMENT_KINDS, Lexicon
from .text import matching_tokens, normalise_punctuation

MIN_WORDS = 4
MAX_WORDS = 25
MAX_QUOTED_WORDS = 2

_OPENING = re.compile(r"an? [^\n]*[^.\n]")
_QUOTED_SPAN = re.compile(r'"[^"]*"')


def opens_correctly(prompt: str) -> bool:
    """R1: starts with "a " or "an ", one line, no outer whitespace, no final full stop."""
    return (
        prompt == prompt.strip()
        and len(prompt.splitlines()) == 1
        and _OPENING.fullmatch(prompt) is not None
    )


def has_allowed_length(prompt: str) -> bool:
    """R2: 4 to 25 words, counted by splitting on whitespace."""
    return MIN_WORDS <= len(prompt.split()) <= MAX_WORDS


def has_capital_outside_quotes(prompt: str) -> bool:
    """R6, code part: any uppercase letter outside quoted text."""
    return any(char.isupper() for char in _QUOTED_SPAN.sub(" ", prompt))


def has_allowed_quote(prompt: str) -> bool:
    """R7: at most one quoted span, one or two words long."""
    quote_marks = prompt.count('"')
    if quote_marks == 0:
        return True
    if quote_marks != 2:
        return False
    quoted = prompt.split('"')[1]
    return 1 <= len(quoted.split()) <= MAX_QUOTED_WORDS


def format_failures(prompt: str, lexicon: Lexicon) -> list[str]:
    """Rules that apply to every prompt text: R1, R2, R5, R6, R7 and R9."""
    prompt = normalise_punctuation(prompt)
    tokens = lexicon.free_tokens(prompt)
    failures = []
    if not opens_correctly(prompt):
        failures.append("R1")
    if not has_allowed_length(prompt):
        failures.append("R2")
    if not lexicon.style_terms.isdisjoint(tokens):
        failures.append("R5")
    if has_capital_outside_quotes(prompt):
        failures.append("R6")
    if not has_allowed_quote(prompt):
        failures.append("R7")
    if not lexicon.overused_terms.isdisjoint(tokens):
        failures.append("R9")
    return failures


def in_scope_failures(prompt: str, lexicon: Lexicon) -> list[str]:
    """All code-checked rules for an in-scope prompt, including R8."""
    failures = format_failures(prompt, lexicon)
    if lexicon.matcher.find(prompt):
        failures.append("R8")
    return failures


def inserted_spans(prompt_words: list[str], base_words: list[str]) -> list[tuple[int, int]]:
    """Every (start, end) such that deleting prompt_words[start:end] leaves base_words."""
    extra = len(prompt_words) - len(base_words)
    if extra <= 0:
        return []
    prefix = 0
    while prefix < len(base_words) and prompt_words[prefix] == base_words[prefix]:
        prefix += 1
    suffix = 0
    while suffix < len(base_words) and prompt_words[-1 - suffix] == base_words[-1 - suffix]:
        suffix += 1
    first = len(base_words) - suffix
    return [(start, start + extra) for start in range(first, prefix + 1)]


def _token_offsets(words: list[str]) -> list[int]:
    """Token position at which each word starts, plus the total token count.

    Words are separated by whitespace, which is also a token separator, so the
    tokens of the whole text are the tokens of each word in turn.
    """
    offsets = [0]
    for word in words:
        offsets.append(offsets[-1] + len(matching_tokens(word)))
    return offsets


@dataclass(frozen=True)
class DistractorCheck:
    failures: tuple[str, ...]
    excluded_element: str | None


def check_distractor(
    prompt: str, base_prompt: str, element_kind: str, lexicon: Lexicon
) -> DistractorCheck:
    """Check a distractor pair. Failures of the base prompt are prefixed with "base_".

    Terms are matched in the full prompt, so allowed phrases that cross the
    edge of the inserted span (such as "hot dog") are respected. If several
    spans could be the inserted one, as happens next to a repeated word, the
    leftmost span that holds the single match is used.
    """
    if element_kind not in ELEMENT_KINDS:
        raise ValueError(f"unknown element kind {element_kind!r}")
    prompt = normalise_punctuation(prompt)
    base_prompt = normalise_punctuation(base_prompt)

    failures = [f"base_{rule}" for rule in in_scope_failures(base_prompt, lexicon)]
    failures += format_failures(prompt, lexicon)

    words = prompt.split()
    spans = inserted_spans(words, base_prompt.split())
    if not spans:
        return DistractorCheck((*failures, "D1"), None)

    matches = lexicon.matcher.find(prompt)
    if len(matches) == 1 and matches[0].term.kind == element_kind:
        match = matches[0]
        offsets = _token_offsets(words)
        for start, end in spans:
            if offsets[start] <= match.start and match.end <= offsets[end]:
                return DistractorCheck(tuple(failures), match.term.lemma)
    return DistractorCheck((*failures, "D2"), None)
