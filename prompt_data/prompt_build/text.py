"""Text normalisation shared by the rule checks and term matching."""

import re
import unicodedata

_QUOTE_MAP = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201a": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u201e": '"',
        "\u00ab": '"',
        "\u00bb": '"',
        "\u2010": "-",
        "\u2011": "-",
    }
)

_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def normalise_punctuation(text: str) -> str:
    """Replace curly quotes and apostrophes with straight ones, and Unicode hyphens
    with the ASCII hyphen."""
    return text.translate(_QUOTE_MAP)


def matching_tokens(text: str) -> list[str]:
    """Split text into tokens for term matching.

    The text is NFKC-normalised and lowercased, and every run of characters
    outside a-z and 0-9 becomes a single separator. Stopwords are kept.
    """
    folded = unicodedata.normalize("NFKC", text).lower()
    return _NON_ALNUM.sub(" ", folded).split()
