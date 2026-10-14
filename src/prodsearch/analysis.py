"""Text analysis shared by indexing and querying: fold, split, drop stopwords, strip plurals.

Product titles are short and plural-heavy ("dining chairs", "bar stools"), so a plural-only
S-stemmer (Harman, 1991) is used instead of Porter, which over-conflates catalogue words
("organic" and "organizer" both become "organ"). One rule is added to Harman's three: plurals of
sibilant stems ("benches", "boxes", "glasses") lose the whole "es".
"""

from __future__ import annotations

import re
import unicodedata

_TOKEN = re.compile(r"[a-z0-9]+")

# Lucene's default English stop set.
STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "but",
        "by",
        "for",
        "if",
        "in",
        "into",
        "is",
        "it",
        "no",
        "not",
        "of",
        "on",
        "or",
        "such",
        "that",
        "the",
        "their",
        "then",
        "there",
        "these",
        "they",
        "this",
        "to",
        "was",
        "will",
        "with",
    }
)

_SIBILANT_PLURALS = ("ches", "shes", "sses", "xes", "zzes")


def s_stem(token: str) -> str:
    if len(token) <= 3 or not token.endswith("s"):
        return token
    if token.endswith("ies") and not token.endswith(("eies", "aies")):
        return token[:-3] + "y"
    if token.endswith(_SIBILANT_PLURALS):
        return token[:-2]
    if token.endswith("es") and not token.endswith(("aes", "ees", "oes")):
        return token[:-1]
    if token.endswith(("us", "ss")):
        return token
    return token[:-1]


def fold(text: str) -> str:
    """Lower-case and strip accents ("Décor" and "decor" must meet)."""
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii").lower()


def analyze(text: str) -> list[str]:
    return [s_stem(t) for t in _TOKEN.findall(fold(text)) if t not in STOPWORDS]
