"""Invalid-output rules for G_fixed (paper Sec. 4.2) and the shared text-equality rule.

Decisions of 2026-09-16:

* equality (2b): strip, collapse internal whitespace, case-sensitive. Used for
  "identical to x0" here and for the controller's "revised_text == x_t" empty-rewrite check.
* non-English (3): after removing digits, punctuation and emoji (everything that is not a
  letter), the share of letters outside the Latin script must not exceed a threshold.
  Accented Latin letters count as Latin. Threshold is ``TODO(decision)`` in
  ``configs/generation.yaml``.
* refusal (1c): provider-level refusal OR a match against ``configs/refusal_phrases.yaml``.
  The list holds PREFIXES (2026-09-18): an output counts as a refusal only when it
  STARTS with one of them. Before comparing, :func:`refusal_head` maps curly apostrophes
  (U+2018 / U+2019) to ASCII ``'``, strips leading markdown markers (``* _ > # - +`` and
  the backtick), whitespace and quotation marks from the head, and lowercases (review,
  2026-09-18: a refusal typed as ``**I’m sorry**, ...`` was slipping through). Substring
  matching was rejected because it flags ordinary rewrites that merely contain such a
  phrase mid-sentence.
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import yaml

from rafr.contracts.schemas import PLACEHOLDER, InvalidReason

_WS = re.compile(r"\s+")
_QUOTES = "\"'“”‘’`"
# Curly single quotes double as apostrophes ("I’m", "can’t"): fold them so that the ASCII
# prefixes in the list match either spelling.
_APOSTROPHES = str.maketrans({"‘": "'", "’": "'", "ʼ": "'"})
# Markdown a model may wrap the head of its answer in: emphasis, block quote, heading, list
# bullet, inline code. Stripped from the HEAD only, together with whitespace and quotes.
_MARKDOWN_HEAD = "*_>#-+`"
REFUSAL_KEY = "prefixes"


def normalize_ws(text: str) -> str:
    """Strip and collapse whitespace runs to a single space ."""
    return _WS.sub(" ", text).strip()


def texts_equal(a: str, b: str) -> bool:
    return normalize_ws(a) == normalize_ws(b)


def is_latin_letter(ch: str) -> bool:
    try:
        return unicodedata.name(ch).startswith("LATIN")
    except ValueError:
        return False


def non_latin_ratio(text: str) -> tuple[float, int]:
    """Return ``(non_latin_letters / letters, letters)``; ratio is 0.0 when there are no letters."""
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return 0.0, 0
    non_latin = sum(1 for ch in letters if not is_latin_letter(ch))
    return non_latin / len(letters), len(letters)


def load_refusal_phrases(path: Path) -> list[str]:
    """Load the refusal PREFIX list (``prefixes:`` key), whitespace-normalized and lowercased.

    A file with the pre-2026-09-18 ``phrases:`` key is refused: that key carried substring
    semantics, and silently reading it as prefixes would change what the list means.
    """
    with Path(path).open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if "phrases" in data:
        raise ValueError(
            f"{path}: key 'phrases' (substring matching) is no longer supported; "
            f"use '{REFUSAL_KEY}' (prefix matching)"
        )
    phrases = data.get(REFUSAL_KEY) or []
    # The placeholder entry is not a phrase: it only keeps formal runs blocked (assert_frozen).
    # Dropping it here lets the dry run use the (empty) list without matching the word
    # PLACEHOLDER inside placeholder prompts.
    return [
        normalize_ws(normalize_apostrophes(str(p))).lower()
        for p in phrases
        if str(p).strip() and PLACEHOLDER not in str(p)
    ]


def normalize_apostrophes(text: str) -> str:
    """Map curly apostrophes (U+2018, U+2019, U+02BC) to ASCII ``'``."""
    return text.translate(_APOSTROPHES)


def refusal_head(text: str) -> str:
    """The comparable head of an output: apostrophes folded to ASCII, leading markdown
    markers / whitespace / quotation marks stripped, trailing whitespace and quotation marks
    stripped, whitespace-normalized, lowercased.

    Only the head is cleaned of markdown: a ``*`` or ``-`` later in the text is content.
    """
    s = normalize_ws(normalize_apostrophes(text))
    s = s.lstrip(_MARKDOWN_HEAD + _QUOTES + " ").rstrip(_QUOTES + " ")
    return normalize_ws(s).lower()


def matches_refusal(text: str, phrases: list[str]) -> bool:
    """True when the output STARTS with one of the refusal prefixes (see module docstring)."""
    hay = refusal_head(text)
    return any(p and hay.startswith(p) for p in phrases)


def judge_validity(
    output: str | None,
    x0: str,
    *,
    provider_refused: bool,
    phrases: list[str],
    non_english_threshold: float | None,
) -> InvalidReason | None:
    """Return the first matching invalid reason (paper Sec. 4.2 order) or ``None`` if valid.

    ``non_english_threshold=None`` disables the language rule; only the dry run may do that
    (formal steps are blocked by ``assert_frozen`` while the threshold is TBD).
    """
    if provider_refused:
        return InvalidReason.REFUSAL
    if output is None or not normalize_ws(output):
        return InvalidReason.EMPTY
    if matches_refusal(output, phrases):
        return InvalidReason.REFUSAL
    if texts_equal(output, x0):
        return InvalidReason.IDENTICAL_TO_X0
    if non_english_threshold is not None:
        ratio, n_letters = non_latin_ratio(output)
        if n_letters == 0:
            return InvalidReason.EMPTY
        if ratio > non_english_threshold:
            return InvalidReason.NON_ENGLISH
    return None
