from __future__ import annotations

import re
from datetime import UTC, datetime


def current_utc_date() -> str:
    return datetime.now(UTC).date().isoformat()


_CURRENT_TERMS = re.compile(
    r"\b(?:current(?:ly)?|latest|today|now|recent(?:ly)?|present|ongoing|"
    r"evolving|outlook|next\s+(?:year|years|month|months|quarter|quarters))\b",
    re.IGNORECASE,
)
_OUTPUT_CURRENT_TERMS = re.compile(
    r"\b(?:current(?:ly)?|latest|today|now|recent(?:ly)?|present|ongoing)\b",
    re.IGNORECASE,
)
_YEAR = re.compile(r"\b(20\d{2})\b")


def requires_current_evidence(text: str, current_date: str) -> bool:
    """Return whether a question asks for evidence current to this run."""
    return current_date[:4] in text or bool(_CURRENT_TERMS.search(text))


def has_current_anchor(text: str, current_date: str) -> bool:
    """Return whether generated work explicitly preserves a current-time anchor."""
    current_year = int(current_date[:4])
    years = [int(value) for value in _YEAR.findall(text)]
    if current_year in years:
        return True
    if years and max(years) < current_year:
        return False
    return bool(_OUTPUT_CURRENT_TERMS.search(text))
