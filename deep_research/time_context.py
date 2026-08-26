from __future__ import annotations

from datetime import UTC, datetime


def current_utc_date() -> str:
    return datetime.now(UTC).date().isoformat()
