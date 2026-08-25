from __future__ import annotations

import json
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .models import to_primitive


class JsonlAuditLogger:
    """Thread-safe append-only event log."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def log(self, event: str, **data: Any) -> None:
        record = {
            "timestamp": datetime.now(UTC).isoformat(),
            "event": event,
            "data": to_primitive(data),
        }
        line = json.dumps(record, ensure_ascii=False, sort_keys=True)
        with self._lock, self.path.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")
