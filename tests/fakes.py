from __future__ import annotations

import hashlib
import threading
import time
from collections.abc import Callable
from typing import Any

from deep_research.domain.models import FetchedPage, SearchResult
from deep_research.domain.urls import normalize_url


class FakeModel:
    def __init__(self, handlers: dict[str, Any | Callable[[str], Any]]) -> None:
        self.handlers = handlers
        self.calls: list[str] = []
        self._lock = threading.Lock()

    def generate_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        schema_name: str,
        schema: dict[str, Any],
        timeout_seconds: float,
    ) -> dict[str, Any]:
        del system_prompt, schema, timeout_seconds
        with self._lock:
            self.calls.append(schema_name)
        handler = self.handlers[schema_name]
        value = handler(user_prompt) if callable(handler) else handler
        if not isinstance(value, dict):
            return value
        result = value.copy()
        if schema_name == "search_queries" and "claim_frames" not in result:
            result["claim_frames"] = [
                {
                    "frame_id": f"H{index}",
                    "proposition": f"Test proposition {index} has distinct evidence.",
                }
                for index in range(1, 5)
            ]
        return result


class FakeSearch:
    def __init__(
        self,
        results: list[SearchResult] | dict[str, list[SearchResult]],
    ) -> None:
        self.results = results
        self.calls: list[str] = []

    def search(
        self, query: str, *, max_results: int, timeout_seconds: float
    ) -> list[SearchResult]:
        del timeout_seconds
        self.calls.append(query)
        results = self.results.get(query, []) if isinstance(self.results, dict) else self.results
        return results[:max_results]


class FakeFetcher:
    def __init__(self, delay: float = 0.0) -> None:
        self.delay = delay
        self.active = 0
        self.max_active = 0
        self._lock = threading.Lock()

    def fetch(self, url: str, *, timeout_seconds: float) -> FetchedPage:
        del timeout_seconds
        with self._lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            time.sleep(self.delay)
            text = f"Source text for {url}. Study reports a measurable effect."
            return FetchedPage(
                url=url,
                normalized_url=normalize_url(url),
                domain=normalize_url(url).split("/")[2],
                title="Source",
                text=text,
                content_hash=hashlib.sha256(text.encode()).hexdigest(),
            )
        finally:
            with self._lock:
                self.active -= 1
