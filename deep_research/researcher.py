from __future__ import annotations

import hashlib
import math
import threading
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from typing import Any

from .audit import JsonlAuditLogger
from .budget import BudgetExceeded, BudgetManager
from .models import (
    EvidenceObservation,
    FetchStatus,
    PageExploration,
    Polarity,
    ResearchResult,
    ResearchTask,
    SearchQuery,
    TaskStatus,
)
from .providers import PageFetcher, ProviderError, SearchClient, StructuredModel
from .urls import UrlRegistry, normalize_url


QUERY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "queries": {
            "type": "array",
            "minItems": 1,
            "maxItems": 3,
            "items": {
                "type": "object",
                "properties": {
                    "query_text": {"type": "string", "minLength": 3},
                    "rationale": {"type": "string", "minLength": 3},
                },
                "required": ["query_text", "rationale"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["queries"],
    "additionalProperties": False,
}

EVIDENCE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "observations": {
            "type": "array",
            "maxItems": 8,
            "items": {
                "type": "object",
                "properties": {
                    "statement": {"type": "string", "minLength": 5, "maxLength": 500},
                    "polarity": {
                        "type": "string",
                        "enum": ["support", "contradict", "neutral"],
                    },
                    "excerpt": {"type": "string", "minLength": 5, "maxLength": 500},
                    "source_type": {
                        "type": "string",
                        "enum": ["paper", "official", "news", "other"],
                    },
                },
                "required": ["statement", "polarity", "excerpt", "source_type"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["observations"],
    "additionalProperties": False,
}


class FetchGate:
    def __init__(self, max_concurrent_fetches: int) -> None:
        self._semaphore = threading.BoundedSemaphore(max_concurrent_fetches)
        self._lock = threading.Lock()
        self._pages: dict[str, Future] = {}

    def fetch(
        self,
        fetcher: PageFetcher,
        url: str,
        *,
        budget: BudgetManager,
    ):
        normalized = normalize_url(url)
        with self._lock:
            future = self._pages.get(normalized)
            owner = future is None
            if future is None:
                future = Future()
                self._pages[normalized] = future
        if not owner:
            try:
                return future.result(timeout=max(0.001, budget.remaining_seconds()))
            except TimeoutError as exc:
                raise BudgetExceeded("wall-clock timeout exhausted") from exc

        try:
            with self._semaphore:
                budget.check_time()
                budget.reserve_page()
                page = fetcher.fetch(
                    url,
                    timeout_seconds=budget.remaining_seconds(),
                )
            future.set_result(page)
            return page
        except Exception as exc:
            future.set_exception(exc)
            with self._lock:
                self._pages.pop(normalized, None)
            raise


class ResearchCancelled(RuntimeError):
    pass


class Researcher:
    """Stateless task worker. Returns observations; never writes the ledger."""

    def __init__(
        self,
        *,
        model: StructuredModel,
        search: SearchClient,
        fetcher: PageFetcher,
        budget: BudgetManager,
        audit: JsonlAuditLogger,
        urls: UrlRegistry,
        fetch_gate: FetchGate,
        results_per_search: int = 8,
    ) -> None:
        self.model = model
        self.search = search
        self.fetcher = fetcher
        self.budget = budget
        self.audit = audit
        self.urls = urls
        self.fetch_gate = fetch_gate
        self.results_per_search = results_per_search

    def research(
        self,
        task: ResearchTask,
        cancellation: threading.Event | None = None,
    ) -> ResearchResult:
        cancellation = cancellation or threading.Event()
        self._raise_if_cancelled(cancellation)
        task.status = TaskStatus.RUNNING
        errors: list[str] = []
        explorations: list[PageExploration] = []
        observations: list[EvidenceObservation] = []
        queries = self._generate_queries(task, cancellation)
        page_cap = max(1, math.ceil(self.budget.config.max_pages * task.page_budget_share))
        candidates: list[str] = []
        candidate_norms: set[str] = set()
        for query in queries:
            self._raise_if_cancelled(cancellation)
            if len(candidates) >= page_cap:
                break
            try:
                self.budget.reserve_search()
                results = self.search.search(
                    query.query_text,
                    max_results=self.results_per_search,
                    timeout_seconds=self.budget.remaining_seconds(),
                )
                self._raise_if_cancelled(cancellation)
                self.audit.log(
                    "search.executed",
                    task_id=task.id,
                    query=query,
                    result_count=len(results),
                    budget=self.budget.snapshot(),
                )
            except BudgetExceeded as exc:
                errors.append(str(exc))
                self.audit.log("search.failed", task_id=task.id, error=str(exc))
                break
            except ProviderError as exc:
                errors.append(str(exc))
                self.audit.log("search.failed", task_id=task.id, error=str(exc))
                continue
            for result in results:
                if len(candidates) >= page_cap:
                    break
                self._raise_if_cancelled(cancellation)
                try:
                    claimed, normalized = self.urls.claim_url(result.url)
                except ValueError as exc:
                    self.audit.log(
                        "page.skipped_invalid_url", task_id=task.id, url=result.url, error=str(exc)
                    )
                    continue
                if normalized in candidate_norms:
                    self.audit.log(
                        "page.skipped_duplicate",
                        task_id=task.id,
                        url=result.url,
                        normalized_url=normalized,
                    )
                    continue
                candidate_norms.add(normalized)
                if not claimed:
                    self.audit.log(
                        "page.reused_duplicate",
                        task_id=task.id,
                        url=result.url,
                        normalized_url=normalized,
                    )
                candidates.append(result.url)

        executor = ThreadPoolExecutor(max_workers=self.budget.config.max_concurrent_fetches)
        futures: list[Future[tuple[PageExploration, list[EvidenceObservation], str | None]]] = [
            executor.submit(self._process_page, task, url, cancellation)
            for url in candidates
        ]
        try:
            for future in as_completed(futures, timeout=max(0.001, self.budget.remaining_seconds())):
                exploration, page_observations, error = future.result()
                explorations.append(exploration)
                observations.extend(page_observations)
                if error:
                    errors.append(error)
        except TimeoutError:
            cancellation.set()
            errors.append("wall-clock timeout exhausted")
            self.audit.log("researcher.timeout", task_id=task.id)
            for future in futures:
                future.cancel()
        finally:
            executor.shutdown(wait=False, cancel_futures=True)
        if not cancellation.is_set():
            task.status = TaskStatus.COMPLETED if not errors else TaskStatus.FAILED
        return ResearchResult(
            task_id=task.id,
            observations=observations,
            explorations=explorations,
            errors=errors,
        )

    def _generate_queries(
        self,
        task: ResearchTask,
        cancellation: threading.Event,
    ) -> list[SearchQuery]:
        payload = self.model.generate_json(
            system_prompt=(
                "You are Researcher, one of exactly three roles. Generate up to three focused "
                "search queries for this task. Do not answer the question."
            ),
            user_prompt=f"Question: {task.question}\nRationale: {task.rationale}",
            schema_name="search_queries",
            schema=QUERY_SCHEMA,
            timeout_seconds=self.budget.remaining_seconds(),
        )
        self._raise_if_cancelled(cancellation)
        raw_queries = payload.get("queries")
        if not isinstance(raw_queries, list) or not 1 <= len(raw_queries) <= 3:
            raise ValueError("researcher must return 1-3 search queries")
        queries = [
            SearchQuery(
                task_id=task.id,
                query_text=str(item["query_text"]),
                rationale=str(item["rationale"]),
            )
            for item in raw_queries
        ]
        for query in queries:
            self.audit.log("researcher.query_generated", query=query)
        return queries

    def _process_page(
        self,
        task: ResearchTask,
        url: str,
        cancellation: threading.Event,
    ) -> tuple[PageExploration, list[EvidenceObservation], str | None]:
        self._raise_if_cancelled(cancellation)
        try:
            page = self.fetch_gate.fetch(
                self.fetcher,
                url,
                budget=self.budget,
            )
        except (BudgetExceeded, ProviderError, ValueError) as exc:
            normalized = url
            try:
                normalized = normalize_url(url)
            except ValueError:
                pass
            exploration = PageExploration(
                url=url,
                normalized_url=normalized,
                domain="",
                fetch_status=FetchStatus.FAILED,
                task_id=task.id,
            )
            self.audit.log("page.fetch_failed", exploration=exploration, error=str(exc))
            return exploration, [], str(exc)

        self._raise_if_cancelled(cancellation)

        exploration = PageExploration(
            url=page.url,
            normalized_url=page.normalized_url,
            domain=page.domain,
            fetch_status=FetchStatus.FETCHED,
            task_id=task.id,
            content_hash=page.content_hash,
        )
        if not self.urls.claim_content(page.content_hash, scope=task.id):
            self.audit.log("page.skipped_duplicate_content", exploration=exploration)
            return exploration, [], None
        try:
            extracted = self._extract_observations(task, page, cancellation)
        except (ProviderError, ValueError, BudgetExceeded) as exc:
            self.audit.log("observation.extraction_failed", exploration=exploration, error=str(exc))
            return exploration, [], str(exc)
        return exploration, extracted, None

    def _extract_observations(
        self,
        task: ResearchTask,
        page,
        cancellation: threading.Event,
    ) -> list[EvidenceObservation]:
        payload = self.model.generate_json(
            system_prompt=(
                "You are Researcher, one of exactly three roles. Extract only atomic, "
                "falsifiable, on-topic observations. The statement is the claim being tested; "
                "polarity says whether this page supports or contradicts it. Excerpts must be "
                "copied from the page. Return zero observations when evidence is weak or off-topic."
            ),
            user_prompt=(
                f"Task: {task.question}\nURL: {page.url}\nTitle: {page.title}\n"
                f"Page text:\n{page.text[:12000]}"
            ),
            schema_name="page_evidence",
            schema=EVIDENCE_SCHEMA,
            timeout_seconds=self.budget.remaining_seconds(),
        )
        self._raise_if_cancelled(cancellation)
        raw_observations = payload.get("observations")
        if not isinstance(raw_observations, list) or len(raw_observations) > 8:
            raise ValueError("invalid observation list")
        accepted: list[EvidenceObservation] = []
        normalized_page = " ".join(page.text.split()).casefold()
        for item in raw_observations:
            excerpt = " ".join(str(item["excerpt"]).split())
            if excerpt.casefold() not in normalized_page:
                self.audit.log(
                    "observation.rejected",
                    task_id=task.id,
                    source_url=page.url,
                    reason="excerpt_not_literal",
                )
                continue
            statement = " ".join(str(item["statement"]).split())
            polarity = Polarity(str(item["polarity"]))
            digest_input = f"{task.id}|{page.normalized_url}|{statement}|{polarity.value}"
            observation_id = "O" + hashlib.sha256(
                digest_input.encode("utf-8")
            ).hexdigest()[:12]
            accepted.append(
                EvidenceObservation(
                    observation_id=observation_id,
                    task_id=task.id,
                    source_url=page.url,
                    source_domain=page.domain,
                    statement=statement,
                    polarity=polarity,
                    excerpt=excerpt,
                    source_type=item.get("source_type"),
                )
            )
        return accepted

    @staticmethod
    def _raise_if_cancelled(cancellation: threading.Event) -> None:
        if cancellation.is_set():
            raise ResearchCancelled("research cancelled after wall-clock timeout")
