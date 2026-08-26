import hashlib
import json
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from deep_research.agents.planner import InvalidResearchQuery, Planner
from deep_research.agents.researcher import FetchGate, Researcher, _Candidate, _select_candidates
from deep_research.domain.budget import BudgetConfig, BudgetExceeded, BudgetManager
from deep_research.domain.models import (
    FetchedPage,
    Priority,
    ResearchTask,
    SearchResult,
    TaskStatus,
)
from deep_research.domain.urls import UrlRegistry, normalize_url
from deep_research.infrastructure.audit import JsonlAuditLogger
from deep_research.infrastructure.providers import ProviderError
from tests.fakes import FakeFetcher, FakeModel, FakeSearch


class PlannerTests(unittest.TestCase):
    def test_planner_retries_a_stale_plan_for_current_query(self) -> None:
        calls = 0

        def plan(_prompt: str) -> dict[str, object]:
            nonlocal calls
            calls += 1
            period = "evolving in 2024-2025" if calls == 1 else "through August 2026"
            return {
                "disposition": "researchable",
                "reason": "Ready for research.",
                "tasks": [
                    {
                        "question": f"How did market segment {index} change {period}?",
                        "rationale": f"Track segment {index}",
                        "priority": "high",
                        "page_budget_share": 0.25,
                    }
                    for index in range(1, 5)
                ],
            }

        with tempfile.TemporaryDirectory() as tmp, patch(
            "deep_research.agents.planner.current_utc_date", return_value="2026-08-26"
        ):
            tasks = Planner(
                FakeModel({"research_plan": plan}),
                BudgetManager(BudgetConfig()),
                JsonlAuditLogger(Path(tmp) / "events.jsonl"),
            ).plan("How are global markets evolving now?")

        self.assertEqual(2, calls)
        self.assertTrue(all("2026" in task.question for task in tasks))

    def test_planner_receives_authoritative_current_date(self) -> None:
        prompts: list[str] = []
        model = FakeModel(
            {
                "research_plan": lambda prompt: (
                    prompts.append(prompt)
                    or {
                        "disposition": "researchable",
                        "reason": "Ready for research.",
                        "tasks": [
                            {
                                "question": f"Current question {index}?",
                                "rationale": f"Current rationale {index}",
                                "priority": "high",
                                "page_budget_share": 0.25,
                            }
                            for index in range(1, 5)
                        ],
                    }
                )
            }
        )
        with tempfile.TemporaryDirectory() as tmp, patch(
            "deep_research.agents.planner.current_utc_date", return_value="2026-08-26"
        ):
            Planner(
                model,
                BudgetManager(BudgetConfig()),
                JsonlAuditLogger(Path(tmp) / "events.jsonl"),
            ).plan("How are markets evolving now?")

        self.assertIn("Current date: 2026-08-26", prompts[0])

    def test_planner_creates_exactly_four_primary_tasks(self) -> None:
        model = FakeModel(
            {
                "research_plan": {
                    "disposition": "researchable",
                    "reason": "Ready for research.",
                    "tasks": [
                        {
                            "question": f"Question {index}?",
                            "rationale": f"Rationale {index}",
                            "priority": "high" if index == 1 else "medium",
                            "page_budget_share": 0.25,
                        }
                        for index in range(1, 5)
                    ]
                }
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            audit = JsonlAuditLogger(Path(tmp) / "events.jsonl")
            budget = BudgetManager(BudgetConfig())
            tasks = Planner(model, budget, audit).plan("A real research question")

        self.assertEqual(4, len(tasks))
        self.assertEqual(["T1", "T2", "T3", "T4"], [task.id for task in tasks])
        self.assertTrue(all(task.depth == 0 for task in tasks))
        self.assertEqual(4, budget.snapshot().primary_tasks)

    def test_planner_rejects_wrong_task_count(self) -> None:
        model = FakeModel(
            {
                "research_plan": {
                    "disposition": "researchable",
                    "reason": "Ready for research.",
                    "tasks": [],
                }
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            planner = Planner(
                model,
                BudgetManager(BudgetConfig()),
                JsonlAuditLogger(Path(tmp) / "events.jsonl"),
            )
            with self.assertRaises(ValueError):
                planner.plan("A real research question")

    def test_planner_rejects_non_research_input_without_reserving_tasks(self) -> None:
        model = FakeModel(
            {
                "research_plan": {
                    "disposition": "reject",
                    "reason": "Ask a specific question about a topic or outcome.",
                    "tasks": [],
                }
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            budget = BudgetManager(BudgetConfig())
            planner = Planner(
                model,
                budget,
                JsonlAuditLogger(Path(tmp) / "events.jsonl"),
            )

            with self.assertRaisesRegex(InvalidResearchQuery, "specific question"):
                planner.plan("hello")

        self.assertEqual(0, budget.snapshot().primary_tasks)


class ResearcherTests(unittest.TestCase):
    def test_batched_evidence_preserves_page_attribution_and_literal_excerpts(self) -> None:
        def batch_evidence(prompt: str) -> dict[str, object]:
            supplied = json.loads(prompt)["untrusted_pages"]
            return {
                "pages": [
                    {
                        "page_id": page["page_id"],
                        "observations": [
                            {
                                "statement": f"Finding from {page['page_id']}",
                                "claim_frame_id": "NOVEL",
                                "polarity": "support",
                                "excerpt": f"Unique evidence for {page['page_id']}.",
                                "source_type": "paper",
                            }
                        ],
                    }
                    for page in supplied
                ]
            }

        model = FakeModel({"batch_page_evidence": batch_evidence})
        pages = [
            FetchedPage(
                url=f"https://source{index}.example/report",
                normalized_url=f"https://source{index}.example/report",
                domain=f"source{index}.example",
                title="Source",
                text=f"Unique evidence for P{index + 1}.",
                content_hash=str(index),
            )
            for index in range(2)
        ]
        with tempfile.TemporaryDirectory() as tmp:
            researcher = Researcher(
                model=model,
                search=FakeSearch([]),
                fetcher=FakeFetcher(),
                budget=BudgetManager(BudgetConfig(max_pages=2)),
                audit=JsonlAuditLogger(Path(tmp) / "events.jsonl"),
                urls=UrlRegistry(),
                fetch_gate=FetchGate(2),
                evidence_batch_size=2,
            )
            outcomes = researcher._process_fetched_pages(
                ResearchTask(
                    id="T1",
                    question="What does the evidence show?",
                    rationale="Test batching",
                    priority=Priority.HIGH,
                    page_budget_share=1,
                ),
                pages,
                threading.Event(),
            )

        self.assertEqual(["batch_page_evidence"], model.calls)
        self.assertEqual(["source0.example", "source1.example"], [
            item.observations[0].source_domain for item in outcomes
        ])

    def test_batched_evidence_rejects_excerpt_from_another_page(self) -> None:
        model = FakeModel({
            "batch_page_evidence": {
                "pages": [
                    {
                        "page_id": "P1",
                        "observations": [{
                            "statement": "A misplaced finding.",
                            "claim_frame_id": "NOVEL",
                            "polarity": "support",
                            "excerpt": "Only page two contains this sentence.",
                            "source_type": "paper",
                        }],
                    },
                    {"page_id": "P2", "observations": []},
                ]
            }
        })
        pages = [
            FetchedPage(
                url=f"https://source{index}.example/report",
                normalized_url=f"https://source{index}.example/report",
                domain=f"source{index}.example",
                title="Source",
                text=(
                    "Only page one contains this sentence."
                    if index == 1
                    else "Only page two contains this sentence."
                ),
                content_hash=str(index),
            )
            for index in (1, 2)
        ]
        with tempfile.TemporaryDirectory() as tmp:
            researcher = Researcher(
                model=model,
                search=FakeSearch([]),
                fetcher=FakeFetcher(),
                budget=BudgetManager(BudgetConfig(max_pages=2)),
                audit=JsonlAuditLogger(Path(tmp) / "events.jsonl"),
                urls=UrlRegistry(),
                fetch_gate=FetchGate(2),
                evidence_batch_size=2,
            )
            result = researcher._extract_observations_batch(
                ResearchTask(
                    id="T1",
                    question="What does the evidence show?",
                    rationale="Test page isolation",
                    priority=Priority.HIGH,
                    page_budget_share=1,
                ),
                pages,
                threading.Event(),
            )

        self.assertEqual([], result[pages[0].normalized_url])

    def test_source_screening_stops_at_configured_ceiling(self) -> None:
        results = [
            SearchResult(f"https://source{index}.example/report", f"Source {index}")
            for index in range(10)
        ]
        with tempfile.TemporaryDirectory() as tmp:
            budget = BudgetManager(BudgetConfig(max_sources=3, max_pages=2))
            Researcher(
                model=FakeModel({
                    "search_queries": {
                        "queries": [{"query_text": "focused query", "rationale": "coverage"}]
                    },
                    "page_evidence": {"observations": []},
                }),
                search=FakeSearch(results),
                fetcher=FakeFetcher(),
                budget=budget,
                audit=JsonlAuditLogger(Path(tmp) / "events.jsonl"),
                urls=UrlRegistry(),
                fetch_gate=FetchGate(2),
            ).research(
                ResearchTask(
                    id="T1",
                    question="What does the evidence show?",
                    rationale="Test screening budget",
                    priority=Priority.HIGH,
                    page_budget_share=1,
                )
            )

        self.assertEqual(3, budget.snapshot().sources)
        self.assertEqual(2, budget.snapshot().pages)

    def test_claim_frame_generation_retries_duplicate_frames(self) -> None:
        calls = 0

        def queries(_prompt: str) -> dict[str, object]:
            nonlocal calls
            calls += 1
            frames = (
                [
                    {
                        "frame_id": "H1",
                        "proposition": "The same proposition is repeated across every frame.",
                    }
                    for _ in range(4)
                ]
                if calls == 1
                else [
                    {
                        "frame_id": f"H{index}",
                        "proposition": f"Distinct proposition {index} has test evidence.",
                    }
                    for index in range(1, 5)
                ]
            )
            return {
                "queries": [{"query_text": "focused query", "rationale": "coverage"}],
                "claim_frames": frames,
            }

        with tempfile.TemporaryDirectory() as tmp:
            researcher = Researcher(
                model=FakeModel({"search_queries": queries}),
                search=FakeSearch([]),
                fetcher=FakeFetcher(),
                budget=BudgetManager(BudgetConfig()),
                audit=JsonlAuditLogger(Path(tmp) / "events.jsonl"),
                urls=UrlRegistry(),
                fetch_gate=FetchGate(2),
            )
            result = researcher._generate_queries(
                ResearchTask(
                    id="T1",
                    question="What does the evidence show?",
                    rationale="Test claim frames",
                    priority=Priority.HIGH,
                    page_budget_share=1,
                ),
                threading.Event(),
                target=1,
            )

        self.assertEqual(2, calls)
        self.assertEqual(1, len(result))
        self.assertEqual(4, len(researcher._claim_frames["T1"]))

    def test_claim_frame_generation_rejects_repeated_invalid_frames(self) -> None:
        invalid = {
            "queries": [{"query_text": "focused query", "rationale": "coverage"}],
            "claim_frames": [
                {
                    "frame_id": "H1",
                    "proposition": "The same proposition is repeated across every frame.",
                }
                for _ in range(4)
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            researcher = Researcher(
                model=FakeModel({"search_queries": invalid}),
                search=FakeSearch([]),
                fetcher=FakeFetcher(),
                budget=BudgetManager(BudgetConfig()),
                audit=JsonlAuditLogger(Path(tmp) / "events.jsonl"),
                urls=UrlRegistry(),
                fetch_gate=FetchGate(2),
            )
            with self.assertRaisesRegex(ValueError, "invalid canonical claim frames"):
                researcher._generate_queries(
                    ResearchTask(
                        id="T1",
                        question="What does the evidence show?",
                        rationale="Test claim frames",
                        priority=Priority.HIGH,
                        page_budget_share=1,
                    ),
                    threading.Event(),
                    target=1,
                )

    def test_claim_frames_canonicalize_only_explicitly_matched_evidence(self) -> None:
        proposition = "AI adoption reduces entry-level software hiring through 2035."
        model = FakeModel({
            "page_evidence": lambda prompt: {
                "observations": [{
                    "statement": proposition if "matched.example" in prompt else "Source-specific paraphrase.",
                    "claim_frame_id": "H1" if "novel.example" not in prompt else "NOVEL",
                    "polarity": "support",
                    "excerpt": "Study reports a measurable effect.",
                    "source_type": "paper",
                }]
            }
        })
        with tempfile.TemporaryDirectory() as tmp:
            researcher = Researcher(
                model=model, search=FakeSearch([]), fetcher=FakeFetcher(),
                budget=BudgetManager(BudgetConfig()),
                audit=JsonlAuditLogger(Path(tmp) / "events.jsonl"),
                urls=UrlRegistry(), fetch_gate=FetchGate(2),
            )
            researcher._claim_frames["T1"] = {
                "H1": proposition
            }
            research_task = ResearchTask(
                id="T1", question="How will AI affect software hiring?",
                rationale="Forecast employment effects", priority=Priority.HIGH,
                page_budget_share=1,
            )
            matched = FetchedPage(
                url="https://matched.example/a",
                normalized_url="https://matched.example/a",
                domain="matched.example", title="Matched",
                text="Study reports a measurable effect.", content_hash="a",
            )
            novel = FetchedPage(
                url="https://novel.example/a",
                normalized_url="https://novel.example/a",
                domain="novel.example", title="Novel",
                text="Study reports a measurable effect.", content_hash="b",
            )
            mismatch = FetchedPage(
                url="https://mismatch.example/a",
                normalized_url="https://mismatch.example/a",
                domain="mismatch.example", title="Mismatch",
                text="Study reports a measurable effect.", content_hash="c",
            )
            matched_observation = researcher._extract_observations(
                research_task, matched, threading.Event()
            )[0]
            novel_observation = researcher._extract_observations(
                research_task, novel, threading.Event()
            )[0]
            mismatch_observation = researcher._extract_observations(
                research_task, mismatch, threading.Event()
            )[0]

        self.assertEqual(
            "AI adoption reduces entry-level software hiring through 2035.",
            matched_observation.statement,
        )
        self.assertEqual("Source-specific paraphrase.", novel_observation.statement)
        self.assertEqual("Source-specific paraphrase.", mismatch_observation.statement)

    def test_partial_page_failures_do_not_fail_a_productive_task(self) -> None:
        class PartialExtractor:
            def extract(
                self,
                urls: list[str],
                *,
                query: str,
                timeout_seconds: float,
            ) -> list[FetchedPage]:
                del query, timeout_seconds
                url = urls[0]
                return [
                    FetchedPage(
                        url=url,
                        normalized_url=normalize_url(url),
                        domain=normalize_url(url).split("/")[2],
                        title="Evidence",
                        text="The source reports a measurable effect.",
                        content_hash=hashlib.sha256(url.encode()).hexdigest(),
                    )
                ]

        model = FakeModel(
            {
                "search_queries": {
                    "queries": [{"query_text": "query", "rationale": "coverage"}]
                },
                "page_evidence": {"observations": []},
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            task = ResearchTask(
                id="T1",
                question="What does the evidence show?",
                rationale="Test resilient extraction",
                priority=Priority.HIGH,
                page_budget_share=1,
            )
            result = Researcher(
                model=model,
                search=FakeSearch(
                    [
                        SearchResult("https://one.example/report", "One"),
                        SearchResult("https://two.example/report", "Two"),
                    ]
                ),
                fetcher=FakeFetcher(),
                batch_extractor=PartialExtractor(),
                budget=BudgetManager(BudgetConfig(max_pages=2)),
                audit=JsonlAuditLogger(Path(tmp) / "events.jsonl"),
                urls=UrlRegistry(),
                fetch_gate=FetchGate(2),
            ).research(task)

        self.assertEqual(TaskStatus.COMPLETED, task.status)
        self.assertEqual([], result.errors)
        self.assertEqual(2, len(result.explorations))

    def test_serious_run_spends_full_task_page_share(self) -> None:
        class Extractor:
            def __init__(self) -> None:
                self.calls: list[list[str]] = []

            def extract(
                self,
                urls: list[str],
                *,
                query: str,
                timeout_seconds: float,
            ) -> list[FetchedPage]:
                del query, timeout_seconds
                self.calls.append(urls)
                return [
                    FetchedPage(
                        url=url,
                        normalized_url=normalize_url(url),
                        domain=normalize_url(url).split("/")[2],
                        title="",
                        text=f"Focused evidence from {url}",
                        content_hash=hashlib.sha256(url.encode()).hexdigest(),
                    )
                    for url in urls
                ]

        extractor = Extractor()
        fetcher = FakeFetcher()
        results = {
            f"distinct query {query_index}": [
                SearchResult(
                    f"https://source{query_index}-{index}.example/report/{index}",
                    f"Research report {query_index}-{index}",
                    "Detailed evidence " * 20,
                )
                for index in range(15)
            ]
            for query_index in range(15)
        }
        with tempfile.TemporaryDirectory() as tmp:
            budget = BudgetManager(BudgetConfig.serious())
            Researcher(
                model=FakeModel(
                    {
                        "search_queries": {
                            "queries": [
                                {
                                    "query_text": f"distinct query {index}",
                                    "rationale": "broad coverage",
                                }
                                for index in range(15)
                            ]
                        },
                        "page_evidence": {"observations": []},
                    }
                ),
                search=FakeSearch(results),
                fetcher=fetcher,
                batch_extractor=extractor,
                budget=budget,
                audit=JsonlAuditLogger(Path(tmp) / "events.jsonl"),
                urls=UrlRegistry(),
                fetch_gate=FetchGate(12),
            ).research(
                ResearchTask(
                    id="T1",
                    question="What does broad evidence show?",
                    rationale="Screen broadly, then read strong sources",
                    priority=Priority.HIGH,
                    page_budget_share=0.25,
                )
            )

        self.assertEqual([12] * 12 + [6], [len(batch) for batch in extractor.calls])
        self.assertEqual(150, budget.snapshot().pages)
        self.assertEqual(0, fetcher.max_active)
        query_angles = {
            int(url.split("source", 1)[1].split("-", 1)[0])
            for batch in extractor.calls
            for url in batch
        }
        self.assertEqual(set(range(15)), query_angles)

    def test_batch_extract_gate_enforces_weighted_global_capacity(self) -> None:
        class SlowExtractor:
            def __init__(self) -> None:
                self.active = 0
                self.max_active = 0
                self.lock = threading.Lock()

            def extract(
                self,
                urls: list[str],
                *,
                query: str,
                timeout_seconds: float,
            ) -> list[FetchedPage]:
                del query, timeout_seconds
                with self.lock:
                    self.active += len(urls)
                    self.max_active = max(self.max_active, self.active)
                try:
                    time.sleep(0.02)
                    return []
                finally:
                    with self.lock:
                        self.active -= len(urls)

        extractor = SlowExtractor()
        budget = BudgetManager(
            BudgetConfig(max_pages=4, max_concurrent_fetches=2)
        )
        gate = FetchGate(2)
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(
                    gate.extract,
                    extractor,
                    [f"https://source{index}.example/a", f"https://source{index}.example/b"],
                    query="question",
                    budget=budget,
                )
                for index in range(2)
            ]
            for future in futures:
                future.result()

        self.assertEqual(2, extractor.max_active)

    def test_model_slot_wait_rechecks_wall_clock_deadline(self) -> None:
        class SlowModel:
            def __init__(self) -> None:
                self.calls = 0

            def generate_json(self, **kwargs: object) -> dict[str, object]:
                del kwargs
                self.calls += 1
                time.sleep(0.04)
                return {}

        model = SlowModel()
        with tempfile.TemporaryDirectory() as tmp:
            researcher = Researcher(
                model=model,
                search=FakeSearch([]),
                fetcher=FakeFetcher(),
                budget=BudgetManager(
                    BudgetConfig(
                        max_concurrent_fetches=1,
                        wall_clock_timeout_seconds=0.02,
                    )
                ),
                audit=JsonlAuditLogger(Path(tmp) / "events.jsonl"),
                urls=UrlRegistry(),
                fetch_gate=FetchGate(1),
            )
            with ThreadPoolExecutor(max_workers=2) as executor:
                first = executor.submit(researcher._generate_json, timeout_seconds=1)
                time.sleep(0.005)
                second = executor.submit(researcher._generate_json, timeout_seconds=1)
                first.result()
                with self.assertRaises(BudgetExceeded):
                    second.result()

        self.assertEqual(1, model.calls)

    def test_all_evidence_model_failures_fail_the_task(self) -> None:
        def fail_evidence(_prompt: str) -> dict[str, object]:
            raise ProviderError(
                "model unavailable",
                code="bedrock_unavailable",
                public_message="Bedrock is temporarily unavailable.",
            )

        with tempfile.TemporaryDirectory() as tmp:
            task = ResearchTask(
                id="T1",
                question="What does the evidence show?",
                rationale="Validate extraction failure handling",
                priority=Priority.HIGH,
                page_budget_share=1,
            )
            result = Researcher(
                model=FakeModel(
                    {
                        "search_queries": {
                            "queries": [{"query_text": "query", "rationale": "coverage"}]
                        },
                        "page_evidence": fail_evidence,
                    }
                ),
                search=FakeSearch([SearchResult("https://example.com/a", "A")]),
                fetcher=FakeFetcher(),
                budget=BudgetManager(BudgetConfig(max_pages=1)),
                audit=JsonlAuditLogger(Path(tmp) / "events.jsonl"),
                urls=UrlRegistry(),
                fetch_gate=FetchGate(2),
            ).research(task)

        self.assertEqual(TaskStatus.FAILED, task.status)
        self.assertEqual("bedrock_unavailable", result.error_code)
        self.assertEqual(
            [
                "1 source evidence extractions failed: "
                "Bedrock is temporarily unavailable."
            ],
            result.errors,
        )

    def test_duplicate_skip_does_not_hide_all_model_failures(self) -> None:
        class DuplicateExtractor:
            def extract(
                self,
                urls: list[str],
                *,
                query: str,
                timeout_seconds: float,
            ) -> list[FetchedPage]:
                del query, timeout_seconds
                return [
                    FetchedPage(
                        url=url,
                        normalized_url=normalize_url(url),
                        domain=normalize_url(url).split("/")[2],
                        title="Duplicate",
                        text="Identical evidence text.",
                        content_hash="same-content",
                    )
                    for url in urls
                ]

        def deny(_prompt: str) -> dict[str, object]:
            raise ProviderError(
                "denied",
                retryable=False,
                code="bedrock_access_denied",
                public_message="Model access is unavailable.",
            )

        with tempfile.TemporaryDirectory() as tmp:
            task = ResearchTask(
                id="T1",
                question="What does duplicated evidence show?",
                rationale="Test duplicate failure accounting",
                priority=Priority.HIGH,
                page_budget_share=1,
            )
            result = Researcher(
                model=FakeModel(
                    {
                        "search_queries": {
                            "queries": [{"query_text": "query", "rationale": "coverage"}]
                        },
                        "page_evidence": deny,
                    }
                ),
                search=FakeSearch(
                    [
                        SearchResult("https://one.example/a", "One"),
                        SearchResult("https://two.example/a", "Two"),
                    ]
                ),
                fetcher=FakeFetcher(),
                batch_extractor=DuplicateExtractor(),
                budget=BudgetManager(BudgetConfig(max_pages=2, max_concurrent_fetches=2)),
                audit=JsonlAuditLogger(Path(tmp) / "events.jsonl"),
                urls=UrlRegistry(),
                fetch_gate=FetchGate(2),
            ).research(task)

        self.assertEqual(TaskStatus.FAILED, task.status)
        self.assertEqual("bedrock_access_denied", result.error_code)
        self.assertIn("Model access is unavailable.", result.errors[0])

    def test_partial_model_access_failure_is_reported_without_failing_task(self) -> None:
        def evidence(prompt: str) -> dict[str, object]:
            if "bad.example" in prompt:
                raise ProviderError(
                    "denied",
                    retryable=False,
                    code="bedrock_access_denied",
                    public_message="Model access is unavailable.",
                )
            return {"observations": []}

        with tempfile.TemporaryDirectory() as tmp:
            task = ResearchTask(
                id="T1",
                question="What does mixed evidence show?",
                rationale="Test partial failure accounting",
                priority=Priority.HIGH,
                page_budget_share=1,
            )
            result = Researcher(
                model=FakeModel(
                    {
                        "search_queries": {
                            "queries": [{"query_text": "query", "rationale": "coverage"}]
                        },
                        "page_evidence": evidence,
                    }
                ),
                search=FakeSearch(
                    [
                        SearchResult("https://good.example/a", "Good"),
                        SearchResult("https://bad.example/a", "Bad"),
                    ]
                ),
                fetcher=FakeFetcher(),
                budget=BudgetManager(BudgetConfig(max_pages=2)),
                audit=JsonlAuditLogger(Path(tmp) / "events.jsonl"),
                urls=UrlRegistry(),
                fetch_gate=FetchGate(2),
            ).research(task)

        self.assertEqual(TaskStatus.COMPLETED, task.status)
        self.assertEqual("bedrock_access_denied", result.error_code)
        self.assertEqual(1, len(result.errors))

    def test_candidate_selection_prefers_primary_and_diverse_domains(self) -> None:
        candidates = [
            _Candidate(
                result=SearchResult(f"https://blog.example/{index}", "Opinion", "short"),
                normalized_url=f"https://blog.example/{index}",
                domain="blog.example",
                discovery_order=index,
            )
            for index in range(5)
        ]
        candidates.append(
            _Candidate(
                result=SearchResult(
                    "https://agency.gov/research/report.pdf",
                    "Annual research report",
                    "Detailed official evidence " * 20,
                ),
                normalized_url="https://agency.gov/research/report.pdf",
                domain="agency.gov",
                discovery_order=5,
            )
        )

        selected = _select_candidates(candidates, 4)

        self.assertEqual("agency.gov", selected[0].domain)
        self.assertEqual(3, sum(item.domain == "blog.example" for item in selected))

    def test_candidate_selection_demotes_social_and_commercial_forecasts(self) -> None:
        candidates = [
            _Candidate(
                result=SearchResult(
                    "https://www.linkedin.com/posts/example",
                    "Industry commentary",
                    "Long commentary " * 20,
                ),
                normalized_url="https://www.linkedin.com/posts/example",
                domain="www.linkedin.com",
                discovery_order=0,
            ),
            _Candidate(
                result=SearchResult(
                    "https://forecast.example/report",
                    "Market Size, Market Share and CAGR Forecast Report",
                    "Commercial forecast " * 20,
                ),
                normalized_url="https://forecast.example/report",
                domain="forecast.example",
                discovery_order=1,
            ),
            _Candidate(
                result=SearchResult(
                    "https://company.example/investor/filing",
                    "Quarterly earnings filing",
                    "Direct company disclosure " * 20,
                ),
                normalized_url="https://company.example/investor/filing",
                domain="company.example",
                discovery_order=2,
            ),
        ]

        selected = _select_candidates(candidates, 3)

        self.assertEqual(["company.example"], [item.domain for item in selected])

    def test_search_generation_retries_when_current_anchor_is_missing(self) -> None:
        calls = 0

        def queries(_prompt: str) -> dict[str, object]:
            nonlocal calls
            calls += 1
            query = (
                "market evolving evidence 2024-2025"
                if calls == 1
                else "latest market evidence 2026"
            )
            return {"queries": [{"query_text": query, "rationale": "freshness"}]}

        with tempfile.TemporaryDirectory() as tmp, patch(
            "deep_research.agents.researcher.current_utc_date", return_value="2026-08-26"
        ):
            Researcher(
                model=FakeModel(
                    {"search_queries": queries, "page_evidence": {"observations": []}}
                ),
                search=FakeSearch([]),
                fetcher=FakeFetcher(),
                budget=BudgetManager(BudgetConfig(max_pages=1)),
                audit=JsonlAuditLogger(Path(tmp) / "events.jsonl"),
                urls=UrlRegistry(),
                fetch_gate=FetchGate(2),
            ).research(
                ResearchTask(
                    id="T1",
                    question="What is changing in markets now?",
                    rationale="Find current evidence",
                    priority=Priority.HIGH,
                    page_budget_share=1,
                )
            )

        self.assertEqual(2, calls)

    def test_search_generation_receives_authoritative_current_date(self) -> None:
        prompts: list[str] = []
        model = FakeModel(
            {
                "search_queries": lambda prompt: (
                    prompts.append(prompt)
                    or {
                        "queries": [
                            {"query_text": "current evidence", "rationale": "freshness"}
                        ]
                    }
                ),
                "page_evidence": {"observations": []},
            }
        )
        with tempfile.TemporaryDirectory() as tmp, patch(
            "deep_research.agents.researcher.current_utc_date", return_value="2026-08-26"
        ):
            Researcher(
                model=model,
                search=FakeSearch([]),
                fetcher=FakeFetcher(),
                budget=BudgetManager(BudgetConfig(max_pages=1)),
                audit=JsonlAuditLogger(Path(tmp) / "events.jsonl"),
                urls=UrlRegistry(),
                fetch_gate=FetchGate(2),
            ).research(
                ResearchTask(
                    id="T1",
                    question="What changed recently?",
                    rationale="Find current evidence",
                    priority=Priority.HIGH,
                    page_budget_share=1,
                )
            )

        self.assertIn("Current date: 2026-08-26", prompts[0])

    def test_serious_run_generates_fifteen_queries_per_primary_task(self) -> None:
        model = FakeModel(
            {
                "search_queries": {
                    "queries": [
                        {
                            "query_text": f"distinct query {index}",
                            "rationale": "broader source coverage",
                        }
                        for index in range(15)
                    ]
                },
                "page_evidence": {"observations": []},
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            budget = BudgetManager(BudgetConfig.serious())
            researcher = Researcher(
                model=model,
                search=FakeSearch([]),
                fetcher=FakeFetcher(),
                budget=budget,
                audit=JsonlAuditLogger(Path(tmp) / "events.jsonl"),
                urls=UrlRegistry(),
                fetch_gate=FetchGate(2),
            )
            researcher.research(
                ResearchTask(
                    id="T1",
                    question="What does broad evidence show?",
                    rationale="Cover distinct evidence paths",
                    priority=Priority.HIGH,
                    page_budget_share=0.25,
                )
            )

        self.assertEqual(15, budget.snapshot().searches)

    def test_skips_duplicate_search_queries(self) -> None:
        model = FakeModel(
            {
                "search_queries": {
                    "queries": [
                        {"query_text": "same query", "rationale": "one angle"},
                        {"query_text": " SAME   QUERY ", "rationale": "duplicate angle"},
                    ]
                },
                "page_evidence": {"observations": []},
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            researcher = Researcher(
                model=model,
                search=FakeSearch([]),
                fetcher=FakeFetcher(),
                budget=BudgetManager(BudgetConfig()),
                audit=JsonlAuditLogger(Path(tmp) / "events.jsonl"),
                urls=UrlRegistry(),
                fetch_gate=FetchGate(2),
            )
            result = researcher.research(
                ResearchTask(
                    id="T1",
                    question="What does the evidence show?",
                    rationale="Cover distinct evidence paths",
                    priority=Priority.HIGH,
                    page_budget_share=1,
                )
            )

        self.assertEqual([], result.errors)
        self.assertEqual(1, researcher.budget.snapshot().searches)

    def test_serious_run_rejects_fewer_than_fifteen_distinct_queries(self) -> None:
        duplicate_queries = [
            {"query_text": f"distinct query {index}", "rationale": "one angle"}
            for index in range(14)
        ]
        duplicate_queries.append(
            {"query_text": " DISTINCT   QUERY 0 ", "rationale": "duplicate angle"}
        )
        model = FakeModel(
            {
                "search_queries": {"queries": duplicate_queries},
                "page_evidence": {"observations": []},
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            researcher = Researcher(
                model=model,
                search=FakeSearch([]),
                fetcher=FakeFetcher(),
                budget=BudgetManager(BudgetConfig.serious()),
                audit=JsonlAuditLogger(Path(tmp) / "events.jsonl"),
                urls=UrlRegistry(),
                fetch_gate=FetchGate(2),
            )
            with self.assertRaisesRegex(ValueError, "14 distinct queries"):
                researcher.research(
                    ResearchTask(
                        id="T1",
                        question="What does broad evidence show?",
                        rationale="Cover distinct evidence paths",
                        priority=Priority.HIGH,
                        page_budget_share=0.25,
                    )
                )
        self.assertEqual(2, model.calls.count("search_queries"))

    def test_page_content_is_delimited_as_untrusted_data(self) -> None:
        prompts: list[str] = []
        model = FakeModel(
            {
                "search_queries": {
                    "queries": [{"query_text": "query", "rationale": "coverage"}]
                },
                "page_evidence": lambda prompt: (
                    prompts.append(prompt) or {"observations": []}
                ),
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            researcher = Researcher(
                model=model,
                search=FakeSearch([SearchResult("https://example.com/a", "A")]),
                fetcher=FakeFetcher(),
                budget=BudgetManager(BudgetConfig()),
                audit=JsonlAuditLogger(Path(tmp) / "events.jsonl"),
                urls=UrlRegistry(),
                fetch_gate=FetchGate(2),
            )
            researcher.research(
                ResearchTask(
                    id="T1",
                    question="Does X improve Y?",
                    rationale="Core outcome",
                    priority=Priority.HIGH,
                    page_budget_share=1,
                )
            )

        prompt = json.loads(prompts[0])
        self.assertIn("untrusted_page", prompt)
        self.assertIn("text", prompt["untrusted_page"])
    def test_deduplicates_fetches_and_rejects_nonliteral_excerpt(self) -> None:
        model = FakeModel(
            {
                "search_queries": {
                    "queries": [
                        {"query_text": "query one", "rationale": "coverage"},
                        {"query_text": "query two", "rationale": "cross-check"},
                    ]
                },
                "page_evidence": {
                    "observations": [
                        {
                            "statement": "Intervention X improves outcome Y.",
                            "polarity": "support",
                            "excerpt": "not present in source",
                            "source_type": "paper",
                        }
                    ]
                },
            }
        )
        search = FakeSearch(
            [
                SearchResult("https://example.com/a", "A"),
                SearchResult("https://example.com/a?utm_source=x", "A duplicate"),
            ]
        )
        fetcher = FakeFetcher()
        with tempfile.TemporaryDirectory() as tmp:
            audit = JsonlAuditLogger(Path(tmp) / "events.jsonl")
            budget = BudgetManager(BudgetConfig(max_pages=5))
            researcher = Researcher(
                model=model,
                search=search,
                fetcher=fetcher,
                budget=budget,
                audit=audit,
                urls=UrlRegistry(),
                fetch_gate=FetchGate(2),
            )
            task = ResearchTask(
                id="T1",
                question="Does X improve Y?",
                rationale="Core outcome",
                priority=Priority.HIGH,
                page_budget_share=1,
            )
            result = researcher.research(task)
            records = [json.loads(line) for line in audit.path.read_text().splitlines()]

        self.assertEqual(1, budget.snapshot().pages)
        self.assertEqual([], result.observations)
        self.assertEqual(1, len(result.explorations))

        self.assertEqual(1, sum(record["event"] == "page.explored" for record in records))

    def test_rejects_excerpt_with_changed_case(self) -> None:
        model = FakeModel(
            {
                "search_queries": {
                    "queries": [{"query_text": "query", "rationale": "coverage"}]
                },
                "page_evidence": {
                    "observations": [
                        {
                            "statement": "The source reports an effect.",
                            "polarity": "support",
                            "excerpt": "source text for https://example.com/a.",
                            "source_type": "other",
                        }
                    ]
                },
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            researcher = Researcher(
                model=model,
                search=FakeSearch([SearchResult("https://example.com/a", "A")]),
                fetcher=FakeFetcher(),
                budget=BudgetManager(BudgetConfig()),
                audit=JsonlAuditLogger(Path(tmp) / "events.jsonl"),
                urls=UrlRegistry(),
                fetch_gate=FetchGate(2),
            )
            result = researcher.research(
                ResearchTask(
                    id="T1",
                    question="Does the source report an effect?",
                    rationale="Check literal evidence",
                    priority=Priority.HIGH,
                    page_budget_share=1,
                )
            )

        self.assertEqual([], result.observations)

    def test_fetch_concurrency_obeys_shared_gate(self) -> None:
        model = FakeModel(
            {
                "search_queries": {
                    "queries": [{"query_text": "query", "rationale": "coverage"}]
                },
                "page_evidence": {"observations": []},
            }
        )
        search = FakeSearch(
            [SearchResult(f"https://site{index}.example/a", str(index)) for index in range(6)]
        )
        fetcher = FakeFetcher(delay=0.02)
        with tempfile.TemporaryDirectory() as tmp:
            budget = BudgetManager(
                BudgetConfig(max_pages=6, max_concurrent_fetches=2)
            )
            researcher = Researcher(
                model=model,
                search=search,
                fetcher=fetcher,
                budget=budget,
                audit=JsonlAuditLogger(Path(tmp) / "events.jsonl"),
                urls=UrlRegistry(),
                fetch_gate=FetchGate(2),
            )
            researcher.research(
                ResearchTask(
                    id="T1",
                    question="Question?",
                    rationale="Rationale",
                    priority=Priority.HIGH,
                    page_budget_share=1,
                )
            )

        self.assertLessEqual(fetcher.max_active, 2)
        self.assertGreaterEqual(fetcher.max_active, 2)


if __name__ == "__main__":
    unittest.main()
