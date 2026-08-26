import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from deep_research.audit import JsonlAuditLogger
from deep_research.budget import BudgetConfig, BudgetManager
from deep_research.models import Priority, ResearchTask, SearchResult
from deep_research.planner import InvalidResearchQuery, Planner
from deep_research.researcher import FetchGate, Researcher, _Candidate, _select_candidates
from deep_research.urls import UrlRegistry
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
            "deep_research.planner.current_utc_date", return_value="2026-08-26"
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
            "deep_research.planner.current_utc_date", return_value="2026-08-26"
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
            "deep_research.researcher.current_utc_date", return_value="2026-08-26"
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
            "deep_research.researcher.current_utc_date", return_value="2026-08-26"
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
