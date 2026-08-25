import tempfile
import unittest
from pathlib import Path

from deep_research.audit import JsonlAuditLogger
from deep_research.budget import BudgetConfig, BudgetManager
from deep_research.models import Priority, ResearchTask, SearchResult
from deep_research.planner import InvalidResearchQuery, Planner
from deep_research.researcher import FetchGate, Researcher
from deep_research.urls import UrlRegistry
from tests.fakes import FakeFetcher, FakeModel, FakeSearch


class PlannerTests(unittest.TestCase):
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

        self.assertEqual(1, budget.snapshot().pages)
        self.assertEqual([], result.observations)
        self.assertEqual(1, len(result.explorations))

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
