import json
import tempfile
import threading
import unittest
from pathlib import Path

from deep_research.agents.critic import CriticSynthesizer
from deep_research.agents.planner import Planner
from deep_research.agents.researcher import FetchGate, Researcher
from deep_research.application.pipeline import ResearchPipeline
from deep_research.domain.budget import BudgetConfig, BudgetManager
from deep_research.domain.ledger import EvidenceLedger
from deep_research.domain.models import Priority, ResearchResult, ResearchTask, SearchResult
from deep_research.domain.urls import UrlRegistry
from deep_research.infrastructure.audit import JsonlAuditLogger
from tests.fakes import FakeFetcher, FakeModel, FakeSearch


class PipelineTests(unittest.TestCase):
    def test_runs_primary_followup_final_check_and_synthesis(self) -> None:
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
                },
                "search_queries": lambda prompt: {
                    "queries": [
                        {
                            "query_text": (
                                "followup query"
                                if "independent source" in prompt.casefold()
                                else "focused query"
                            ),
                            "rationale": "coverage",
                        }
                    ]
                },
                "page_evidence": {
                    "observations": [
                        {
                            "statement": "Intervention X improves outcome Y.",
                            "polarity": "support",
                            "excerpt": "Study reports a measurable effect.",
                            "source_type": "paper",
                        }
                    ]
                },
                "initial_critique": {
                    "coverage": [],
                    "contested_claim_ids": [],
                    "remaining_gaps": ["Independent source"],
                    "followups": [
                        {
                            "parent_task_id": "T1",
                            "question": "Find an independent source?",
                            "rationale": "Strengthen evidence",
                            "priority": "high",
                            "page_budget_share": 0.2,
                        }
                    ],
                },
                "final_critique": {
                    "coverage": [],
                    "contested_claim_ids": [],
                    "remaining_gaps": ["Long-term evidence"],
                    "followups": [],
                },
                "final_report": lambda prompt: self._report_for_prompt(prompt),
            }
        )
        search = FakeSearch(
            {
                "focused query": [SearchResult("https://example.com/a", "A")],
                "followup query": [SearchResult("https://example.org/b", "B")],
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            audit = JsonlAuditLogger(Path(tmp) / "events.jsonl")
            budget = BudgetManager(BudgetConfig(max_pages=10))
            pipeline = ResearchPipeline(
                planner=Planner(model, budget, audit),
                researcher=Researcher(
                    model=model,
                    search=search,
                    fetcher=FakeFetcher(),
                    budget=budget,
                    audit=audit,
                    urls=UrlRegistry(),
                    fetch_gate=FetchGate(2),
                ),
                critic=CriticSynthesizer(model, budget, audit),
                ledger=EvidenceLedger(audit),
                budget=budget,
                audit=audit,
            )
            result = pipeline.run("Does intervention X improve outcome Y?")

        self.assertEqual(5, len(result.tasks))
        self.assertEqual(1, len(result.ledger.claims()))
        self.assertEqual(5, len(result.ledger.claims()[0].supporting_observations))
        self.assertEqual(2, budget.snapshot().pages)
        self.assertIn("F1", result.ledger.claims()[0].task_ids)
        self.assertIn("followup query", search.calls)
        self.assertIn("## Sources", result.report)
        self.assertIn("https://example.com/a", result.report)
        self.assertEqual(1, budget.snapshot().followup_tasks)
        self.assertEqual(1, model.calls.count("final_critique"))

    def test_primary_tasks_run_in_parallel(self) -> None:
        class OverlapResearcher:
            def __init__(self) -> None:
                self.barrier = threading.Barrier(4)
                self.active = 0
                self.max_active = 0
                self.lock = threading.Lock()

            def research(self, task, cancellation=None):
                del cancellation
                with self.lock:
                    self.active += 1
                    self.max_active = max(self.max_active, self.active)
                self.barrier.wait(timeout=1)
                with self.lock:
                    self.active -= 1
                return ResearchResult(task.id, [], [], [])

        researcher = OverlapResearcher()
        with tempfile.TemporaryDirectory() as tmp:
            audit = JsonlAuditLogger(Path(tmp) / "events.jsonl")
            budget = BudgetManager(BudgetConfig())
            pipeline = ResearchPipeline(
                planner=None,  # type: ignore[arg-type]
                researcher=researcher,  # type: ignore[arg-type]
                critic=None,  # type: ignore[arg-type]
                ledger=EvidenceLedger(audit),
                budget=budget,
                audit=audit,
            )
            tasks = [
                ResearchTask(
                    id=f"T{index}",
                    question=f"Question {index}?",
                    rationale="Coverage",
                    priority=Priority.MEDIUM,
                    page_budget_share=0.25,
                )
                for index in range(1, 5)
            ]
            pipeline._run_tasks(tasks)

        self.assertEqual(4, researcher.max_active)

    def test_timeout_cancels_late_shared_state_mutation(self) -> None:
        class BlockingSearch:
            def __init__(self) -> None:
                self.started = threading.Event()
                self.release = threading.Event()

            def search(self, query, *, max_results, timeout_seconds):
                del query, max_results, timeout_seconds
                self.started.set()
                self.release.wait(timeout=1)
                return [SearchResult("https://late.example/a", "Late")]

        class RecordingResearcher(Researcher):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.done = threading.Event()

            def research(self, task, cancellation=None):
                try:
                    return super().research(task, cancellation=cancellation)
                finally:
                    self.done.set()

        model = FakeModel(
            {
                "search_queries": {
                    "queries": [{"query_text": "query", "rationale": "coverage"}]
                },
                "page_evidence": {"observations": []},
            }
        )
        search = BlockingSearch()
        with tempfile.TemporaryDirectory() as tmp:
            events_path = Path(tmp) / "events.jsonl"
            audit = JsonlAuditLogger(events_path)
            budget = BudgetManager(BudgetConfig(wall_clock_timeout_seconds=0.02))
            urls = UrlRegistry()
            researcher = RecordingResearcher(
                model=model,
                search=search,
                fetcher=FakeFetcher(),
                budget=budget,
                audit=audit,
                urls=urls,
                fetch_gate=FetchGate(1),
            )
            pipeline = ResearchPipeline(
                planner=None,  # type: ignore[arg-type]
                researcher=researcher,
                critic=None,  # type: ignore[arg-type]
                ledger=EvidenceLedger(audit),
                budget=budget,
                audit=audit,
            )
            timed_task = ResearchTask(
                id="T1",
                question="Question?",
                rationale="Coverage",
                priority=Priority.HIGH,
                page_budget_share=1,
            )
            pipeline._run_tasks([timed_task])
            self.assertTrue(search.started.is_set())
            search.release.set()
            self.assertTrue(researcher.done.wait(timeout=1))
            event_names = [
                json.loads(line)["event"] for line in events_path.read_text().splitlines()
            ]

        self.assertEqual("failed", timed_task.status.value)
        self.assertEqual({}, urls.domain_counts())
        self.assertNotIn("search.executed", event_names)

    @staticmethod
    def _report_for_prompt(prompt: str) -> dict[str, object]:
        context = json.loads(prompt)
        claim = context["structured_claims"][0]
        source_id = claim["support"][0]["source_id"]
        return {
            "executive_summary": "Evidence supports a measurable effect.",
            "main_findings": [
                {
                    "claim_id": claim["claim_id"],
                    "synthesis": f"Independent sources report an effect [{source_id}].",
                    "source_ids": [source_id],
                }
            ],
            "contested_findings": [],
            "weak_evidence": [],
            "remaining_gaps": ["Long-term evidence"],
        }


if __name__ == "__main__":
    unittest.main()
