import tempfile
import unittest
from pathlib import Path

from deep_research.audit import JsonlAuditLogger
from deep_research.budget import BudgetConfig, BudgetManager
from deep_research.critic import CriticSynthesizer
from deep_research.ledger import EvidenceLedger
from deep_research.models import SearchResult
from deep_research.pipeline import ResearchPipeline
from deep_research.planner import Planner
from deep_research.researcher import FetchGate, Researcher
from deep_research.urls import UrlRegistry
from tests.fakes import FakeFetcher, FakeModel, FakeSearch


class PipelineTests(unittest.TestCase):
    def test_runs_primary_followup_final_check_and_synthesis(self) -> None:
        model = FakeModel(
            {
                "research_plan": {
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
                "search_queries": {
                    "queries": [{"query_text": "focused query", "rationale": "coverage"}]
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
                "final_report": {
                    "executive_summary": "Evidence supports a measurable effect [S1].",
                    "main_findings": [],
                    "contested_findings": [],
                    "weak_evidence": [],
                    "remaining_gaps": ["Long-term evidence"],
                },
            }
        )
        search = FakeSearch([SearchResult("https://example.com/a", "A")])
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
        self.assertIn("## Sources", result.report)
        self.assertIn("https://example.com/a", result.report)
        self.assertEqual(1, budget.snapshot().followup_tasks)
        self.assertEqual(1, model.calls.count("final_critique"))


if __name__ == "__main__":
    unittest.main()
