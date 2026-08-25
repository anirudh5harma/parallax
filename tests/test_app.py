import json
import tempfile
import unittest
from pathlib import Path

from deep_research.app import run_query
from deep_research.budget import BudgetConfig
from deep_research.models import SearchResult
from tests.fakes import FakeFetcher, FakeModel, FakeSearch


class AppTests(unittest.TestCase):
    def test_run_writes_auditable_artifacts(self) -> None:
        model = FakeModel(
            {
                "research_plan": {
                    "tasks": [
                        {
                            "question": f"Question {index}?",
                            "rationale": f"Rationale {index}",
                            "priority": "medium",
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
                    "remaining_gaps": [],
                    "followups": [],
                },
                "final_critique": {
                    "coverage": [],
                    "contested_claim_ids": [],
                    "remaining_gaps": ["Long-term outcomes"],
                    "followups": [],
                },
                "final_report": {
                    "executive_summary": "One source reports an effect [S1].",
                    "main_findings": [],
                    "contested_findings": [],
                    "weak_evidence": [],
                    "remaining_gaps": ["Long-term outcomes"],
                },
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            artifacts = run_query(
                query="Does intervention X improve outcome Y?",
                output_root=Path(tmp),
                config=BudgetConfig(max_pages=5),
                model=model,
                search=FakeSearch([SearchResult("https://example.com/a", "A")]),
                fetcher=FakeFetcher(),
            )

            self.assertTrue(artifacts.report_path.exists())
            self.assertTrue(artifacts.ledger_path.exists())
            self.assertTrue(artifacts.events_path.exists())
            self.assertTrue(artifacts.run_path.exists())
            ledger = json.loads(artifacts.ledger_path.read_text())
            run = json.loads(artifacts.run_path.read_text())
            event_names = [
                json.loads(line)["event"]
                for line in artifacts.events_path.read_text().splitlines()
            ]

        self.assertEqual(1, len(ledger["claims"]))
        self.assertEqual("completed", run["status"])
        self.assertIn("run.completed", event_names)


if __name__ == "__main__":
    unittest.main()
