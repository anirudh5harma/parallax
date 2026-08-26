import json
import stat
import tempfile
import unittest
from pathlib import Path

from deep_research.app import _completion_status, create_plan, run_query
from deep_research.budget import BudgetConfig
from deep_research.models import (
    EvidenceObservation,
    Polarity,
    Priority,
    ResearchResult,
    ResearchTask,
    SearchResult,
    TaskStatus,
)
from deep_research.providers import ProviderError
from tests.fakes import FakeFetcher, FakeModel, FakeSearch


class AppTests(unittest.TestCase):
    def test_plan_failure_writes_safe_provider_error(self) -> None:
        failure = ProviderError(
            "provider HTTP error: 432: internal detail",
            retryable=False,
            status=432,
            code="tavily_quota_exhausted",
            public_message="The Tavily usage limit is exhausted.",
        )
        def fail(_prompt: str) -> dict[str, object]:
            raise failure

        model = FakeModel({"research_plan": fail})
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "plan.json"
            with self.assertRaises(ProviderError):
                create_plan(
                    query="A bounded research question",
                    output_path=output,
                    config=BudgetConfig(),
                    model=model,
                )
            payload = json.loads(output.read_text())

        self.assertEqual("failed", payload["status"])
        self.assertEqual("tavily_quota_exhausted", payload["error_code"])
        self.assertEqual("tavily", payload["error_provider"])
        self.assertFalse(payload["error_retryable"])
        self.assertEqual("The Tavily usage limit is exhausted.", payload["error"])
        self.assertNotIn("internal detail", payload["error"])

    def test_partial_evidence_survives_nonfatal_task_errors(self) -> None:
        tasks = [
            ResearchTask(
                id=f"T{index}",
                question=f"Question {index}?",
                rationale="Coverage",
                priority=Priority.HIGH,
                page_budget_share=0.25,
                status=TaskStatus.FAILED,
            )
            for index in range(1, 5)
        ]
        observation = EvidenceObservation(
            observation_id="O1",
            task_id="T1",
            source_url="https://example.com/a",
            source_domain="example.com",
            statement="A testable claim has evidence.",
            polarity=Polarity.SUPPORT,
            excerpt="A testable claim has evidence.",
        )
        results = [
            ResearchResult(
                task_id=task.id,
                observations=[observation] if task.id == "T1" else [],
                explorations=[],
                errors=["one source failed"],
            )
            for task in tasks
        ]

        self.assertEqual("completed_with_errors", _completion_status(tasks, results))

    def test_run_writes_auditable_artifacts(self) -> None:
        model = FakeModel(
            {
                "research_plan": {
                    "disposition": "researchable",
                    "reason": "Ready for research.",
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
                "final_report": lambda prompt: self._report_for_prompt(prompt),
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
            modes = {
                "directory": stat.S_IMODE(artifacts.run_dir.stat().st_mode),
                "report": stat.S_IMODE(artifacts.report_path.stat().st_mode),
                "ledger": stat.S_IMODE(artifacts.ledger_path.stat().st_mode),
                "events": stat.S_IMODE(artifacts.events_path.stat().st_mode),
                "run": stat.S_IMODE(artifacts.run_path.stat().st_mode),
            }

        self.assertEqual(1, len(ledger["claims"]))
        self.assertEqual("completed", run["status"])
        self.assertIn("run.completed", event_names)
        self.assertIn("page.explored", event_names)
        self.assertEqual(0o700, modes.pop("directory"))
        self.assertEqual({0o600}, set(modes.values()))

    @staticmethod
    def _report_for_prompt(prompt: str) -> dict[str, object]:
        context = json.loads(prompt)
        claim = context["structured_claims"][0]
        source_id = claim["support"][0]["source_id"]
        return {
            "executive_summary": "One source reports an effect.",
            "main_findings": [
                {
                    "claim_id": claim["claim_id"],
                    "synthesis": f"Evidence reports an effect [{source_id}].",
                    "source_ids": [source_id],
                }
            ],
            "contested_findings": [],
            "weak_evidence": [],
            "remaining_gaps": ["Long-term outcomes"],
        }


if __name__ == "__main__":
    unittest.main()
