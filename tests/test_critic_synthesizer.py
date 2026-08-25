import tempfile
import unittest
from pathlib import Path

from deep_research.audit import JsonlAuditLogger
from deep_research.budget import BudgetConfig, BudgetManager
from deep_research.critic import CriticSynthesizer
from deep_research.ledger import EvidenceLedger
from deep_research.models import (
    EvidenceObservation,
    Polarity,
    Priority,
    ResearchTask,
)
from deep_research.synthesizer import CitationError
from tests.fakes import FakeModel


def task(task_id: str = "T1") -> ResearchTask:
    return ResearchTask(
        id=task_id,
        question="Does X improve Y?",
        rationale="Core outcome",
        priority=Priority.HIGH,
        page_budget_share=0.25,
    )


class CriticSynthesizerTests(unittest.TestCase):
    def test_critic_creates_at_most_two_depth_one_followups(self) -> None:
        model = FakeModel(
            {
                "initial_critique": {
                    "coverage": [
                        {"task_id": "T1", "assessment": "weak: one source"}
                    ],
                    "contested_claim_ids": [],
                    "remaining_gaps": ["Independent replication"],
                    "followups": [
                        {
                            "parent_task_id": "T1",
                            "question": "Is there independent replication?",
                            "rationale": "Resolve weak support",
                            "priority": "high",
                            "page_budget_share": 0.5,
                        }
                    ],
                }
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            budget = BudgetManager(BudgetConfig())
            audit = JsonlAuditLogger(Path(tmp) / "events.jsonl")
            critique = CriticSynthesizer(model, budget, audit).critique(
                original_query="Does X improve Y?",
                tasks=[task()],
                claims=[],
                allow_followups=True,
            )

        self.assertEqual(1, len(critique.followup_tasks))
        self.assertEqual(1, critique.followup_tasks[0].depth)
        self.assertEqual("T1", critique.followup_tasks[0].parent_task_id)
        self.assertEqual(1, budget.snapshot().followup_tasks)

    def test_final_critique_rejects_more_recursion(self) -> None:
        model = FakeModel(
            {
                "final_critique": {
                    "coverage": [],
                    "contested_claim_ids": [],
                    "remaining_gaps": [],
                    "followups": [
                        {
                            "parent_task_id": "T1",
                            "question": "Another level?",
                            "rationale": "Should be rejected",
                            "priority": "low",
                            "page_budget_share": 0.5,
                        }
                    ],
                }
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            critic = CriticSynthesizer(
                model,
                BudgetManager(BudgetConfig()),
                JsonlAuditLogger(Path(tmp) / "events.jsonl"),
            )
            with self.assertRaises(ValueError):
                critic.critique(
                    original_query="Query",
                    tasks=[task()],
                    claims=[],
                    allow_followups=False,
                )

    def test_synthesis_rejects_unknown_source_id(self) -> None:
        model = FakeModel(
            {
                "final_report": {
                    "executive_summary": "Summary [S99].",
                    "main_findings": [],
                    "contested_findings": [],
                    "weak_evidence": [],
                    "remaining_gaps": [],
                }
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            audit = JsonlAuditLogger(Path(tmp) / "events.jsonl")
            ledger = EvidenceLedger(audit)
            ledger.add_observations(
                [
                    EvidenceObservation(
                        observation_id="O1",
                        task_id="T1",
                        source_url="https://example.com/a",
                        source_domain="example.com",
                        statement="X improves Y.",
                        polarity=Polarity.SUPPORT,
                        excerpt="Measured effect.",
                    )
                ]
            )
            critic = CriticSynthesizer(model, BudgetManager(BudgetConfig()), audit)
            with self.assertRaises(CitationError):
                critic.synthesize(
                    original_query="Query",
                    tasks=[task()],
                    claims=ledger.claims(),
                    observations=ledger.observations(),
                    remaining_gaps=[],
                )


if __name__ == "__main__":
    unittest.main()
