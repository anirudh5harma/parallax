import json
import tempfile
import unittest
from pathlib import Path

from deep_research.audit import JsonlAuditLogger
from deep_research.budget import BudgetConfig, BudgetManager
from deep_research.critic import CriticSynthesizer, _report_schema
from deep_research.ledger import EvidenceLedger
from deep_research.models import (
    EvidenceObservation,
    Polarity,
    Priority,
    ResearchTask,
)
from deep_research.synthesizer import CitationError, build_synthesis_context, render_report
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
    def test_disputed_claim_cannot_render_as_main_finding(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            audit = JsonlAuditLogger(Path(tmp) / "events.jsonl")
            ledger = EvidenceLedger(audit)
            ledger.add_observations(
                [
                    EvidenceObservation(
                        observation_id="O1", task_id="T1",
                        source_url="https://support.example/a",
                        source_domain="support.example", statement="X improves Y.",
                        polarity=Polarity.SUPPORT, excerpt="Measured effect.",
                    ),
                    EvidenceObservation(
                        observation_id="O2", task_id="T1",
                        source_url="https://contradict.example/b",
                        source_domain="contradict.example", statement="X improves Y.",
                        polarity=Polarity.CONTRADICT, excerpt="No measured effect.",
                    ),
                ]
            )
            context = build_synthesis_context(
                ledger.claims(), ledger.observations()
            )
            claim = ledger.claims()[0]
            payload = {
                "executive_summary": "Summary.",
                "main_findings": [{
                    "claim_id": claim.claim_id,
                    "synthesis": "Conflicting evidence.",
                    "source_ids": sorted(context.allowed_sources_by_claim[claim.claim_id]),
                }],
                "contested_findings": [], "weak_evidence": [], "remaining_gaps": [],
            }

            with self.assertRaisesRegex(ValueError, "disputed claims outside"):
                render_report(payload, context)

            payload["main_findings"] = []
            with self.assertRaisesRegex(ValueError, "missing from contested"):
                render_report(payload, context)

    def test_non_disputed_claim_cannot_render_as_contested(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            audit = JsonlAuditLogger(Path(tmp) / "events.jsonl")
            ledger = EvidenceLedger(audit)
            ledger.add_observations(
                [EvidenceObservation(
                    observation_id="O1", task_id="T1",
                    source_url="https://support.example/a",
                    source_domain="support.example", statement="X improves Y.",
                    polarity=Polarity.SUPPORT, excerpt="Measured effect.",
                )]
            )
            context = build_synthesis_context(ledger.claims(), ledger.observations())
            claim = ledger.claims()[0]
            payload = {
                "executive_summary": "Summary.", "main_findings": [],
                "contested_findings": [{
                    "claim_id": claim.claim_id, "synthesis": "Supported finding.",
                    "source_ids": ["S1"],
                }],
                "weak_evidence": [], "remaining_gaps": [],
            }

            with self.assertRaisesRegex(ValueError, "non-disputed claims"):
                render_report(payload, context)

    def test_synthesis_handles_empty_ledger_without_model_invention(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            audit = JsonlAuditLogger(Path(tmp) / "events.jsonl")
            report = CriticSynthesizer(
                FakeModel({}), BudgetManager(BudgetConfig()), audit
            ).synthesize(
                original_query="Query",
                tasks=[task()],
                claims=[],
                observations=[],
                remaining_gaps=["Primary studies were not recovered."],
            )

        self.assertIn("remains insufficient", report)
        self.assertIn("Primary studies were not recovered.", report)
        self.assertIn("No sources cited", report)

    def test_report_uses_canonical_final_critic_gaps(self) -> None:
        context = build_synthesis_context([], [])
        report = render_report(
            {
                "executive_summary": "No supported summary available.",
                "main_findings": [],
                "contested_findings": [],
                "weak_evidence": [],
                "remaining_gaps": ["Model paraphrase"],
            },
            context,
            remaining_gaps=["Canonical critic gap"],
        )

        self.assertIn("Canonical critic gap", report)
        self.assertNotIn("Model paraphrase", report)

    def test_report_schema_constrains_ledger_identifiers(self) -> None:
        schema = _report_schema(["C1", "C2"], ["S1", "S2"])
        for section in ("main_findings", "contested_findings", "weak_evidence"):
            properties = schema["properties"][section]["items"]["properties"]
            self.assertEqual(["C1", "C2"], properties["claim_id"]["enum"])
            self.assertEqual(["S1", "S2"], properties["source_ids"]["items"]["enum"])

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

    def test_synthesis_rejects_cross_claim_citation(self) -> None:
        model = FakeModel({})
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
                    ),
                    EvidenceObservation(
                        observation_id="O2",
                        task_id="T1",
                        source_url="https://example.org/b",
                        source_domain="example.org",
                        statement="A different claim.",
                        polarity=Polarity.SUPPORT,
                        excerpt="Different evidence.",
                    ),
                ]
            )
            target_claim = next(
                claim for claim in ledger.claims() if claim.text == "X improves Y."
            )
            model.handlers["final_report"] = {
                "executive_summary": "Summary.",
                "main_findings": [
                    {
                        "claim_id": target_claim.claim_id,
                        "synthesis": "Wrongly cites another claim [S2].",
                        "source_ids": ["S1"],
                    }
                ],
                "contested_findings": [],
                "weak_evidence": [],
                "remaining_gaps": [],
            }
            critic = CriticSynthesizer(model, BudgetManager(BudgetConfig()), audit)
            with self.assertRaises(CitationError):
                critic.synthesize(
                    original_query="Query",
                    tasks=[task()],
                    claims=ledger.claims(),
                    observations=ledger.observations(),
                    remaining_gaps=[],
                )

    def test_synthesis_repairs_one_invalid_citation_pass(self) -> None:
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
                    ),
                    EvidenceObservation(
                        observation_id="O2",
                        task_id="T1",
                        source_url="https://example.org/b",
                        source_domain="example.org",
                        statement="A different claim.",
                        polarity=Polarity.SUPPORT,
                        excerpt="Different evidence.",
                    ),
                ]
            )
            target = next(claim for claim in ledger.claims() if claim.text == "X improves Y.")
            calls = 0

            def report_payload(prompt: str) -> dict[str, object]:
                nonlocal calls
                calls += 1
                source_id = "S2" if calls == 1 else "S1"
                return {
                    "executive_summary": "Summary.",
                    "main_findings": [
                        {
                            "claim_id": target.claim_id,
                            "synthesis": "Evidence supports the finding.",
                            "source_ids": [source_id],
                        }
                    ],
                    "contested_findings": [],
                    "weak_evidence": [],
                    "remaining_gaps": [],
                }

            model = FakeModel({"final_report": report_payload})
            report = CriticSynthesizer(
                model, BudgetManager(BudgetConfig()), audit
            ).synthesize(
                original_query="Query",
                tasks=[task()],
                claims=ledger.claims(),
                observations=ledger.observations(),
                remaining_gaps=[],
            )

        self.assertEqual(2, calls)
        self.assertIn("- Sources: S1", report)

    def test_synthesis_repair_is_bounded_to_one_retry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            event_path = Path(tmp) / "events.jsonl"
            audit = JsonlAuditLogger(event_path)
            ledger = EvidenceLedger(audit)
            ledger.add_observations(
                [EvidenceObservation(
                    observation_id="O1", task_id="T1",
                    source_url="https://example.com/a", source_domain="example.com",
                    statement="X improves Y.", polarity=Polarity.SUPPORT,
                    excerpt="Measured effect.",
                )]
            )
            claim = ledger.claims()[0]
            invalid = {
                "executive_summary": "Summary.",
                "main_findings": [{
                    "claim_id": claim.claim_id,
                    "synthesis": "Uncited finding.", "source_ids": [],
                }],
                "contested_findings": [], "weak_evidence": [], "remaining_gaps": [],
            }
            model = FakeModel({"final_report": invalid})

            with self.assertRaises(CitationError):
                CriticSynthesizer(
                    model, BudgetManager(BudgetConfig()), audit
                ).synthesize(
                    original_query="Query", tasks=[task()], claims=ledger.claims(),
                    observations=ledger.observations(), remaining_gaps=[],
                )

            events = [json.loads(line) for line in event_path.read_text().splitlines()]

        self.assertEqual(2, model.calls.count("final_report"))
        self.assertEqual(
            1, sum(event["event"] == "synthesis.validation_retry" for event in events)
        )

    def test_synthesis_rejects_uncited_claim_finding(self) -> None:
        model = FakeModel({})
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
            claim = ledger.claims()[0]
            model.handlers["final_report"] = {
                "executive_summary": "Summary.",
                "main_findings": [
                    {
                        "claim_id": claim.claim_id,
                        "synthesis": "An unsupported narrative.",
                        "source_ids": [],
                    }
                ],
                "contested_findings": [],
                "weak_evidence": [],
                "remaining_gaps": [],
            }
            critic = CriticSynthesizer(model, BudgetManager(BudgetConfig()), audit)
            with self.assertRaises(CitationError):
                critic.synthesize(
                    original_query="Query",
                    tasks=[task()],
                    claims=ledger.claims(),
                    observations=ledger.observations(),
                    remaining_gaps=[],
                )

    def test_synthesis_rejects_unbound_summary_citation(self) -> None:
        model = FakeModel({})
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
            model.handlers["final_report"] = {
                "executive_summary": "Summary cites evidence without a finding [S1].",
                "main_findings": [],
                "contested_findings": [],
                "weak_evidence": [],
                "remaining_gaps": [],
            }
            critic = CriticSynthesizer(model, BudgetManager(BudgetConfig()), audit)
            with self.assertRaises(CitationError):
                critic.synthesize(
                    original_query="Query",
                    tasks=[task()],
                    claims=ledger.claims(),
                    observations=ledger.observations(),
                    remaining_gaps=[],
                )

    def test_synthesis_renders_claim_local_sources(self) -> None:
        model = FakeModel({})
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
            claim = ledger.claims()[0]
            model.handlers["final_report"] = {
                "executive_summary": "Supported summary without unscoped citations.",
                "main_findings": [
                    {
                        "claim_id": claim.claim_id,
                        "synthesis": "Evidence supports the finding.",
                        "source_ids": ["S1"],
                    }
                ],
                "contested_findings": [],
                "weak_evidence": [],
                "remaining_gaps": [],
            }
            report = CriticSynthesizer(
                model, BudgetManager(BudgetConfig()), audit
            ).synthesize(
                original_query="Query",
                tasks=[task()],
                claims=ledger.claims(),
                observations=ledger.observations(),
                remaining_gaps=[],
            )

        self.assertIn("- Sources: S1", report)


if __name__ == "__main__":
    unittest.main()
