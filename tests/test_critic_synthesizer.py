import json
import tempfile
import unittest
from pathlib import Path

from deep_research.agents.critic import CriticSynthesizer, _report_schema
from deep_research.agents.synthesizer import (
    CitationError,
    build_fallback_report_payload,
    build_synthesis_context,
    contextualize_remaining_gaps,
    render_report,
)
from deep_research.domain.budget import BudgetConfig, BudgetManager
from deep_research.domain.ledger import EvidenceLedger
from deep_research.domain.models import (
    EvidenceObservation,
    Polarity,
    Priority,
    ResearchTask,
)
from deep_research.infrastructure.audit import JsonlAuditLogger
from deep_research.infrastructure.providers import ProviderError
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
    def test_contested_synthesis_removes_duplicated_clause_starters(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ledger = EvidenceLedger(JsonlAuditLogger(Path(tmp) / "events.jsonl"))
            statement = "The productivity benefit differs by developer experience."
            ledger.add_observations(
                [
                    EvidenceObservation(
                        observation_id="O1",
                        task_id="T1",
                        source_url="https://support.example/a",
                        source_domain="support.example",
                        statement=statement,
                        polarity=Polarity.SUPPORT,
                        excerpt="Junior developers completed tasks faster.",
                    ),
                    EvidenceObservation(
                        observation_id="O2",
                        task_id="T1",
                        source_url="https://contradict.example/a",
                        source_domain="contradict.example",
                        statement=statement,
                        polarity=Polarity.CONTRADICT,
                        excerpt="Senior developers gained as much as junior developers.",
                    ),
                ]
            )
            context = build_synthesis_context(ledger.claims(), ledger.observations())
            claim_id = context.packet[0]["claim_id"]
            payload = {
                "executive_summary": "Summary.",
                "main_findings": [],
                "contested_findings": [{
                    "claim_id": claim_id,
                    "synthesis": (
                        "Evidence points in different directions on whether The "
                        "productivity benefit differs by developer experience."
                    ),
                    "source_ids": sorted(context.allowed_sources_by_claim[str(claim_id)]),
                }],
                "weak_evidence": [],
                "remaining_gaps": [],
            }

            report = render_report(payload, context, remaining_gaps=[])

        self.assertIn("on whether the productivity benefit", report)
        self.assertNotIn("on whether The", report)

    def test_contested_synthesis_preserves_conditional_when_clause(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ledger = EvidenceLedger(JsonlAuditLogger(Path(tmp) / "events.jsonl"))
            statement = "When interest rates rise, housing demand falls."
            ledger.add_observations(
                [
                    EvidenceObservation(
                        observation_id="O1",
                        task_id="T1",
                        source_url="https://support.example/a",
                        source_domain="support.example",
                        statement=statement,
                        polarity=Polarity.SUPPORT,
                        excerpt="Demand fell after rates rose.",
                    ),
                    EvidenceObservation(
                        observation_id="O2",
                        task_id="T1",
                        source_url="https://contradict.example/a",
                        source_domain="contradict.example",
                        statement=statement,
                        polarity=Polarity.CONTRADICT,
                        excerpt="Demand held after rates rose.",
                    ),
                ]
            )
            context = build_synthesis_context(ledger.claims(), ledger.observations())
            claim_id = context.packet[0]["claim_id"]
            payload = {
                "executive_summary": "Summary.",
                "main_findings": [],
                "contested_findings": [{
                    "claim_id": claim_id,
                    "synthesis": (
                        "Evidence points in different directions on whether When "
                        "interest rates rise, housing demand falls."
                    ),
                    "source_ids": sorted(context.allowed_sources_by_claim[str(claim_id)]),
                }],
                "weak_evidence": [],
                "remaining_gaps": [],
            }

            report = render_report(payload, context, remaining_gaps=[])

        self.assertIn(
            "Evidence differs on this proposition: When interest rates rise", report
        )

    def test_synthesis_packet_balances_planner_task_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ledger = EvidenceLedger(JsonlAuditLogger(Path(tmp) / "events.jsonl"))
            observations: list[EvidenceObservation] = []
            for claim_index in range(25):
                statement = f"Highly supported task-one claim {claim_index}."
                for domain_index in range(3):
                    observations.append(
                        EvidenceObservation(
                            observation_id=f"OT1-{claim_index}-{domain_index}",
                            task_id="T1",
                            source_url=(
                                f"https://t1-{claim_index}-{domain_index}.example/a"
                            ),
                            source_domain=f"t1-{claim_index}-{domain_index}.example",
                            statement=statement,
                            polarity=Polarity.SUPPORT,
                            excerpt="Measured result.",
                        )
                    )
            for task_id in ("T2", "T3", "T4"):
                observations.append(
                    EvidenceObservation(
                        observation_id=f"O{task_id}",
                        task_id=task_id,
                        source_url=f"https://{task_id.casefold()}.example/a",
                        source_domain=f"{task_id.casefold()}.example",
                        statement=f"Lower-confidence finding for {task_id}.",
                        polarity=Polarity.SUPPORT,
                        excerpt="Preliminary result.",
                    )
                )
            ledger.add_observations(observations)
            context = build_synthesis_context(ledger.claims(), ledger.observations())

        covered_tasks = {
            task_id
            for claim in context.claims_by_id.values()
            for task_id in claim.task_ids
        }
        self.assertEqual(24, len(context.packet))
        self.assertTrue({"T1", "T2", "T3", "T4"}.issubset(covered_tasks))

    def test_synthesis_packet_prefers_distinct_source_domains(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ledger = EvidenceLedger(JsonlAuditLogger(Path(tmp) / "events.jsonl"))
            statement = "A result is supported across distinct domains."
            observations = [
                EvidenceObservation(
                    observation_id=f"O{index}",
                    task_id="T1",
                    source_url=url,
                    source_domain=domain,
                    statement=statement,
                    polarity=Polarity.SUPPORT,
                    excerpt="Measured result.",
                )
                for index, (url, domain) in enumerate(
                    [
                        ("https://same.example/a", "same.example"),
                        ("https://same.example/b", "same.example"),
                        ("https://same.example/c", "same.example"),
                        ("https://other.example/a", "other.example"),
                        ("https://third.example/a", "third.example"),
                    ]
                )
            ]
            ledger.add_observations(observations)
            context = build_synthesis_context(ledger.claims(), ledger.observations())

        selected_urls = {
            context.source_urls[str(item["source_id"])]
            for item in context.packet[0]["support"]
        }
        self.assertEqual(
            {
                "https://same.example/a",
                "https://other.example/a",
                "https://third.example/a",
            },
            selected_urls,
        )

    def test_synthesis_packet_prefers_primary_source_types(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ledger = EvidenceLedger(JsonlAuditLogger(Path(tmp) / "events.jsonl"))
            statement = "A result is supported by several source types."
            observations = [
                EvidenceObservation(
                    observation_id=f"O{index}",
                    task_id="T1",
                    source_url=url,
                    source_domain=domain,
                    statement=statement,
                    polarity=Polarity.SUPPORT,
                    excerpt="Measured result.",
                    source_type=source_type,
                )
                for index, (url, domain, source_type) in enumerate(
                    [
                        ("https://blog.example/a", "blog.example", "other"),
                        ("https://news.example/a", "news.example", "news"),
                        ("https://study.example/a", "study.example", "paper"),
                        ("https://agency.gov/a", "agency.gov", "official"),
                    ]
                )
            ]
            ledger.add_observations(observations)
            context = build_synthesis_context(
                ledger.claims(), ledger.observations(), max_sources_per_polarity=3
            )

        selected_urls = [
            context.source_urls[str(item["source_id"])]
            for item in context.packet[0]["support"]
        ]
        self.assertEqual(
            [
                "https://agency.gov/a",
                "https://study.example/a",
                "https://news.example/a",
            ],
            selected_urls,
        )

    def test_synthesis_provider_timeout_returns_cited_fallback(self) -> None:
        def timeout(_prompt: str) -> dict[str, object]:
            raise ProviderError(
                "request failed: TimeoutError",
                code="bedrock_unavailable",
                public_message="Bedrock is temporarily unavailable.",
            )

        with tempfile.TemporaryDirectory() as tmp:
            audit_path = Path(tmp) / "events.jsonl"
            audit = JsonlAuditLogger(audit_path)
            ledger = EvidenceLedger(audit)
            ledger.add_observations(
                [
                    EvidenceObservation(
                        observation_id="O1",
                        task_id="T1",
                        source_url="https://support.example/a",
                        source_domain="support.example",
                        statement="AI adoption changes software hiring.",
                        polarity=Polarity.SUPPORT,
                        excerpt="AI adoption changed software hiring.",
                    )
                ]
            )
            report = CriticSynthesizer(
                FakeModel({"final_report": timeout}),
                BudgetManager(BudgetConfig()),
                audit,
            ).synthesize(
                original_query="How will AI change tech?",
                tasks=[task()],
                claims=ledger.claims(),
                observations=ledger.observations(),
                remaining_gaps=["Long-term evidence remains limited."],
            )
            events = [json.loads(line) for line in audit_path.read_text().splitlines()]

        self.assertIn("AI adoption changes software hiring.", report)
        self.assertIn("https://support.example/a", report)
        self.assertEqual(
            1,
            sum(event["event"] == "synthesis.fallback_rendered" for event in events),
        )
        self.assertEqual(
            1,
            sum(event["event"] == "synthesis.completed" for event in events),
        )

    def test_synthesis_retry_timeout_returns_cited_fallback(self) -> None:
        calls = 0
        with tempfile.TemporaryDirectory() as tmp:
            audit_path = Path(tmp) / "events.jsonl"
            audit = JsonlAuditLogger(audit_path)
            ledger = EvidenceLedger(audit)
            ledger.add_observations(
                [
                    EvidenceObservation(
                        observation_id="O1",
                        task_id="T1",
                        source_url="https://support.example/a",
                        source_domain="support.example",
                        statement="AI adoption changes software hiring.",
                        polarity=Polarity.SUPPORT,
                        excerpt="AI adoption changed software hiring.",
                    )
                ]
            )
            claim = ledger.claims()[0]

            def invalid_then_timeout(_prompt: str) -> dict[str, object]:
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise ProviderError(
                        "request failed: TimeoutError",
                        code="bedrock_unavailable",
                        public_message="Bedrock is temporarily unavailable.",
                    )
                return {
                    "executive_summary": "Summary.",
                    "main_findings": [],
                    "contested_findings": [
                        {
                            "claim_id": claim.claim_id,
                            "synthesis": "Incorrectly contested.",
                            "source_ids": ["S1"],
                        }
                    ],
                    "weak_evidence": [],
                    "remaining_gaps": [],
                }

            report = CriticSynthesizer(
                FakeModel({"final_report": invalid_then_timeout}),
                BudgetManager(BudgetConfig()),
                audit,
            ).synthesize(
                original_query="How will AI change tech?",
                tasks=[task()],
                claims=ledger.claims(),
                observations=ledger.observations(),
                remaining_gaps=[],
            )
            events = [json.loads(line) for line in audit_path.read_text().splitlines()]

        self.assertEqual(2, calls)
        self.assertIn("AI adoption changes software hiring.", report)
        self.assertIn("https://support.example/a", report)
        self.assertEqual(
            1,
            sum(event["event"] == "synthesis.fallback_rendered" for event in events),
        )

    def test_synthesis_packet_caps_contested_claims_for_bounded_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ledger = EvidenceLedger(JsonlAuditLogger(Path(tmp) / "events.jsonl"))
            observations = []
            for index in range(10):
                statement = f"Contested claim {index}."
                observations.extend(
                    [
                        EvidenceObservation(
                            observation_id=f"OS{index}", task_id="T1",
                            source_url=f"https://support{index}.example/a",
                            source_domain=f"support{index}.example", statement=statement,
                            polarity=Polarity.SUPPORT, excerpt="Measured effect.",
                        ),
                        EvidenceObservation(
                            observation_id=f"OC{index}", task_id="T1",
                            source_url=f"https://contradict{index}.example/b",
                            source_domain=f"contradict{index}.example", statement=statement,
                            polarity=Polarity.CONTRADICT, excerpt="No measured effect.",
                        ),
                    ]
                )
            ledger.add_observations(observations)
            context = build_synthesis_context(ledger.claims(), ledger.observations())
            gaps = contextualize_remaining_gaps(
                context,
                ["gap " * 2_000, "gap two", "gap three", "gap four", "gap five", "gap six"],
            )
            fallback = build_fallback_report_payload(context, gaps)
            report = render_report(
                fallback,
                context,
                remaining_gaps=fallback["remaining_gaps"],
            )

        self.assertEqual(6, len(fallback["contested_findings"]))
        self.assertEqual(4, context.omitted_contested_count)
        self.assertEqual(6, len(gaps))
        self.assertIn("4 additional contested findings", gaps[-1])
        self.assertIn("## Contested Findings", report)
        self.assertIn("4 additional contested findings", report)
        self.assertLessEqual(len(report.split()), 1_800)

    def test_fallback_bounds_long_contested_claim_titles(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ledger = EvidenceLedger(JsonlAuditLogger(Path(tmp) / "events.jsonl"))
            observations = []
            for index in range(6):
                statement = f"Claim {index} " + "evidence " * 55
                for prefix, polarity in (("support", Polarity.SUPPORT), ("oppose", Polarity.CONTRADICT)):
                    observations.append(
                        EvidenceObservation(
                            observation_id=f"O{prefix}{index}", task_id="T1",
                            source_url=f"https://{prefix}{index}.example/a",
                            source_domain=f"{prefix}{index}.example", statement=statement,
                            polarity=polarity, excerpt="Measured result.",
                        )
                    )
            ledger.add_observations(observations)
            context = build_synthesis_context(ledger.claims(), ledger.observations())
            fallback = build_fallback_report_payload(context, [])
            render_report(
                fallback,
                context,
                remaining_gaps=fallback["remaining_gaps"],
            )

        self.assertLess(len(fallback["contested_findings"]), 6)

    def test_neutral_only_fallback_does_not_assert_the_proposition(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ledger = EvidenceLedger(JsonlAuditLogger(Path(tmp) / "events.jsonl"))
            ledger.add_observations([
                EvidenceObservation(
                    observation_id="O1", task_id="T1",
                    source_url="https://neutral.example/a",
                    source_domain="neutral.example",
                    statement="Intervention X improves outcome Y.",
                    polarity=Polarity.NEUTRAL,
                    excerpt="The study did not evaluate effectiveness.",
                )
            ])
            context = build_synthesis_context(ledger.claims(), ledger.observations())
            fallback = build_fallback_report_payload(context, [])
            report = render_report(
                fallback,
                context,
                remaining_gaps=fallback["remaining_gaps"],
            )

        self.assertIn(
            "Evidence is insufficient to establish whether Intervention X improves outcome Y.",
            report,
        )
        self.assertNotIn(
            "## Executive Summary\n\nIntervention X improves outcome Y.",
            report,
        )
        self.assertLessEqual(len(report.split()), 1_800)

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

    def test_main_finding_cannot_cite_non_supporting_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ledger = EvidenceLedger(JsonlAuditLogger(Path(tmp) / "events.jsonl"))
            ledger.add_observations([
                EvidenceObservation(
                    observation_id="O1", task_id="T1",
                    source_url="https://support.example/a",
                    source_domain="support.example", statement="X improves Y.",
                    polarity=Polarity.SUPPORT, excerpt="Measured effect.",
                ),
                EvidenceObservation(
                    observation_id="O2", task_id="T1",
                    source_url="https://neutral.example/a",
                    source_domain="neutral.example", statement="X improves Y.",
                    polarity=Polarity.NEUTRAL, excerpt="The outcome was not evaluated.",
                )
            ])
            context = build_synthesis_context(ledger.claims(), ledger.observations())
            claim = ledger.claims()[0]
            payload = {
                "executive_summary": "Ignored.",
                "main_findings": [{
                    "claim_id": claim.claim_id,
                    "synthesis": "X improves Y.",
                    "source_ids": ["S1"],
                }],
                "contested_findings": [], "weak_evidence": [], "remaining_gaps": [],
            }

            with self.assertRaisesRegex(CitationError, "do not support"):
                render_report(payload, context)

    def test_contested_finding_must_cite_both_available_polarities(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ledger = EvidenceLedger(JsonlAuditLogger(Path(tmp) / "events.jsonl"))
            ledger.add_observations([
                EvidenceObservation(
                    observation_id="O1", task_id="T1",
                    source_url="https://support.example/a",
                    source_domain="support.example", statement="X improves Y.",
                    polarity=Polarity.SUPPORT, excerpt="Measured effect.",
                ),
                EvidenceObservation(
                    observation_id="O2", task_id="T1",
                    source_url="https://contradict.example/a",
                    source_domain="contradict.example", statement="X improves Y.",
                    polarity=Polarity.CONTRADICT, excerpt="No measured effect.",
                ),
            ])
            context = build_synthesis_context(ledger.claims(), ledger.observations())
            claim = ledger.claims()[0]
            support_id = next(iter(context.supporting_sources_by_claim[claim.claim_id]))
            payload = {
                "executive_summary": "Ignored.", "main_findings": [],
                "contested_findings": [{
                    "claim_id": claim.claim_id,
                    "synthesis": "Evidence disagrees.",
                    "source_ids": [support_id],
                }],
                "weak_evidence": [], "remaining_gaps": [],
            }

            with self.assertRaisesRegex(CitationError, "contradicting evidence"):
                render_report(payload, context)

    def test_critic_provider_failure_preserves_collected_evidence(self) -> None:
        def unavailable(_prompt: str) -> dict[str, object]:
            raise ProviderError("temporary failure", code="bedrock_unavailable")

        with tempfile.TemporaryDirectory() as tmp:
            audit_path = Path(tmp) / "events.jsonl"
            critic = CriticSynthesizer(
                FakeModel({"initial_critique": unavailable}),
                BudgetManager(BudgetConfig()),
                JsonlAuditLogger(audit_path),
            )
            critique = critic.critique(
                original_query="Query",
                tasks=[task()],
                claims=[],
                allow_followups=True,
            )

        self.assertEqual([], critique.followup_tasks)
        self.assertIn("Available evidence", critique.remaining_gaps[0])
        self.assertIn("T1", critique.coverage_by_task)

    def test_final_critic_failure_discloses_unavailable_coverage_review(self) -> None:
        def unavailable(_prompt: str) -> dict[str, object]:
            raise ProviderError("temporary failure", code="bedrock_unavailable")

        with tempfile.TemporaryDirectory() as tmp:
            audit = JsonlAuditLogger(Path(tmp) / "events.jsonl")
            ledger = EvidenceLedger(audit)
            ledger.add_observations([
                EvidenceObservation(
                    observation_id="O1", task_id="T1",
                    source_url="https://support.example/a",
                    source_domain="support.example", statement="X improves Y.",
                    polarity=Polarity.SUPPORT, excerpt="Measured effect.",
                )
            ])
            critique = CriticSynthesizer(
                FakeModel({"final_critique": unavailable}),
                BudgetManager(BudgetConfig()),
                audit,
            ).critique(
                original_query="Query",
                tasks=[task()],
                claims=ledger.claims(),
                allow_followups=False,
            )

        self.assertEqual(
            "Final coverage review was unavailable; unresolved gaps may remain.",
            critique.remaining_gaps[0],
        )

    def test_critic_can_surface_contradiction_only_claim_without_merging_it(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ledger = EvidenceLedger(JsonlAuditLogger(Path(tmp) / "events.jsonl"))
            ledger.add_observations(
                [EvidenceObservation(
                    observation_id="O1", task_id="T1",
                    source_url="https://contradict.example/a",
                    source_domain="contradict.example",
                    statement="X improves Y.", polarity=Polarity.CONTRADICT,
                    excerpt="No measured effect.",
                ), EvidenceObservation(
                    observation_id="O2", task_id="T1",
                    source_url="https://neutral.example/a",
                    source_domain="neutral.example",
                    statement="X improves Y.", polarity=Polarity.NEUTRAL,
                    excerpt="The outcome was not measured.",
                )]
            )
            claim = ledger.claims()[0]
            context = build_synthesis_context(
                ledger.claims(),
                ledger.observations(),
                critic_contested_claim_ids=[claim.claim_id],
            )
            fallback = build_fallback_report_payload(context, [])
            report = render_report(fallback, context)
            source_id = next(iter(context.contradicting_sources_by_claim[claim.claim_id]))
            model_report = render_report(
                {
                    "executive_summary": "Ignored.",
                    "main_findings": [],
                    "contested_findings": [{
                        "claim_id": claim.claim_id,
                        "synthesis": "Sources disagree and X improves Y.",
                        "source_ids": [source_id],
                    }],
                    "weak_evidence": [],
                    "remaining_gaps": [],
                },
                context,
            )
            neutral_source_id = next(
                source
                for source in context.allowed_sources_by_claim[claim.claim_id]
                if source not in context.contradicting_sources_by_claim[claim.claim_id]
            )
            with self.assertRaisesRegex(CitationError, "do not support"):
                render_report(
                    {
                        "executive_summary": "Ignored.",
                        "main_findings": [],
                        "contested_findings": [{
                            "claim_id": claim.claim_id,
                            "synthesis": "The proposition is contradicted.",
                            "source_ids": [neutral_source_id],
                        }],
                        "weak_evidence": [],
                        "remaining_gaps": [],
                    },
                    context,
                )

        self.assertEqual(frozenset({claim.claim_id}), context.contested_claim_ids)
        self.assertFalse(claim.disagreement_flag)
        self.assertEqual(0, claim.supporting_domain_count)
        self.assertEqual(1, claim.contradicting_domain_count)
        self.assertEqual(claim.claim_id, fallback["contested_findings"][0]["claim_id"])
        self.assertIn("Available sources contradict this proposition", report)
        self.assertNotIn("Sources disagree about this finding", report)
        self.assertIn("## Contested Findings", report)
        self.assertIn("Available sources contradict this proposition", model_report)
        self.assertNotIn("Sources disagree and X improves Y", model_report)

    def test_critic_cannot_mark_support_only_claim_as_contested(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ledger = EvidenceLedger(JsonlAuditLogger(Path(tmp) / "events.jsonl"))
            ledger.add_observations(
                [EvidenceObservation(
                    observation_id="O1", task_id="T1",
                    source_url="https://support.example/a",
                    source_domain="support.example",
                    statement="X improves Y.", polarity=Polarity.SUPPORT,
                    excerpt="Measured effect.",
                )]
            )
            claim = ledger.claims()[0]
            context = build_synthesis_context(
                ledger.claims(),
                ledger.observations(),
                critic_contested_claim_ids=[claim.claim_id],
            )

        self.assertEqual(frozenset(), context.contested_claim_ids)

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

        self.assertIn("Available evidence is insufficient", report)
        self.assertNotIn("No page", report)
        self.assertNotIn("run", report.casefold())
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

    def test_remaining_gaps_strip_internal_claim_ids(self) -> None:
        context = build_synthesis_context([], [])
        gaps = contextualize_remaining_gaps(
            context,
            [
                "Evidence is flagged as a gap (Cb7343f4bbe) but remains unresolved.",
                "AI convergence (C3f7470b328) is not substantively developed.",
                "Sustainability (C5e3b56aba4, Cedf62d166f) is partially addressed.",
                "Claim `cb7343f4bbe` remains unresolved.",
            ],
        )

        self.assertEqual(
            [
                "Evidence is flagged as a gap but remains unresolved.",
                "AI convergence is not substantively developed.",
                "Sustainability is partially addressed.",
                "Claim remains unresolved.",
            ],
            gaps,
        )
        self.assertNotRegex(" ".join(gaps), r"\bC[0-9a-fA-F]{10}\b")

    def test_report_schema_constrains_ledger_identifiers(self) -> None:
        schema = _report_schema(["C1", "C2"], ["S1", "S2"])
        for section in ("main_findings", "contested_findings", "weak_evidence"):
            properties = schema["properties"][section]["items"]["properties"]
            self.assertEqual(["C1", "C2"], properties["claim_id"]["enum"])
            self.assertEqual(["S1", "S2"], properties["source_ids"]["items"]["enum"])
        self.assertEqual(8, schema["properties"]["main_findings"]["maxItems"])
        self.assertEqual(5, schema["properties"]["weak_evidence"]["maxItems"])
        self.assertIn(
            "at most 100 words",
            schema["properties"]["main_findings"]["items"]["properties"][
                "synthesis"
            ]["description"],
        )

    def test_report_ignores_unbound_model_executive_summary(self) -> None:
        context = build_synthesis_context([], [])
        payload = {
            "executive_summary": "word " * 1_801,
            "main_findings": [],
            "contested_findings": [],
            "weak_evidence": [],
            "remaining_gaps": [],
        }

        report = render_report(payload, context)

        self.assertNotIn("word word", report)
        self.assertIn("Available evidence is insufficient", report)

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
            critique = critic.critique(
                original_query="Query",
                tasks=[task()],
                claims=[],
                allow_followups=False,
            )

        self.assertEqual([], critique.followup_tasks)

    def test_critic_ignores_followup_with_invalid_parent(self) -> None:
        model = FakeModel({
            "initial_critique": {
                "coverage": [],
                "contested_claim_ids": [],
                "remaining_gaps": [],
                "followups": [{
                    "parent_task_id": "missing",
                    "question": "What evidence resolves the gap?",
                    "rationale": "The gap is material.",
                    "priority": "high",
                    "page_budget_share": 0.1,
                }],
            }
        })
        with tempfile.TemporaryDirectory() as tmp:
            critique = CriticSynthesizer(
                model,
                BudgetManager(BudgetConfig()),
                JsonlAuditLogger(Path(tmp) / "events.jsonl"),
            ).critique(
                original_query="Query",
                tasks=[task()],
                claims=[],
                allow_followups=True,
            )

        self.assertEqual([], critique.followup_tasks)

    def test_synthesis_fallback_removes_unknown_source_id(self) -> None:
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
            report = critic.synthesize(
                original_query="Query",
                tasks=[task()],
                claims=ledger.claims(),
                observations=ledger.observations(),
                remaining_gaps=[],
            )

        self.assertNotIn("S99", report)
        self.assertIn("- Sources: S1", report)

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
            report = critic.synthesize(
                original_query="Query",
                tasks=[task()],
                claims=ledger.claims(),
                observations=ledger.observations(),
                remaining_gaps=[],
            )

        self.assertIn("- Sources: S1", report)
        claim_section = report.split("### X improves Y.", 1)[1].split("###", 1)[0]
        self.assertNotIn("S2", claim_section)
        self.assertNotIn("Wrongly cites another claim", report)

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

    def test_synthesis_uses_safe_fallback_after_one_failed_retry(self) -> None:
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
                "main_findings": [],
                "contested_findings": [{
                    "claim_id": claim.claim_id,
                    "synthesis": "Misclassified finding.", "source_ids": ["S1"],
                }],
                "weak_evidence": [], "remaining_gaps": [],
            }
            model = FakeModel({"final_report": invalid})

            report = CriticSynthesizer(
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
        self.assertEqual(
            1, sum(event["event"] == "synthesis.fallback_rendered" for event in events)
        )
        self.assertIn("X improves Y.", report)
        self.assertIn("- Sources: S1", report)

    def test_synthesis_fallback_binds_uncited_claim_to_verified_source(self) -> None:
        model = FakeModel({})
        with tempfile.TemporaryDirectory() as tmp:
            event_path = Path(tmp) / "events.jsonl"
            audit = JsonlAuditLogger(event_path)
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
            report = critic.synthesize(
                original_query="Query",
                tasks=[task()],
                claims=ledger.claims(),
                observations=ledger.observations(),
                remaining_gaps=[],
            )
            events = [json.loads(line) for line in event_path.read_text().splitlines()]

        self.assertEqual(
            1, sum(event["event"] == "synthesis.validation_retry" for event in events),
        )
        self.assertIn("- Sources: S1", report)

    def test_synthesis_fallback_never_keeps_cross_claim_source(self) -> None:
        model = FakeModel({})
        with tempfile.TemporaryDirectory() as tmp:
            audit = JsonlAuditLogger(Path(tmp) / "events.jsonl")
            ledger = EvidenceLedger(audit)
            ledger.add_observations(
                [
                    EvidenceObservation(
                        observation_id="O1", task_id="T1",
                        source_url="https://a.example/source", source_domain="a.example",
                        statement="X improves Y.", polarity=Polarity.SUPPORT,
                        excerpt="Measured effect.",
                    ),
                    EvidenceObservation(
                        observation_id="O2", task_id="T1",
                        source_url="https://b.example/source", source_domain="b.example",
                        statement="Z improves Q.", polarity=Polarity.SUPPORT,
                        excerpt="Different measured effect.",
                    ),
                ]
            )
            claims = ledger.claims()
            first_claim = next(claim for claim in claims if claim.text == "X improves Y.")
            model.handlers["final_report"] = {
                "executive_summary": "Supported summary.",
                "main_findings": [
                    {
                        "claim_id": first_claim.claim_id,
                        "synthesis": "Z improves Q [S2].",
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
                original_query="Query", tasks=[task()], claims=claims,
                observations=ledger.observations(), remaining_gaps=[],
            )

        self.assertEqual(2, model.calls.count("final_report"))
        claim_section = report.split("### X improves Y.", 1)[1].split("###", 1)[0]
        self.assertIn("- Sources: S1", claim_section)
        self.assertNotIn("- Sources: S2", claim_section)
        self.assertNotIn("Z improves Q", claim_section)

    def test_synthesis_fallback_removes_unbound_summary_citation(self) -> None:
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
            report = critic.synthesize(
                original_query="Query",
                tasks=[task()],
                claims=ledger.claims(),
                observations=ledger.observations(),
                remaining_gaps=[],
            )

        self.assertNotIn("without a finding", report)
        self.assertIn("- Sources: S1", report)

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

    def test_claim_heading_cannot_emit_unknown_citation_shaped_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            audit = JsonlAuditLogger(Path(tmp) / "events.jsonl")
            ledger = EvidenceLedger(audit)
            ledger.add_observations(
                [EvidenceObservation(
                    observation_id="O1", task_id="T1",
                    source_url="https://example.com/a", source_domain="example.com",
                    statement="Evidence [S999] supports X.", polarity=Polarity.SUPPORT,
                    excerpt="Measured effect.",
                )]
            )
            claim = ledger.claims()[0]
            context = build_synthesis_context(ledger.claims(), ledger.observations())
            report = render_report(
                {
                    "executive_summary": "Supported summary.",
                    "main_findings": [{
                        "claim_id": claim.claim_id,
                        "synthesis": "Evidence supports X.",
                        "source_ids": ["S1"],
                    }],
                    "contested_findings": [],
                    "weak_evidence": [],
                    "remaining_gaps": [],
                },
                context,
            )

        self.assertNotIn("S999", report)
        self.assertIn("### Evidence supports X.", report)
        self.assertIn("- S1: https://example.com/a", report)

    def test_claim_heading_preserves_bare_product_model_tokens(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            audit = JsonlAuditLogger(Path(tmp) / "events.jsonl")
            ledger = EvidenceLedger(audit)
            ledger.add_observations(
                [EvidenceObservation(
                    observation_id="O1", task_id="T1",
                    source_url="https://example.com/a", source_domain="example.com",
                    statement="Samsung Galaxy S24 supports satellite messaging.",
                    polarity=Polarity.SUPPORT, excerpt="Satellite messaging is supported.",
                )]
            )
            context = build_synthesis_context(ledger.claims(), ledger.observations())
            fallback = build_fallback_report_payload(context, [])
            report = render_report(
                fallback,
                context,
                remaining_gaps=fallback["remaining_gaps"],
            )

        self.assertIn("### Samsung Galaxy S24 supports satellite messaging.", report)


if __name__ == "__main__":
    unittest.main()
