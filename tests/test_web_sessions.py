import tempfile
import unittest
from pathlib import Path

from deep_research.web_sessions import (
    ResearchSessionService,
    _public_audit_event,
    evidence_view,
)


def ledger() -> dict[str, object]:
    return {
        "claims": [
            {
                "claim_id": "C1",
                "text": "Intervention X improves outcome Y.",
                "supporting_observations": ["O1"],
                "contradicting_observations": ["O2"],
                "neutral_observations": [],
                "supporting_domain_count": 1,
                "contradicting_domain_count": 1,
                "confidence_tag": "Low",
                "disagreement_flag": True,
            }
        ],
        "observations": [
            {
                "observation_id": "O1",
                "source_url": "https://support.example/a",
                "source_domain": "support.example",
                "statement": "Intervention X improves outcome Y.",
                "polarity": "support",
                "excerpt": "A measurable improvement was observed.",
            },
            {
                "observation_id": "O2",
                "source_url": "https://contradict.example/b",
                "source_domain": "contradict.example",
                "statement": "Intervention X improves outcome Y.",
                "polarity": "contradict",
                "excerpt": "No measurable improvement was observed.",
            },
        ],
    }


class EvidenceViewTests(unittest.TestCase):
    def test_assigns_deterministic_source_ids_and_preserves_polarity(self) -> None:
        claims = evidence_view(ledger())

        self.assertEqual(1, len(claims))
        self.assertTrue(claims[0]["disagreement"])
        observations = claims[0]["observations"]
        self.assertEqual(["S2", "S1"], [item["source_id"] for item in observations])
        self.assertEqual(["support", "contradict"], [item["polarity"] for item in observations])


class ResearchSessionServiceTests(unittest.TestCase):
    def test_contradiction_starts_user_directed_child_session(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            service = ResearchSessionService(
                output_root=Path(tmp),
                auto_start=False,
            )
            parent = service.create("Does intervention X improve outcome Y?")
            parent.status = "completed"
            parent.ledger = ledger()

            child = service.create_branch(parent.id, "O2")

        self.assertEqual(parent.id, child.parent_session_id)
        self.assertEqual("O2", child.branch["observation_id"])
        self.assertEqual("S1", child.branch["source_id"])
        self.assertIn("independent corroboration", child.query)
        self.assertIn("Do not assume", child.query)

    def test_supporting_observation_cannot_start_contradiction_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            service = ResearchSessionService(
                output_root=Path(tmp),
                auto_start=False,
            )
            parent = service.create("Does intervention X improve outcome Y?")
            parent.status = "completed"
            parent.ledger = ledger()

            with self.assertRaisesRegex(ValueError, "only contradicting"):
                service.create_branch(parent.id, "O1")

    def test_progress_events_do_not_expose_observation_content(self) -> None:
        event = _public_audit_event(
            {
                "event": "observation.extracted",
                "data": {
                    "observation": {
                        "excerpt": "Sensitive excerpt",
                        "statement": "Sensitive statement",
                    }
                },
            }
        )

        self.assertEqual("Evidence observation added to the ledger", event["message"])
        self.assertNotIn("Sensitive", str(event))


if __name__ == "__main__":
    unittest.main()
