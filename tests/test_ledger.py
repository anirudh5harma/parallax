import tempfile
import unittest
from pathlib import Path

from deep_research.audit import JsonlAuditLogger
from deep_research.ledger import EvidenceLedger
from deep_research.models import EvidenceObservation, Polarity


def observation(
    observation_id: str,
    domain: str,
    polarity: Polarity = Polarity.SUPPORT,
) -> EvidenceObservation:
    return EvidenceObservation(
        observation_id=observation_id,
        task_id="task-1",
        source_url=f"https://{domain}/article",
        source_domain=domain,
        statement="Intervention X improves outcome Y.",
        polarity=polarity,
        excerpt="Study reports a measurable effect.",
        source_type="paper",
    )


class EvidenceLedgerTests(unittest.TestCase):
    def test_confidence_uses_distinct_domains(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ledger = EvidenceLedger(JsonlAuditLogger(Path(tmp) / "events.jsonl"))
            ledger.add_observations(
                [
                    observation("o1", "one.example"),
                    observation("o2", "two.example"),
                    observation("o3", "three.example"),
                ]
            )
            claim = ledger.claims()[0]

        self.assertEqual("High", claim.confidence_tag.value)
        self.assertEqual(3, claim.supporting_domain_count)

    def test_meaningful_disagreement_stays_visible(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ledger = EvidenceLedger(JsonlAuditLogger(Path(tmp) / "events.jsonl"))
            ledger.add_observations(
                [
                    observation("o1", "one.example"),
                    observation("o2", "two.example"),
                    observation("o3", "three.example", Polarity.CONTRADICT),
                    observation("o4", "four.example", Polarity.CONTRADICT),
                ]
            )
            claim = ledger.claims()[0]

        self.assertTrue(claim.disagreement_flag)
        self.assertEqual("Low", claim.confidence_tag.value)
        self.assertEqual(2, claim.contradicting_domain_count)

    def test_no_support_is_insufficient(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ledger = EvidenceLedger(JsonlAuditLogger(Path(tmp) / "events.jsonl"))
            ledger.add_observations(
                [observation("o1", "one.example", Polarity.NEUTRAL)]
            )
            claim = ledger.claims()[0]

        self.assertEqual("Insufficient", claim.confidence_tag.value)


if __name__ == "__main__":
    unittest.main()
