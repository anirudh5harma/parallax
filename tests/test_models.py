import unittest

from deep_research.domain.models import (
    EvidenceObservation,
    Polarity,
    Priority,
    ResearchTask,
    TaskStatus,
)


class ResearchTaskTests(unittest.TestCase):
    def test_rejects_depth_above_one(self) -> None:
        with self.assertRaises(ValueError):
            ResearchTask(
                id="task-1",
                question="What changed?",
                rationale="Needed for scope",
                priority=Priority.HIGH,
                page_budget_share=0.5,
                parent_task_id="task-0",
                depth=2,
                status=TaskStatus.PENDING,
            )

    def test_followup_requires_parent(self) -> None:
        with self.assertRaises(ValueError):
            ResearchTask(
                id="task-1",
                question="What changed?",
                rationale="Needed for scope",
                priority=Priority.HIGH,
                page_budget_share=0.5,
                depth=1,
            )


class ObservationTests(unittest.TestCase):
    def test_requires_http_source(self) -> None:
        with self.assertRaises(ValueError):
            EvidenceObservation(
                observation_id="obs-1",
                task_id="task-1",
                source_url="file:///tmp/source",
                source_domain="example.com",
                statement="A falsifiable claim",
                polarity=Polarity.SUPPORT,
                excerpt="Evidence excerpt",
            )


if __name__ == "__main__":
    unittest.main()
