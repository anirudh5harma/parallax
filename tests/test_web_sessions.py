import tempfile
import threading
import unittest
import json
from unittest.mock import patch
from pathlib import Path

from deep_research.web_sessions import (
    MAX_SESSION_EVENTS,
    ResearchSession,
    ResearchSessionService,
    SessionCapacityError,
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
    def test_active_session_capacity_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            service = ResearchSessionService(
                output_root=Path(tmp), auto_start=False, max_active_sessions=1
            )
            service.create("First bounded query")

            with self.assertRaisesRegex(SessionCapacityError, "capacity"):
                service.create("Second bounded query")

    def test_event_retention_is_bounded_with_monotonic_ids(self) -> None:
        session = ResearchSession(
            id="session", query="query", title="title", created_at="now"
        )
        for index in range(MAX_SESSION_EVENTS + 3):
            session.publish("event", index=index)

        events = session.event_slice(0)
        self.assertEqual(MAX_SESSION_EVENTS, len(events))
        self.assertEqual(3, events[0]["id"])
        self.assertEqual(MAX_SESSION_EVENTS + 2, events[-1]["id"])

    def test_terminal_status_and_final_events_publish_together(self) -> None:
        session = ResearchSession(
            id="session", query="query", title="title", created_at="now",
            status="running",
        )
        session.finish(
            "completed",
            [
                ("report.chunk", {"text": "report"}),
                ("session.completed", {"status": "completed"}),
            ],
        )

        self.assertEqual("completed", session.status)
        self.assertEqual(
            ["report.chunk", "session.completed"],
            [event["event"] for event in session.event_slice(0)],
        )

    def test_ready_status_and_plan_event_publish_together(self) -> None:
        session = ResearchSession(
            id="session", query="query", title="title", created_at="now"
        )
        session.mark_ready([{"id": f"T{index}"} for index in range(1, 5)])

        status, events = session.stream_snapshot(0)

        self.assertEqual("ready", status)
        self.assertEqual("plan.ready", events[-1]["event"])
        self.assertEqual(4, len(session.summary()["plan"]))

    def test_concurrent_stream_never_observes_terminal_without_final_event(self) -> None:
        session = ResearchSession(
            id="session", query="query", title="title", created_at="now",
            status="running",
        )
        reader_started = threading.Event()
        writer_done = threading.Event()
        snapshots: list[tuple[str, list[dict[str, object]]]] = []

        def read_stream() -> None:
            reader_started.set()
            while not writer_done.is_set():
                snapshots.append(session.stream_snapshot(0))
            snapshots.append(session.stream_snapshot(0))

        reader = threading.Thread(target=read_stream)
        reader.start()
        reader_started.wait(timeout=1)
        session.finish(
            "completed",
            [
                *[("report.chunk", {"text": str(index)}) for index in range(1_000)],
                ("session.completed", {"status": "completed"}),
            ],
        )
        writer_done.set()
        reader.join(timeout=1)

        self.assertFalse(reader.is_alive())
        for status, events in snapshots:
            if status in {"completed", "completed_with_errors", "failed"}:
                self.assertIn("session.completed", [event["event"] for event in events])

    def test_sse_capacity_is_bounded_and_released(self) -> None:
        service = ResearchSessionService(
            auto_start=False, max_sse_connections=1
        )
        self.assertTrue(service.acquire_sse())
        self.assertFalse(service.acquire_sse())
        service.release_sse()
        self.assertTrue(service.acquire_sse())

    def test_worker_start_failure_becomes_terminal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            service = ResearchSessionService(output_root=Path(tmp), auto_start=False)
            session = service.create("A bounded query")
            session_root = Path(tmp) / session.id
            session_root.mkdir()
            (session_root / "plan.json").write_text(
                json.dumps({"query": session.query, "tasks": [{}] * 4}),
                encoding="utf-8",
            )
            with (
                patch.dict(
                    "os.environ",
                    {"AWS_BEARER_TOKEN_BEDROCK": "key", "TAVILY_API_KEY": "key"},
                    clear=False,
                ),
                patch(
                    "deep_research.web_sessions.subprocess.Popen",
                    side_effect=OSError("startup failed"),
                ),
            ):
                service._run(session)

        self.assertEqual("failed", session.status)
        self.assertEqual("session.failed", session.events[-1]["event"])
        self.assertIn("startup failed", session.error)

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
