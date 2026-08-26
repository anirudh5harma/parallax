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
    SessionLaunchError,
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
    def test_failure_exposes_safe_structured_error(self) -> None:
        service = ResearchSessionService(auto_start=False)
        session = service.create("A bounded research question")

        service._fail(
            session,
            "The Tavily usage limit is exhausted.",
            code="tavily_quota_exhausted",
            provider="tavily",
        )

        summary = session.summary()
        self.assertEqual("failed", summary["status"])
        self.assertEqual("tavily_quota_exhausted", summary["error_code"])
        self.assertEqual("tavily", summary["error_provider"])
        self.assertFalse(summary["error_retryable"])
        self.assertEqual("The Tavily usage limit is exhausted.", summary["error"])
        self.assertEqual(
            "tavily_quota_exhausted", session.events[-1]["data"]["error_code"]
        )

    def test_planner_thread_launch_failure_rolls_back_session(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch(
            "deep_research.web_sessions.threading.Thread.start",
            side_effect=RuntimeError("thread unavailable"),
        ):
            service = ResearchSessionService(output_root=Path(tmp))

            with self.assertRaisesRegex(SessionLaunchError, "planner could not start"):
                service.create("A bounded research question")

        self.assertEqual([], service.list_sessions())

    def test_worker_thread_launch_failure_restores_ready_session(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            service = ResearchSessionService(output_root=Path(tmp), auto_start=False)
            session = service.create("A bounded research question")
            session.mark_ready([{"id": f"T{index}"} for index in range(1, 5)])

            with patch(
                "deep_research.web_sessions.threading.Thread.start",
                side_effect=RuntimeError("thread unavailable"),
            ), self.assertRaisesRegex(SessionLaunchError, "worker could not start"):
                service.start(session.id)

        self.assertEqual("ready", session.status)

    def test_active_session_capacity_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            service = ResearchSessionService(
                output_root=Path(tmp), auto_start=False, max_active_sessions=1
            )
            service.create("First bounded query")

            with self.assertRaisesRegex(SessionCapacityError, "capacity"):
                service.create("Second bounded query")

    def test_workspace_scope_hides_other_sessions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            service = ResearchSessionService(output_root=Path(tmp), auto_start=False)
            first = service.create("First bounded query", workspace_id="a" * 32)
            first.mark_ready([{"id": f"T{index}"} for index in range(1, 5)])
            second = service.create("Second bounded query", workspace_id="b" * 32)

            self.assertEqual([first.id], [item["id"] for item in service.list_sessions("a" * 32)])
            with self.assertRaises(KeyError):
                service.get(second.id, workspace_id="a" * 32)

    def test_control_characters_are_rejected_before_worker_start(self) -> None:
        service = ResearchSessionService(auto_start=False)

        with self.assertRaisesRegex(ValueError, "control characters"):
            service.create("A valid-looking query\x00with a null byte")

    def test_unpaired_surrogate_is_rejected_before_worker_start(self) -> None:
        service = ResearchSessionService(auto_start=False)

        with self.assertRaisesRegex(ValueError, "valid UTF-8"):
            service.create("A valid-looking query\ud800with a surrogate")

    def test_retention_promotes_children_of_evicted_session(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            service = ResearchSessionService(
                output_root=Path(tmp), auto_start=False, max_retained_sessions=2
            )
            parent = service.create("Parent research")
            parent.status = "completed"
            child = service.create("Child research", parent_session_id=parent.id)
            child.status = "completed"

            service.create("Newest research")

        self.assertIsNone(child.parent_session_id)
        self.assertNotIn(parent.id, [item["id"] for item in service.list_sessions()])
        self.assertIn(child.id, [item["id"] for item in service.list_sessions()])

    def test_start_rechecks_active_capacity_atomically(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            service = ResearchSessionService(
                output_root=Path(tmp), auto_start=False, max_active_sessions=1
            )
            first = service.create("First bounded query")
            first.mark_ready([{"id": f"T{index}"} for index in range(1, 5)])
            second = service.create("Second bounded query")
            second.mark_ready([{"id": f"T{index}"} for index in range(1, 5)])

            with patch("deep_research.web_sessions.threading.Thread") as worker:
                service.start(first.id)
                with self.assertRaisesRegex(SessionCapacityError, "capacity"):
                    service.start(second.id)

            worker.assert_called_once()
            self.assertEqual("queued", first.status)
            self.assertEqual("ready", second.status)

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

    def test_rejected_query_is_terminal_with_rewrite_guidance(self) -> None:
        session = ResearchSession(
            id="session", query="hello", title="hello", created_at="now"
        )

        session.reject("Ask a specific evidence-answerable question.")
        status, events = session.stream_snapshot(0)

        self.assertEqual("rejected", status)
        self.assertEqual("plan.rejected", events[-1]["event"])
        self.assertIn("specific", session.error)

    def test_report_streaming_status_publishes_before_chunks(self) -> None:
        session = ResearchSession(
            id="session", query="query", title="title", created_at="now",
            status="running",
        )

        session.begin_report()
        status, events = session.stream_snapshot(0)

        self.assertEqual("synthesizing", status)
        self.assertEqual("report.started", events[-1]["event"])

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

    def test_branch_prompt_stays_inside_query_ceiling(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            service = ResearchSessionService(output_root=Path(tmp), auto_start=False)
            parent = service.create("Q" * 2_000)
            parent.status = "completed"
            parent.ledger = ledger()
            parent.ledger["claims"][0]["text"] = "C" * 500
            parent.ledger["observations"][1]["source_url"] = (
                "https://contradict.example/" + "u" * 1_000
            )
            parent.ledger["observations"][1]["excerpt"] = "E" * 500

            child = service.create_branch(parent.id, "O2")

        self.assertLessEqual(len(child.query), 4_000)
        self.assertEqual(parent.id, child.parent_session_id)

    def test_branch_prompt_strips_control_heavy_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            service = ResearchSessionService(output_root=Path(tmp), auto_start=False)
            parent = service.create("Does intervention X improve outcome Y?")
            parent.status = "completed"
            parent.ledger = ledger()
            parent.ledger["claims"][0]["text"] = "C\x00" * 500
            parent.ledger["observations"][1]["excerpt"] = "E\x01" * 500

            child = service.create_branch(parent.id, "O2")

        self.assertLessEqual(len(child.query), 4_000)
        self.assertNotIn("\x00", child.query)
        self.assertNotIn("\x01", child.query)

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

    def test_page_progress_exposes_only_safe_source_domain(self) -> None:
        event = _public_audit_event(
            {
                "event": "page.explored",
                "data": {
                    "exploration": {
                        "domain": "research.example",
                        "url": "https://research.example/private-path",
                        "content_hash": "secret-hash",
                    }
                },
            }
        )

        self.assertEqual("research.example", event["source_domain"])
        self.assertNotIn("private-path", str(event))
        self.assertNotIn("secret-hash", str(event))

    def test_discovery_progress_exposes_domain_without_url(self) -> None:
        event = _public_audit_event(
            {
                "event": "source.discovered",
                "data": {
                    "source_domain": "agency.gov",
                    "url": "https://agency.gov/private-path",
                },
            }
        )

        self.assertEqual("agency.gov", event["source_domain"])
        self.assertNotIn("private-path", str(event))

    def test_fetch_failures_expose_only_generic_progress(self) -> None:
        event = _public_audit_event(
            {
                "event": "page.fetch_failed",
                "data": {"error": "unsupported content type: application/pdf"},
            }
        )

        self.assertEqual("page.fetch_failed", event["stage"])
        self.assertNotIn("application/pdf", str(event))


if __name__ == "__main__":
    unittest.main()
