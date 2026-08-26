import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from deep_research.api.sessions import ResearchSessionService, SessionCapacityError
from deep_research.api.http import (
    GLOBAL_LIMITS,
    AnonymousWorkspaceQuota,
    create_app,
)


WORKSPACE_A = "a" * 32
WORKSPACE_B = "b" * 32


class LocalApiBoundaryTests(unittest.TestCase):
    def test_rejects_untrusted_host_and_origin(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            service = ResearchSessionService(
                output_root=Path(tmp), auto_start=False
            )
            client = TestClient(create_app(service))

            hostile_host = client.get(
                "/api/health", headers={"host": "attacker.example"}
            )
            hostile_origin = client.get(
                "/api/health", headers={"origin": "https://attacker.example"}
            )
            local = client.get(
                "/api/health", headers={"origin": "http://localhost:3000"}
            )

        self.assertEqual(400, hostile_host.status_code)
        self.assertEqual(403, hostile_origin.status_code)
        self.assertEqual(200, local.status_code)
        self.assertEqual("no-store", local.headers["cache-control"])

    def test_production_origin_and_host_are_explicitly_configurable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            "os.environ",
            {
                "WEB_ALLOWED_ORIGINS": "https://research.example",
                "WEB_ALLOWED_HOSTS": "api.example",
            },
        ):
            client = TestClient(
                create_app(ResearchSessionService(output_root=Path(tmp), auto_start=False)),
                base_url="https://api.example",
            )
            response = client.get(
                "/api/health",
                headers={"origin": "https://research.example"},
            )

        self.assertEqual(200, response.status_code)
        self.assertEqual(
            "https://research.example",
            response.headers["access-control-allow-origin"],
        )

    def test_wildcard_deployment_boundary_is_rejected(self) -> None:
        with patch.dict("os.environ", {"WEB_ALLOWED_ORIGINS": "*"}):
            with self.assertRaisesRegex(ValueError, "explicit values"):
                create_app(ResearchSessionService(auto_start=False))

    def test_anonymous_workspaces_are_isolated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            "os.environ",
            {
                "AWS_BEARER_TOKEN_BEDROCK": "key",
                "TAVILY_API_KEY": "key",
            },
            clear=False,
        ):
            service = ResearchSessionService(output_root=Path(tmp), auto_start=False)
            client = TestClient(create_app(service))
            first = client.post(
                "/api/sessions",
                headers={"X-Workspace-Key": WORKSPACE_A},
                json={"query": "First bounded research question"},
            ).json()
            service.get(first["id"]).mark_ready(
                [{"id": f"T{index}"} for index in range(1, 5)]
            )
            second = client.post(
                "/api/sessions",
                headers={"X-Workspace-Key": WORKSPACE_B},
                json={"query": "Second bounded research question"},
            ).json()

            listed = client.get(
                "/api/sessions", headers={"X-Workspace-Key": WORKSPACE_A}
            )
            hidden = client.get(
                f"/api/sessions/{second['id']}",
                headers={"X-Workspace-Key": WORKSPACE_A},
            )

        self.assertEqual([first["id"]], [item["id"] for item in listed.json()])
        self.assertEqual(404, hidden.status_code)

    def test_start_capacity_returns_429(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            service = ResearchSessionService(
                output_root=Path(tmp), auto_start=False, max_active_sessions=1
            )
            first = service.create("First bounded query", workspace_id=WORKSPACE_A)
            first.mark_ready([{"id": f"T{index}"} for index in range(1, 5)])
            second = service.create("Second bounded query", workspace_id=WORKSPACE_A)
            second.mark_ready([{"id": f"T{index}"} for index in range(1, 5)])
            client = TestClient(create_app(service))

            with patch("deep_research.api.sessions.threading.Thread"):
                service.start(first.id, workspace_id=WORKSPACE_A)
                responses = [
                    client.post(
                        f"/api/sessions/{second.id}/start",
                        headers={"X-Workspace-Key": WORKSPACE_A},
                    )
                    for _ in range(6)
                ]

        self.assertTrue(all(response.status_code == 429 for response in responses))
        self.assertTrue(all("active research capacity" in response.text for response in responses))

    def test_daily_anonymous_quota_fails_closed(self) -> None:
        quota = AnonymousWorkspaceQuota()
        for _ in range(10):
            quota.reserve(WORKSPACE_A, "plan")

        with self.assertRaisesRegex(SessionCapacityError, "daily anonymous plan quota"):
            quota.reserve(WORKSPACE_A, "plan")

    def test_global_anonymous_quota_cannot_be_bypassed_with_new_workspace_keys(self) -> None:
        quota = AnonymousWorkspaceQuota()
        for index in range(GLOBAL_LIMITS["plan"]):
            quota.reserve(f"{index:032x}", "plan")

        with self.assertRaisesRegex(SessionCapacityError, "service"):
            quota.reserve("f" * 32, "plan")

    def test_oversized_request_is_rejected_before_parsing(self) -> None:
        client = TestClient(create_app(ResearchSessionService(auto_start=False)))

        response = client.get("/api/health", headers={"Content-Length": "20000"})

        self.assertEqual(413, response.status_code)

    def test_headerless_streamed_body_is_size_limited(self) -> None:
        client = TestClient(create_app(ResearchSessionService(auto_start=False)))

        response = client.post(
            "/api/sessions",
            headers={"X-Workspace-Key": WORKSPACE_A},
            content=iter([b"x" * 9_000, b"y" * 9_000]),
        )

        self.assertEqual(413, response.status_code)


if __name__ == "__main__":
    unittest.main()
