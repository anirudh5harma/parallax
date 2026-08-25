import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from deep_research.web_sessions import ResearchSessionService
from deep_research.webapp import create_app


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


if __name__ == "__main__":
    unittest.main()
