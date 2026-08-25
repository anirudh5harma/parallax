import tempfile
import unittest
from pathlib import Path

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


if __name__ == "__main__":
    unittest.main()
