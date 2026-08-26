import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from deep_research.domain.budget import BudgetConfig
from deep_research.cli import _worker_timeout
from deep_research.web_cli import main
from deep_research.api.sessions import ResearchSessionService


class WebCliTests(unittest.TestCase):
    def test_defaults_to_serious_web_profile(self) -> None:
        with patch("deep_research.web_cli.uvicorn.run") as run:
            result = main([])

        config = run.call_args.args[0].state.sessions.config
        self.assertEqual(0, result)
        self.assertEqual(100, config.max_searches)
        self.assertEqual(600, config.max_pages)
        self.assertEqual(12, config.max_concurrent_fetches)
        self.assertEqual(1800, config.wall_clock_timeout_seconds)

    def test_forwards_bounded_budget_controls(self) -> None:
        with patch("deep_research.web_cli.uvicorn.run") as run:
            result = main([
                "--port", "8123", "--max-searches", "7", "--max-pages", "9",
                "--max-concurrent-fetches", "3", "--timeout", "42",
            ])

        app = run.call_args.args[0]
        config = app.state.sessions.config
        self.assertEqual(0, result)
        self.assertEqual(7, config.max_searches)
        self.assertEqual(9, config.max_pages)
        self.assertEqual(3, config.max_concurrent_fetches)
        self.assertEqual(42, config.wall_clock_timeout_seconds)
        self.assertEqual("127.0.0.1", run.call_args.kwargs["host"])
        self.assertEqual(8123, run.call_args.kwargs["port"])

    def test_invalid_budget_never_starts_server(self) -> None:
        with patch("deep_research.web_cli.uvicorn.run") as run:
            with self.assertRaises(SystemExit):
                main(["--max-pages", "601"])

        run.assert_not_called()

    def test_worker_receives_every_configured_ceiling(self) -> None:
        config = BudgetConfig(
            max_searches=7, max_pages=9, max_concurrent_fetches=3,
            wall_clock_timeout_seconds=42,
        )

        class ExitedProcess:
            stderr = io.StringIO("")

            @staticmethod
            def poll() -> int:
                return 1

        with tempfile.TemporaryDirectory() as tmp:
            service = ResearchSessionService(
                output_root=Path(tmp), config=config, auto_start=False
            )
            session = service.create("Bounded worker forwarding")
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
                    "deep_research.api.sessions.subprocess.Popen",
                    return_value=ExitedProcess(),
                ) as popen,
            ):
                service._run(session)

        command = popen.call_args.args[0]
        environment = popen.call_args.kwargs["env"]
        self.assertEqual("7", command[command.index("--max-searches") + 1])
        self.assertEqual("9", command[command.index("--max-pages") + 1])
        self.assertEqual(
            "3", command[command.index("--max-concurrent-fetches") + 1]
        )
        self.assertEqual("42", command[command.index("--timeout") + 1])
        self.assertEqual(
            str(_worker_timeout(42)), environment["DEEP_RESEARCH_INTERNAL_TIMEOUT"]
        )


if __name__ == "__main__":
    unittest.main()
