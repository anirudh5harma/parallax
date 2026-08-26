import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from deep_research.api.sessions import ResearchSessionService
from deep_research.cli import _worker_timeout
from deep_research.domain.budget import BudgetConfig
from deep_research.web_cli import main


class WebCliTests(unittest.TestCase):
    def test_defaults_to_fast_and_deep_web_profiles(self) -> None:
        with patch("deep_research.web_cli.uvicorn.run") as run:
            result = main([])

        configs = run.call_args.args[0].state.sessions.configs
        self.assertEqual(0, result)
        self.assertEqual((64, 600, 220, 900), (
            configs["fast"].max_searches,
            configs["fast"].max_sources,
            configs["fast"].max_pages,
            configs["fast"].wall_clock_timeout_seconds,
        ))
        self.assertEqual((100, 800, 400, 1200), (
            configs["deep"].max_searches,
            configs["deep"].max_sources,
            configs["deep"].max_pages,
            configs["deep"].wall_clock_timeout_seconds,
        ))

    def test_forwards_bounded_budget_controls(self) -> None:
        with patch("deep_research.web_cli.uvicorn.run") as run:
            result = main([
                "--port", "8123", "--fast-max-searches", "7",
                "--fast-max-sources", "11", "--fast-max-pages", "9",
                "--deep-max-searches", "13", "--deep-max-sources", "17",
                "--deep-max-pages", "15", "--max-concurrent-fetches", "3",
                "--timeout", "42", "--deep-timeout", "84",
            ])

        app = run.call_args.args[0]
        fast = app.state.sessions.configs["fast"]
        deep = app.state.sessions.configs["deep"]
        self.assertEqual(0, result)
        self.assertEqual((7, 11, 9, 3, 42), (
            fast.max_searches, fast.max_sources, fast.max_pages,
            fast.max_concurrent_fetches, fast.wall_clock_timeout_seconds,
        ))
        self.assertEqual((13, 17, 15, 3, 84), (
            deep.max_searches, deep.max_sources, deep.max_pages,
            deep.max_concurrent_fetches, deep.wall_clock_timeout_seconds,
        ))
        self.assertEqual("127.0.0.1", run.call_args.kwargs["host"])
        self.assertEqual(8123, run.call_args.kwargs["port"])

    def test_invalid_budget_never_starts_server(self) -> None:
        with patch("deep_research.web_cli.uvicorn.run") as run:
            with self.assertRaises(SystemExit):
                main(["--deep-max-pages", "601"])

        run.assert_not_called()

    def test_worker_receives_every_configured_ceiling(self) -> None:
        config = BudgetConfig(
            max_searches=7, max_sources=8, max_pages=9, max_concurrent_fetches=3,
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
            session = service.create(
                "Bounded worker forwarding",
                seed_observations=[{
                    "observation_id": "Oseed",
                    "task_id": "B0",
                    "source_url": "https://parent.example/evidence",
                    "source_domain": "parent.example",
                    "statement": "Seed statement.",
                    "polarity": "support",
                    "excerpt": "Seed excerpt.",
                    "source_type": "paper",
                }],
            )
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
            seed_path = Path(command[command.index("--seed-evidence") + 1])
            seed_payload = json.loads(seed_path.read_text())

        environment = popen.call_args.kwargs["env"]
        self.assertEqual("7", command[command.index("--max-searches") + 1])
        self.assertEqual("8", command[command.index("--max-sources") + 1])
        self.assertEqual("9", command[command.index("--max-pages") + 1])
        self.assertEqual(
            "0", command[command.index("--followup-source-reserve") + 1]
        )
        self.assertEqual(
            "0", command[command.index("--followup-page-reserve") + 1]
        )
        self.assertEqual(
            "3", command[command.index("--max-concurrent-fetches") + 1]
        )
        self.assertEqual("42", command[command.index("--timeout") + 1])
        self.assertEqual("Oseed", seed_payload["observations"][0]["observation_id"])
        self.assertEqual(
            str(_worker_timeout(42)), environment["DEEP_RESEARCH_INTERNAL_TIMEOUT"]
        )


if __name__ == "__main__":
    unittest.main()
