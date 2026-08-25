import unittest
from unittest.mock import patch

from deep_research.web_cli import main


class WebCliTests(unittest.TestCase):
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
                main(["--max-pages", "401"])

        run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
