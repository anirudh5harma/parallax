import os
import subprocess
import unittest
from unittest.mock import patch

from deep_research.cli import _worker_timeout, build_parser, config_from_args, main


class CliTests(unittest.TestCase):
    @staticmethod
    def _credentials() -> dict[str, str]:
        return {
            "AWS_BEARER_TOKEN_BEDROCK": "bedrock-key",
            "TAVILY_API_KEY": "tavily-key",
        }

    def test_serious_profile_has_documented_ceiling(self) -> None:
        args = build_parser().parse_args(["Query", "--profile", "serious"])
        config = config_from_args(args)
        self.assertEqual(100, config.max_searches)
        self.assertEqual(600, config.max_pages)
        self.assertEqual(12, config.max_concurrent_fetches)

    def test_fast_and_deep_profiles_are_selectable(self) -> None:
        fast = config_from_args(build_parser().parse_args(["Query", "--profile", "fast"]))
        deep = config_from_args(build_parser().parse_args(["Query", "--profile", "deep"]))

        self.assertEqual(220, fast.max_pages)
        self.assertEqual(600, fast.max_sources)
        self.assertEqual(400, deep.max_pages)
        self.assertEqual(800, deep.max_sources)

    def test_override_cannot_exceed_absolute_guard(self) -> None:
        args = build_parser().parse_args(["Query", "--max-pages", "601"])
        with self.assertRaises(ValueError):
            config_from_args(args)

    def test_main_reports_missing_bedrock_and_search_keys(self) -> None:
        with patch.dict(os.environ, {}, clear=True), patch(
            "sys.stderr"
        ) as stderr:
            status = main(["Query"])

        self.assertEqual(2, status)
        message = "".join(call.args[0] for call in stderr.write.call_args_list)
        self.assertIn("AWS_BEARER_TOKEN_BEDROCK", message)
        self.assertIn("TAVILY_API_KEY", message)

    def test_main_supervises_worker_with_absolute_timeout(self) -> None:
        completed = subprocess.CompletedProcess([], 0)
        with patch.dict(os.environ, self._credentials(), clear=True), patch(
            "deep_research.cli.subprocess.run", return_value=completed
        ) as run:
            status = main(["Query", "--timeout", "1"])

        self.assertEqual(0, status)
        command = run.call_args.args[0]
        self.assertEqual("-I", command[1])
        self.assertTrue(command[2].endswith("/deep_research/_worker.py"))
        self.assertNotIn("-m", command)
        self.assertEqual(1, run.call_args.kwargs["timeout"])
        self.assertLess(
            float(run.call_args.kwargs["env"]["DEEP_RESEARCH_INTERNAL_TIMEOUT"]),
            1,
        )

    def test_worker_timeout_reserves_cleanup_inside_ceiling(self) -> None:
        self.assertEqual(295, _worker_timeout(300))
        self.assertGreater(_worker_timeout(0.01), 0)
        self.assertLess(_worker_timeout(0.01), 0.01)

    def test_main_terminates_worker_at_wall_clock_ceiling(self) -> None:
        with patch.dict(os.environ, self._credentials(), clear=True), patch(
            "deep_research.cli.subprocess.run",
            side_effect=subprocess.TimeoutExpired(["worker"], 0.01),
        ), patch("sys.stderr") as stderr:
            status = main(["Query", "--timeout", "0.01"])

        self.assertEqual(2, status)
        message = "".join(call.args[0] for call in stderr.write.call_args_list)
        self.assertIn("worker terminated", message)


if __name__ == "__main__":
    unittest.main()
