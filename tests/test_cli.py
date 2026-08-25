import unittest

from deep_research.cli import build_parser, config_from_args


class CliTests(unittest.TestCase):
    def test_serious_profile_has_documented_ceiling(self) -> None:
        args = build_parser().parse_args(["Query", "--profile", "serious"])
        config = config_from_args(args)
        self.assertEqual(80, config.max_searches)
        self.assertEqual(200, config.max_pages)
        self.assertEqual(10, config.max_concurrent_fetches)

    def test_override_cannot_exceed_absolute_guard(self) -> None:
        args = build_parser().parse_args(["Query", "--max-pages", "401"])
        with self.assertRaises(ValueError):
            config_from_args(args)


if __name__ == "__main__":
    unittest.main()
