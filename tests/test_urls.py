import threading
import unittest

from deep_research.urls import UrlRegistry, normalize_url


class UrlTests(unittest.TestCase):
    def test_normalizes_tracking_query_fragment_and_case(self) -> None:
        self.assertEqual(
            "https://example.com/path?a=1&b=2",
            normalize_url(
                "HTTPS://WWW.Example.com/path/?b=2&utm_source=x&a=1#section"
            ),
        )

    def test_registry_claim_is_atomic(self) -> None:
        registry = UrlRegistry()
        outcomes: list[bool] = []

        def claim(url: str) -> None:
            outcomes.append(registry.claim_url(url)[0])

        variants = [
            "https://example.com/page",
            "https://example.com/page/",
            "https://www.example.com/page?utm_campaign=x",
        ] * 10
        threads = [
            threading.Thread(target=claim, args=(url,)) for url in variants
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.assertEqual(1, sum(outcomes))


if __name__ == "__main__":
    unittest.main()
