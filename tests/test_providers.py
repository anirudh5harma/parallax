import json
import socket
import unittest
from email.message import Message
from unittest.mock import Mock, patch

from deep_research.pdf_extraction import PdfExtraction
from deep_research.providers import (
    BedrockConverseModel,
    HttpPageFetcher,
    ProviderError,
    TavilySearchClient,
    _NoRedirectHandler,
)


class BedrockConverseModelTests(unittest.TestCase):
    @staticmethod
    def _simple_response() -> dict[str, object]:
        return {
            "output": {"message": {"content": [{"text": '{"ok":true}'}]}}
        }

    @staticmethod
    def _simple_schema() -> dict[str, object]:
        return {
            "type": "object",
            "properties": {"ok": {"type": "boolean"}},
            "required": ["ok"],
            "additionalProperties": False,
        }

    def test_retries_retryable_failure_then_succeeds(self) -> None:
        error = ProviderError("busy", retryable=True, status=429, retry_after=0)
        with (
            patch(
                "deep_research.providers._post_json",
                side_effect=[error, self._simple_response()],
            ) as post,
            patch("deep_research.providers.time.sleep") as sleep,
        ):
            result = BedrockConverseModel("secret").generate_json(
                system_prompt="system", user_prompt="user", schema_name="result",
                schema=self._simple_schema(), timeout_seconds=10,
            )

        self.assertEqual({"ok": True}, result)
        self.assertEqual(2, post.call_count)
        sleep.assert_called_once_with(0)

    def test_model_request_timeout_is_capped(self) -> None:
        with patch(
            "deep_research.providers._post_json", return_value=self._simple_response()
        ) as post:
            BedrockConverseModel("secret", max_request_seconds=60).generate_json(
                system_prompt="system",
                user_prompt="user",
                schema_name="result",
                schema=self._simple_schema(),
                timeout_seconds=1800,
            )

        self.assertLessEqual(post.call_args.args[3], 60)

    def test_retries_schema_violation_with_correction_guidance(self) -> None:
        too_long = {
            "output": {
                "message": {"content": [{"text": '{"name":"far too long"}'}]}
            }
        }
        valid = {
            "output": {"message": {"content": [{"text": '{"name":"okay"}'}]}}
        }
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string", "maxLength": 4}},
            "required": ["name"],
            "additionalProperties": False,
        }
        with patch(
            "deep_research.providers._post_json",
            side_effect=[too_long, valid],
        ) as post:
            result = BedrockConverseModel("secret").generate_json(
                system_prompt="system",
                user_prompt="user",
                schema_name="result",
                schema=schema,
                timeout_seconds=10,
            )

        self.assertEqual({"name": "okay"}, result)
        corrected_payload = post.call_args_list[1].args[1]
        description = corrected_payload["outputConfig"]["textFormat"]["structure"][
            "jsonSchema"
        ]["description"]
        self.assertIn("too long", description)

    def test_does_not_retry_non_retryable_failure(self) -> None:
        error = ProviderError("bad request", retryable=False, status=400)
        with patch("deep_research.providers._post_json", side_effect=error) as post:
            with self.assertRaises(ProviderError):
                BedrockConverseModel("secret").generate_json(
                    system_prompt="system", user_prompt="user", schema_name="result",
                    schema=self._simple_schema(), timeout_seconds=10,
                )

        self.assertEqual(1, post.call_count)

    def test_retry_after_is_bounded_and_attempts_are_finite(self) -> None:
        error = ProviderError("busy", retryable=True, status=503, retry_after=99)
        with (
            patch("deep_research.providers._post_json", side_effect=error) as post,
            patch("deep_research.providers.time.sleep") as sleep,
        ):
            with self.assertRaises(ProviderError):
                BedrockConverseModel("secret").generate_json(
                    system_prompt="system", user_prompt="user", schema_name="result",
                    schema=self._simple_schema(), timeout_seconds=0.1,
                )

        self.assertEqual(2, post.call_count)
        self.assertEqual(1, sleep.call_count)
        self.assertLessEqual(sleep.call_args.args[0], 0.1)

    def test_evidence_source_type_schema_uses_supported_enum_type(self) -> None:
        from deep_research.researcher import EVIDENCE_SCHEMA

        source_type = EVIDENCE_SCHEMA["properties"]["observations"]["items"][
            "properties"
        ]["source_type"]
        self.assertEqual("string", source_type["type"])
        self.assertNotIn(None, source_type["enum"])

    def test_uses_bearer_auth_converse_and_supported_schema(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string", "minLength": 3, "maxLength": 20},
                "score": {"type": "number", "minimum": 0, "maximum": 1},
                "items": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 3,
                    "items": {"type": "string"},
                },
            },
            "required": ["name", "score", "items"],
            "additionalProperties": False,
        }
        response = {
            "output": {
                "message": {
                    "content": [
                        {"text": '{"name":"okay","score":1,"items":["one"]}'}
                    ]
                }
            }
        }
        with patch("deep_research.providers._post_json", return_value=response) as post:
            model = BedrockConverseModel(
                "secret",
                model_id="us.anthropic.claude-sonnet-4-6",
                region="ap-south-1",
            )
            result = model.generate_json(
                system_prompt="system",
                user_prompt="user",
                schema_name="result",
                schema=schema,
                timeout_seconds=10,
            )

        url, payload, headers, timeout = post.call_args.args
        sent_schema = json.loads(
            payload["outputConfig"]["textFormat"]["structure"]["jsonSchema"]["schema"]
        )
        self.assertEqual({"name": "okay", "score": 1, "items": ["one"]}, result)
        self.assertEqual(
            "https://bedrock-runtime.ap-south-1.amazonaws.com/model/"
            "us.anthropic.claude-sonnet-4-6/converse",
            url,
        )
        self.assertEqual("Bearer secret", headers["Authorization"])
        self.assertGreater(timeout, 9.9)
        self.assertNotIn("minLength", sent_schema["properties"]["name"])
        self.assertNotIn("maximum", sent_schema["properties"]["score"])
        self.assertIn("length must be at most 20", sent_schema["properties"]["name"]["description"])
        self.assertIn("must be at most 1", sent_schema["properties"]["score"]["description"])
        self.assertEqual(1, sent_schema["properties"]["items"]["minItems"])
        self.assertNotIn("maxItems", sent_schema["properties"]["items"])

    def test_rejects_response_without_text(self) -> None:
        with self.assertRaises(ProviderError):
            BedrockConverseModel._extract_json({"output": {"message": {"content": []}}})

    def test_rejects_response_violating_original_schema(self) -> None:
        response = {
            "output": {
                "message": {"content": [{"text": '{"items":["one","two"]}'}]}
            }
        }
        schema = {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "maxItems": 1,
                    "items": {"type": "string"},
                }
            },
            "required": ["items"],
            "additionalProperties": False,
        }
        with patch("deep_research.providers._post_json", return_value=response):
            with self.assertRaisesRegex(ProviderError, "too many items"):
                BedrockConverseModel("secret").generate_json(
                    system_prompt="system",
                    user_prompt="user",
                    schema_name="result",
                    schema=schema,
                    timeout_seconds=10,
                )

    def test_rejects_unsafe_region(self) -> None:
        with self.assertRaisesRegex(ValueError, "invalid AWS region"):
            BedrockConverseModel("secret", region="us-east-1@attacker.example")

    def test_rejects_control_characters_in_api_keys(self) -> None:
        for factory in (BedrockConverseModel, TavilySearchClient):
            with self.subTest(factory=factory.__name__):
                with self.assertRaisesRegex(ValueError, "printable ASCII"):
                    factory("secret\r\nInjected: value")

    def test_credentialed_provider_requests_do_not_redirect(self) -> None:
        handler = _NoRedirectHandler()
        self.assertIsNone(
            handler.redirect_request(None, None, 302, "Found", {}, "https://attacker.test")
        )

    def test_tavily_rejects_untrusted_endpoint(self) -> None:
        with self.assertRaisesRegex(ValueError, "trusted API origin"):
            TavilySearchClient("secret", endpoint="https://attacker.test/search")


class HttpPageFetcherSecurityTests(unittest.TestCase):
    def test_extracts_valid_pdf_without_relaxing_url_safety(self) -> None:
        headers = Message()
        headers["Content-Type"] = "application/pdf"
        fetcher = HttpPageFetcher()
        extraction = PdfExtraction(
            text="A literal finding from the paper.",
            title="Paper title",
            page_count=12,
        )
        with (
            patch.object(fetcher, "_fetch_once", return_value=(200, headers, b"%PDF-1.7")),
            patch("deep_research.providers.extract_pdf", return_value=extraction) as extract,
        ):
            page = fetcher.fetch("https://example.com/paper.pdf", timeout_seconds=5)

        self.assertEqual("Paper title", page.title)
        self.assertEqual(extraction.text, page.text)
        extract.assert_called_once()

    def test_accepts_signature_sniffed_pdf_from_generic_content_type(self) -> None:
        headers = Message()
        headers["Content-Type"] = "application/octet-stream"
        fetcher = HttpPageFetcher()
        extraction = PdfExtraction("Useful evidence text.", "", 1)
        with (
            patch.object(fetcher, "_fetch_once", return_value=(200, headers, b"%PDF-1.7")),
            patch("deep_research.providers.extract_pdf", return_value=extraction),
        ):
            page = fetcher.fetch("https://example.com/download", timeout_seconds=5)

        self.assertEqual(extraction.text, page.text)

    def test_rejects_pdf_content_type_without_pdf_signature(self) -> None:
        headers = Message()
        headers["Content-Type"] = "application/pdf"
        fetcher = HttpPageFetcher()
        with patch.object(fetcher, "_fetch_once", return_value=(200, headers, b"not a pdf")):
            with self.assertRaisesRegex(ProviderError, "invalid PDF signature"):
                fetcher.fetch("https://example.com/paper.pdf", timeout_seconds=5)

    def test_applies_separate_html_and_pdf_byte_limits(self) -> None:
        fetcher = HttpPageFetcher(max_bytes=10, max_pdf_bytes=20)
        html_headers = Message()
        html_headers["Content-Type"] = "text/html"
        pdf_headers = Message()
        pdf_headers["Content-Type"] = "application/pdf"

        self.assertEqual(10, fetcher._response_byte_limit("https://example.com", html_headers))
        self.assertEqual(
            20,
            fetcher._response_byte_limit("https://example.com/download", pdf_headers),
        )
        self.assertEqual(
            20,
            fetcher._response_byte_limit(
                "https://example.com/download",
                html_headers,
                b"%PDF-1.7",
            ),
        )

    def test_rejects_pdf_over_its_separate_byte_limit(self) -> None:
        headers = Message()
        headers["Content-Type"] = "application/pdf"
        fetcher = HttpPageFetcher(max_bytes=10, max_pdf_bytes=20)
        raw = b"%PDF-" + (b"x" * 16)
        with patch.object(fetcher, "_fetch_once", return_value=(200, headers, raw)):
            with self.assertRaisesRegex(ProviderError, "PDF exceeds byte limit"):
                fetcher.fetch("https://example.com/paper.pdf", timeout_seconds=5)

    def test_pdf_extraction_waits_on_a_separate_bounded_slot(self) -> None:
        headers = Message()
        headers["Content-Type"] = "application/pdf"
        fetcher = HttpPageFetcher(
            max_concurrent_pdf_extractions=1,
            max_pdf_parse_seconds=0.01,
        )
        self.assertTrue(fetcher._pdf_slots.acquire(blocking=False))
        try:
            with (
                patch.object(
                    fetcher,
                    "_fetch_once",
                    return_value=(200, headers, b"%PDF-1.7"),
                ),
                patch("deep_research.providers.extract_pdf") as extract,
            ):
                with self.assertRaisesRegex(ProviderError, "capacity unavailable"):
                    fetcher.fetch("https://example.com/paper.pdf", timeout_seconds=1)
            extract.assert_not_called()
        finally:
            fetcher._pdf_slots.release()

    def test_pdf_slot_wait_is_charged_to_fetch_deadline(self) -> None:
        headers = Message()
        headers["Content-Type"] = "application/pdf"
        fetcher = HttpPageFetcher(max_pdf_parse_seconds=1)
        fetcher._pdf_slots = Mock()
        fetcher._pdf_slots.acquire.return_value = True
        extraction = PdfExtraction("Useful evidence text.", "", 1)
        with (
            patch.object(
                fetcher,
                "_fetch_once",
                return_value=(200, headers, b"%PDF-1.7"),
            ),
            patch(
                "deep_research.providers.time.monotonic",
                side_effect=[0, 0, 0.04, 0.07],
            ),
            patch(
                "deep_research.providers.extract_pdf",
                return_value=extraction,
            ) as extract,
        ):
            fetcher.fetch("https://example.com/paper.pdf", timeout_seconds=0.1)

        self.assertAlmostEqual(
            0.03,
            extract.call_args.kwargs["timeout_seconds"],
            places=6,
        )
        fetcher._pdf_slots.release.assert_called_once()

    def test_redirect_body_is_not_read_and_request_advertises_pdf(self) -> None:
        headers = Message()
        headers["Location"] = "https://example.com/final"
        response = Mock(status=302, headers=headers)
        connection = Mock()
        connection.getresponse.return_value = response
        fetcher = HttpPageFetcher()
        with (
            patch.object(
                fetcher,
                "_resolve_safe_target",
                return_value=("example.com", 443, ["93.184.216.34"]),
            ),
            patch(
                "deep_research.providers._PinnedHTTPSConnection",
                return_value=connection,
            ),
        ):
            status, _, raw = fetcher._fetch_once("https://example.com/start", 5)

        self.assertEqual(302, status)
        self.assertEqual(b"", raw)
        response.read.assert_not_called()
        request_headers = connection.request.call_args.kwargs["headers"]
        self.assertIn("application/pdf", request_headers["Accept"])

    def test_hinted_fake_pdf_reads_only_signature_prefix(self) -> None:
        headers = Message()
        headers["Content-Type"] = "application/pdf"
        response = Mock(status=200, headers=headers)
        response.read.return_value = b"not a PDF"
        connection = Mock()
        connection.getresponse.return_value = response
        fetcher = HttpPageFetcher()
        with (
            patch.object(
                fetcher,
                "_resolve_safe_target",
                return_value=("example.com", 443, ["93.184.216.34"]),
            ),
            patch(
                "deep_research.providers._PinnedHTTPSConnection",
                return_value=connection,
            ),
        ):
            status, _, raw = fetcher._fetch_once("https://example.com/paper.pdf", 5)

        self.assertEqual(200, status)
        self.assertEqual(b"not a PDF", raw)
        response.read.assert_called_once_with(1024)

    def test_redirect_preserves_server_required_trailing_slash(self) -> None:
        headers = Message()
        headers["Location"] = "/directory/"
        final_headers = Message()
        final_headers["Content-Type"] = "text/plain; charset=utf-8"
        fetcher = HttpPageFetcher()
        with patch.object(
            fetcher,
            "_fetch_once",
            side_effect=[
                (301, headers, b""),
                (200, final_headers, b"Evidence text"),
            ],
        ) as fetch_once:
            page = fetcher.fetch("https://example.com/directory", timeout_seconds=5)

        self.assertEqual(
            "https://example.com/directory/",
            fetch_once.call_args_list[1].args[0],
        )
        self.assertEqual("Evidence text", page.text)

    def test_rejects_private_and_ambiguous_dns(self) -> None:
        private = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 80))]
        mixed = private + [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 80))
        ]
        for addresses in (private, mixed):
            with self.subTest(addresses=addresses):
                with patch("deep_research.providers.socket.getaddrinfo", return_value=addresses):
                    with self.assertRaisesRegex(ProviderError, "unsafe fetch host"):
                        HttpPageFetcher._resolve_safe_target("http://example.com/")

    def test_accepts_and_pins_public_dns(self) -> None:
        public = [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))
        ]
        with patch("deep_research.providers.socket.getaddrinfo", return_value=public):
            target = HttpPageFetcher._resolve_safe_target("https://example.com/path")

        self.assertEqual(("example.com", 443, ["93.184.216.34"]), target)

    def test_rejects_local_alias_credentials_and_nonstandard_port(self) -> None:
        for url in (
            "http://localhost./",
            "http://127.1/",
            "http://user:pass@example.com/",
            "https://example.com:8443/",
        ):
            with self.subTest(url=url):
                with self.assertRaises(ProviderError):
                    HttpPageFetcher._resolve_safe_target(url)


if __name__ == "__main__":
    unittest.main()
