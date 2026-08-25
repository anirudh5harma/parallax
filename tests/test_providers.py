import json
import socket
import unittest
from unittest.mock import patch

from deep_research.providers import BedrockConverseModel, HttpPageFetcher, ProviderError


class BedrockConverseModelTests(unittest.TestCase):
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


class HttpPageFetcherSecurityTests(unittest.TestCase):
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

        self.assertEqual(("example.com", 443, "93.184.216.34"), target)

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
