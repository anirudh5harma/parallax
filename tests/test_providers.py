import json
import unittest
from unittest.mock import patch

from deep_research.providers import BedrockConverseModel, ProviderError


class BedrockConverseModelTests(unittest.TestCase):
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
                "message": {"content": [{"text": '{"name":"ok","score":1,"items":[]}'}]}
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
        self.assertEqual({"name": "ok", "score": 1, "items": []}, result)
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


if __name__ == "__main__":
    unittest.main()
