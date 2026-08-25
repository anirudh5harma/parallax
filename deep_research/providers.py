from __future__ import annotations

import hashlib
import html
import ipaddress
import json
import math
import re
import time
import urllib.error
import urllib.request
from html.parser import HTMLParser
from typing import Any, Protocol
from urllib.parse import quote, urlsplit

from .models import FetchedPage, SearchResult
from .urls import normalize_url


class ProviderError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        retryable: bool = True,
        status: int | None = None,
        retry_after: float | None = None,
    ) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.status = status
        self.retry_after = retry_after


class StructuredModel(Protocol):
    def generate_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        schema_name: str,
        schema: dict[str, Any],
        timeout_seconds: float,
    ) -> dict[str, Any]: ...


class SearchClient(Protocol):
    def search(
        self, query: str, *, max_results: int, timeout_seconds: float
    ) -> list[SearchResult]: ...


class PageFetcher(Protocol):
    def fetch(self, url: str, *, timeout_seconds: float) -> FetchedPage: ...


def _post_json(
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    timeout_seconds: float,
) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=max(0.1, timeout_seconds)) as response:
            data = json.load(response)
    except urllib.error.HTTPError as exc:
        retryable = exc.code in {408, 409, 429} or 500 <= exc.code < 600
        retry_after: float | None = None
        detail = ""
        try:
            error_body = json.loads(exc.read(16_384))
            if isinstance(error_body, dict):
                raw_detail = error_body.get("message") or error_body.get("Message")
                if isinstance(raw_detail, str):
                    detail = ": " + " ".join(raw_detail.split())[:500]
        except (OSError, json.JSONDecodeError):
            pass
        raw_retry_after = exc.headers.get("Retry-After") if exc.headers else None
        if raw_retry_after:
            try:
                retry_after = max(0.0, float(raw_retry_after))
            except ValueError:
                pass
        raise ProviderError(
            f"provider HTTP error: {exc.code}{detail}",
            retryable=retryable,
            status=exc.code,
            retry_after=retry_after,
        ) from exc
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise ProviderError(f"request failed: {type(exc).__name__}") from exc
    if not isinstance(data, dict):
        raise ProviderError("provider returned non-object JSON")
    return data


_UNSUPPORTED_BEDROCK_SCHEMA_KEYS = {
    "exclusiveMaximum",
    "exclusiveMinimum",
    "maxItems",
    "maxLength",
    "maximum",
    "minLength",
    "minimum",
    "multipleOf",
}


def _bedrock_schema(value: Any) -> Any:
    """Project a schema onto Bedrock's supported structured-output subset."""
    if isinstance(value, list):
        return [_bedrock_schema(item) for item in value]
    if not isinstance(value, dict):
        return value
    projected: dict[str, Any] = {}
    for key, item in value.items():
        if key in _UNSUPPORTED_BEDROCK_SCHEMA_KEYS:
            continue
        if key == "minItems" and item not in (0, 1):
            continue
        projected[key] = _bedrock_schema(item)
    return projected


def _validate_json_schema(value: Any, schema: dict[str, Any], path: str = "$") -> None:
    """Validate the small JSON Schema subset used by this project."""
    expected = schema.get("type")
    expected_types = expected if isinstance(expected, list) else [expected]
    type_checks = {
        "null": lambda item: item is None,
        "object": lambda item: isinstance(item, dict),
        "array": lambda item: isinstance(item, list),
        "string": lambda item: isinstance(item, str),
        "number": lambda item: isinstance(item, (int, float)) and not isinstance(item, bool),
        "integer": lambda item: isinstance(item, int) and not isinstance(item, bool),
        "boolean": lambda item: isinstance(item, bool),
    }
    if expected is not None and not any(
        item in type_checks and type_checks[item](value) for item in expected_types
    ):
        raise ProviderError(f"model response violates schema at {path}: wrong type")
    if "enum" in schema and value not in schema["enum"]:
        raise ProviderError(f"model response violates schema at {path}: invalid enum")

    if isinstance(value, dict):
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        missing = [key for key in required if key not in value]
        if missing:
            raise ProviderError(
                f"model response violates schema at {path}: missing {missing}"
            )
        if schema.get("additionalProperties") is False:
            extras = set(value) - set(properties)
            if extras:
                raise ProviderError(
                    f"model response violates schema at {path}: extra properties"
                )
        for key, item in value.items():
            child_schema = properties.get(key)
            if isinstance(child_schema, dict):
                _validate_json_schema(item, child_schema, f"{path}.{key}")
    elif isinstance(value, list):
        if len(value) < schema.get("minItems", 0):
            raise ProviderError(f"model response violates schema at {path}: too few items")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            raise ProviderError(f"model response violates schema at {path}: too many items")
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for index, item in enumerate(value):
                _validate_json_schema(item, item_schema, f"{path}[{index}]")
    elif isinstance(value, str):
        if len(value) < schema.get("minLength", 0):
            raise ProviderError(f"model response violates schema at {path}: too short")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            raise ProviderError(f"model response violates schema at {path}: too long")
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            raise ProviderError(f"model response violates schema at {path}: below minimum")
        if "maximum" in schema and value > schema["maximum"]:
            raise ProviderError(f"model response violates schema at {path}: above maximum")
        if "exclusiveMinimum" in schema and value <= schema["exclusiveMinimum"]:
            raise ProviderError(
                f"model response violates schema at {path}: below exclusive minimum"
            )
        if "exclusiveMaximum" in schema and value >= schema["exclusiveMaximum"]:
            raise ProviderError(
                f"model response violates schema at {path}: above exclusive maximum"
            )
        if "multipleOf" in schema:
            quotient = value / schema["multipleOf"]
            if not math.isclose(quotient, round(quotient)):
                raise ProviderError(
                    f"model response violates schema at {path}: invalid multiple"
                )


class BedrockConverseModel:
    def __init__(
        self,
        api_key: str,
        *,
        model_id: str = "us.anthropic.claude-sonnet-4-6",
        region: str = "us-east-1",
        max_tokens: int = 4096,
        max_attempts: int = 2,
    ) -> None:
        if not api_key:
            raise ValueError("API key must not be empty")
        if not model_id or not region:
            raise ValueError("model_id and region must not be empty")
        if not re.fullmatch(r"[a-z]{2}(?:-gov)?-[a-z]+-\d", region):
            raise ValueError("invalid AWS region")
        self.api_key = api_key
        self.model_id = model_id
        self.region = region
        self.max_tokens = max_tokens
        self.max_attempts = max_attempts

    def generate_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        schema_name: str,
        schema: dict[str, Any],
        timeout_seconds: float,
    ) -> dict[str, Any]:
        endpoint = (
            f"https://bedrock-runtime.{self.region}.amazonaws.com/model/"
            f"{quote(self.model_id, safe='.:_-')}/converse"
        )
        payload = {
            "system": [{"text": system_prompt}],
            "messages": [
                {"role": "user", "content": [{"text": user_prompt}]}
            ],
            "inferenceConfig": {"maxTokens": self.max_tokens, "temperature": 0},
            "outputConfig": {
                "textFormat": {
                    "type": "json_schema",
                    "structure": {
                        "jsonSchema": {
                            "name": schema_name,
                            "description": f"Structured output for {schema_name}",
                            "schema": json.dumps(
                                _bedrock_schema(schema),
                                ensure_ascii=False,
                                separators=(",", ":"),
                            ),
                        }
                    },
                }
            },
        }
        last_error: ProviderError | None = None
        started = time.monotonic()
        for attempt in range(self.max_attempts):
            remaining = timeout_seconds - (time.monotonic() - started)
            if remaining <= 0:
                raise ProviderError("model request timed out")
            try:
                response = _post_json(
                    endpoint,
                    payload,
                    {"Authorization": f"Bearer {self.api_key}"},
                    remaining,
                )
                value = self._extract_json(response)
                _validate_json_schema(value, schema)
                return value
            except ProviderError as exc:
                last_error = exc
                if not exc.retryable or attempt + 1 >= self.max_attempts:
                    break
                delay = exc.retry_after if exc.retry_after is not None else 0.5 * (2**attempt)
                time.sleep(min(delay, max(0.0, remaining)))
        raise last_error or ProviderError("model request failed")

    @staticmethod
    def _extract_json(response: dict[str, Any]) -> dict[str, Any]:
        texts: list[str] = []
        output = response.get("output")
        if isinstance(output, dict):
            message = output.get("message")
            if isinstance(message, dict):
                content_items = message.get("content", [])
                if isinstance(content_items, list):
                    for content in content_items:
                        if isinstance(content, dict) and isinstance(
                            content.get("text"), str
                        ):
                            texts.append(content["text"])
        if not texts:
            raise ProviderError("model response contained no text")
        try:
            value = json.loads(texts[-1])
        except json.JSONDecodeError as exc:
            raise ProviderError("model returned invalid JSON") from exc
        if not isinstance(value, dict):
            raise ProviderError("model returned non-object JSON")
        return value


class TavilySearchClient:
    def __init__(
        self,
        api_key: str,
        *,
        endpoint: str = "https://api.tavily.com/search",
    ) -> None:
        if not api_key:
            raise ValueError("API key must not be empty")
        self.api_key = api_key
        self.endpoint = endpoint

    def search(
        self, query: str, *, max_results: int, timeout_seconds: float
    ) -> list[SearchResult]:
        response = _post_json(
            self.endpoint,
            {
                "query": query,
                "search_depth": "basic",
                "max_results": max_results,
                "include_answer": False,
                "include_raw_content": False,
            },
            {"Authorization": f"Bearer {self.api_key}"},
            timeout_seconds,
        )
        results = response.get("results")
        if not isinstance(results, list):
            raise ProviderError("search response missing results")
        parsed: list[SearchResult] = []
        for item in results:
            if not isinstance(item, dict):
                continue
            url = item.get("url")
            title = item.get("title")
            if isinstance(url, str) and isinstance(title, str):
                parsed.append(
                    SearchResult(url=url, title=title, snippet=str(item.get("content", "")))
                )
        return parsed


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.title_parts: list[str] = []
        self._ignored_depth = 0
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        if tag in {"script", "style", "noscript", "svg"}:
            self._ignored_depth += 1
        elif tag == "title":
            self._in_title = True

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript", "svg"} and self._ignored_depth:
            self._ignored_depth -= 1
        elif tag == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        if self._ignored_depth:
            return
        stripped = data.strip()
        if stripped:
            self.parts.append(stripped)
            if self._in_title:
                self.title_parts.append(stripped)


class HttpPageFetcher:
    def __init__(self, *, max_bytes: int = 2_000_000) -> None:
        self.max_bytes = max_bytes

    def fetch(self, url: str, *, timeout_seconds: float) -> FetchedPage:
        normalized = normalize_url(url)
        self._reject_unsafe_host(normalized)
        request = urllib.request.Request(
            normalized,
            headers={
                "User-Agent": "TransparentResearch/0.1 (+research; contact=local)",
                "Accept": "text/html,text/plain;q=0.9",
                "Accept-Encoding": "identity",
            },
        )
        try:
            with urllib.request.urlopen(
                request, timeout=max(0.1, timeout_seconds)
            ) as response:
                content_type = response.headers.get_content_type()
                if content_type not in {"text/html", "text/plain"}:
                    raise ProviderError(f"unsupported content type: {content_type}")
                raw = response.read(self.max_bytes + 1)
                final_url = response.geturl()
                charset = response.headers.get_content_charset() or "utf-8"
        except ProviderError:
            raise
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise ProviderError(f"fetch failed: {type(exc).__name__}") from exc
        if len(raw) > self.max_bytes:
            raise ProviderError("page exceeds byte limit")
        decoded = raw.decode(charset, errors="replace")
        if content_type == "text/html":
            parser = _TextExtractor()
            parser.feed(decoded)
            text = " ".join(parser.parts)
            title = " ".join(parser.title_parts)
        else:
            text = decoded
            title = ""
        text = " ".join(html.unescape(text).split())
        if not text:
            raise ProviderError("page contained no extractable text")
        final_normalized = normalize_url(final_url)
        return FetchedPage(
            url=final_url,
            normalized_url=final_normalized,
            domain=urlsplit(final_normalized).netloc,
            title=title,
            text=text,
            content_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        )

    @staticmethod
    def _reject_unsafe_host(url: str) -> None:
        host = urlsplit(url).hostname
        if not host or host.casefold() == "localhost":
            raise ProviderError("unsafe fetch host")
        try:
            address = ipaddress.ip_address(host)
        except ValueError:
            return
        if not address.is_global:
            raise ProviderError("unsafe fetch host")
