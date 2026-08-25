from __future__ import annotations

import hashlib
import html
import ipaddress
import json
import time
import urllib.error
import urllib.request
from html.parser import HTMLParser
from typing import Any, Protocol
from urllib.parse import quote, urlsplit

from .models import FetchedPage, SearchResult
from .urls import normalize_url


class ProviderError(RuntimeError):
    pass


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
                return self._extract_json(response)
            except ProviderError as exc:
                last_error = exc
                if attempt + 1 < self.max_attempts:
                    time.sleep(min(0.5 * (2**attempt), max(0.0, remaining)))
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
