from __future__ import annotations

import hashlib
import html
import http.client
import ipaddress
import json
import math
import re
import socket
import ssl
import time
import urllib.error
import urllib.request
from html.parser import HTMLParser
from typing import Any, Protocol
from urllib.parse import quote, urljoin, urlsplit, urlunsplit

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


def _validate_api_key(value: str) -> None:
    if not value or any(ord(character) < 33 or ord(character) > 126 for character in value):
        raise ValueError("API key must be non-empty printable ASCII without whitespace")


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, file_pointer, code, message, headers, new_url):
        del request, file_pointer, code, message, headers, new_url
        return None


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
    try:
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", **headers},
            method="POST",
        )
        opener = urllib.request.build_opener(_NoRedirectHandler())
        with opener.open(request, timeout=max(0.1, timeout_seconds)) as response:
            raw = response.read(2_000_001)
            if len(raw) > 2_000_000:
                raise ProviderError("provider response exceeds byte limit")
            data = json.loads(raw)
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
        _validate_api_key(api_key)
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
                remaining = timeout_seconds - (time.monotonic() - started)
                if remaining <= 0:
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
        _validate_api_key(api_key)
        parsed_endpoint = urlsplit(endpoint)
        if (
            parsed_endpoint.scheme != "https"
            or parsed_endpoint.hostname != "api.tavily.com"
            or parsed_endpoint.port not in (None, 443)
            or parsed_endpoint.path != "/search"
            or parsed_endpoint.username
            or parsed_endpoint.password
        ):
            raise ValueError("Tavily endpoint must use the trusted API origin")
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
    def __init__(self, *, max_bytes: int = 2_000_000, max_redirects: int = 3) -> None:
        self.max_bytes = max_bytes
        self.max_redirects = max_redirects

    def fetch(self, url: str, *, timeout_seconds: float) -> FetchedPage:
        started = time.monotonic()
        current_url = self._prepare_fetch_url(url)
        for redirect_count in range(self.max_redirects + 1):
            remaining = timeout_seconds - (time.monotonic() - started)
            if remaining <= 0:
                raise ProviderError("fetch timed out")
            status, headers, raw = self._fetch_once(current_url, remaining)
            if status in {301, 302, 303, 307, 308}:
                if redirect_count >= self.max_redirects:
                    raise ProviderError("too many redirects")
                location = headers.get("Location")
                if not location:
                    raise ProviderError("redirect missing location")
                current_url = self._prepare_fetch_url(urljoin(current_url, location))
                continue
            if not 200 <= status < 300:
                raise ProviderError(f"page HTTP error: {status}")
            content_type = headers.get_content_type()
            if content_type not in {"text/html", "text/plain"}:
                raise ProviderError(f"unsupported content type: {content_type}")
            charset = headers.get_content_charset() or "utf-8"
            final_url = current_url
            break
        else:  # pragma: no cover - loop always exits or raises
            raise ProviderError("too many redirects")
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
    def _prepare_fetch_url(url: str) -> str:
        parsed = urlsplit(url.strip())
        if parsed.scheme.casefold() not in {"http", "https"} or not parsed.hostname:
            raise ValueError("URL must use HTTP(S)")
        if parsed.username or parsed.password:
            raise ProviderError("unsafe fetch credentials")
        return parsed._replace(scheme=parsed.scheme.casefold(), fragment="").geturl()

    def _fetch_once(
        self,
        url: str,
        timeout_seconds: float,
    ) -> tuple[int, http.client.HTTPMessage, bytes]:
        parsed = urlsplit(url)
        host, port, addresses = self._resolve_safe_target(url)
        path = urlunsplit(("", "", parsed.path or "/", parsed.query, ""))
        headers = {
            "Host": host,
            "User-Agent": "TransparentResearch/0.1 (+research; contact=local)",
            "Accept": "text/html,text/plain;q=0.9",
            "Accept-Encoding": "identity",
            "Connection": "close",
        }
        started = time.monotonic()
        last_error: Exception | None = None
        for address in addresses:
            remaining = timeout_seconds - (time.monotonic() - started)
            if remaining <= 0:
                break
            if parsed.scheme == "https":
                connection: http.client.HTTPConnection = _PinnedHTTPSConnection(
                    host,
                    address,
                    port,
                    timeout=max(0.1, remaining),
                )
            else:
                connection = http.client.HTTPConnection(
                    address,
                    port,
                    timeout=max(0.1, remaining),
                )
            try:
                connection.request("GET", path, headers=headers)
                response = connection.getresponse()
                raw = response.read(self.max_bytes + 1)
                return response.status, response.headers, raw
            except (TimeoutError, OSError, http.client.HTTPException) as exc:
                last_error = exc
            finally:
                connection.close()
        if timeout_seconds - (time.monotonic() - started) <= 0:
            raise ProviderError("fetch timed out") from last_error
        error_name = type(last_error).__name__ if last_error else "ConnectionError"
        raise ProviderError(f"fetch failed: {error_name}") from last_error

    @staticmethod
    def _resolve_safe_target(url: str) -> tuple[str, int, list[str]]:
        parsed = urlsplit(url)
        host = parsed.hostname
        if parsed.username or parsed.password:
            raise ProviderError("unsafe fetch credentials")
        if not host or host.rstrip(".").casefold() == "localhost":
            raise ProviderError("unsafe fetch host")
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        expected_port = 443 if parsed.scheme == "https" else 80
        if port != expected_port:
            raise ProviderError("unsafe fetch port")
        try:
            targets = socket.getaddrinfo(
                host,
                port,
                type=socket.SOCK_STREAM,
                proto=socket.IPPROTO_TCP,
            )
        except socket.gaierror as exc:
            raise ProviderError("fetch host resolution failed") from exc
        addresses = list(dict.fromkeys(target[4][0] for target in targets))
        if not addresses or any(
            not ipaddress.ip_address(address).is_global for address in addresses
        ):
            raise ProviderError("unsafe fetch host")
        return host.rstrip("."), port, addresses


class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    def __init__(
        self,
        hostname: str,
        address: str,
        port: int,
        *,
        timeout: float,
    ) -> None:
        super().__init__(
            hostname,
            port,
            timeout=timeout,
            context=ssl.create_default_context(),
        )
        self._pinned_address = address

    def connect(self) -> None:
        sock = socket.create_connection(
            (self._pinned_address, self.port),
            self.timeout,
            self.source_address,
        )
        self.sock = self._context.wrap_socket(sock, server_hostname=self.host)
