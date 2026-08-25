from __future__ import annotations

import threading
from collections import Counter
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit


TRACKING_PARAMETERS = {
    "fbclid",
    "gclid",
    "mc_cid",
    "mc_eid",
    "ref",
}


def normalize_url(url: str) -> str:
    parsed = urlsplit(url.strip())
    scheme = parsed.scheme.casefold()
    if scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("URL must use HTTP(S)")
    host = parsed.hostname.casefold().removeprefix("www.")
    default_port = (scheme == "http" and parsed.port == 80) or (
        scheme == "https" and parsed.port == 443
    )
    netloc = host if parsed.port is None or default_port else f"{host}:{parsed.port}"
    path = parsed.path or "/"
    if path != "/":
        path = path.rstrip("/")
    query = urlencode(
        sorted(
            (key, value)
            for key, value in parse_qsl(parsed.query, keep_blank_values=True)
            if not key.casefold().startswith("utm_")
            and key.casefold() not in TRACKING_PARAMETERS
        ),
        doseq=True,
    )
    normalized = urlunsplit((scheme, netloc, path, query, ""))
    return normalized[:-1] if normalized.endswith("/") and path == "/" else normalized


class UrlRegistry:
    """Atomic URL/content dedup shared by all parallel researchers."""

    def __init__(self) -> None:
        self._exact_urls: set[str] = set()
        self._normalized_urls: set[str] = set()
        self._content_hashes: set[tuple[str, str]] = set()
        self._domain_counts: Counter[str] = Counter()
        self._lock = threading.Lock()

    def claim_url(self, url: str) -> tuple[bool, str]:
        normalized = normalize_url(url)
        exact = urlsplit(url)._replace(fragment="").geturl()
        with self._lock:
            if exact in self._exact_urls or normalized in self._normalized_urls:
                return False, normalized
            self._exact_urls.add(exact)
            self._normalized_urls.add(normalized)
            self._domain_counts[urlsplit(normalized).netloc] += 1
            return True, normalized

    def claim_content(self, content_hash: str, *, scope: str = "global") -> bool:
        key = (scope, content_hash)
        with self._lock:
            if key in self._content_hashes:
                return False
            self._content_hashes.add(key)
            return True

    def domain_counts(self) -> dict[str, int]:
        with self._lock:
            return dict(self._domain_counts)
